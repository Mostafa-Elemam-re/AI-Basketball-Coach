import cv2
import time
import numpy as np
import math
import os
import urllib.request
import sys
import threading
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION UPGRADED TO 'FULL' ACCURACY ---
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24),                   # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),         # Legs
    (27, 29), (29, 31), (27, 31),                   # Left Foot
    (28, 30), (30, 32), (28, 32),                   # Right Foot
]

class BasketballAnalysisSystem:
    def __init__(self):
        try:
            self.ball_model = YOLO('yolo11n.pt') 
            print("--- YOLO11 Engine Loaded ---")
        except Exception as e:
            print(f"YOLO Load Error: {e}")
            sys.exit()

        self.download_pose_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        # --- STABILITY & TRACKING ---
        self.ball_history = deque(maxlen=5) 
        self.ball_coords = None 
        self.last_raw_center = None
        self.missed_frames = 0
        self.MAX_MISSED = 15  
        self.last_radius = 20
        
        # Advanced Smoothing (EMA)
        self.smooth_landmarks = {} 
        self.angle_history = {} 
        self.alpha_pose = 0.45  
        self.alpha_angle = 0.25 

        # Shot State
        self.is_gathering = False
        
        # Threading
        self.latest_frame = None
        self.processed_data = {"ball": None, "pose": None}
        self.data_lock = threading.Lock()
        self.running = True

        self.fps_start_time = 0
        self.fps_counter = 0
        self.current_fps = 0

    def download_pose_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Downloading HIGH-ACCURACY Pose Model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    def get_dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def get_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(rad * 180.0 / np.pi)
        if angle > 180: angle = 360 - angle
        return angle

    def filter_point(self, idx, new_pt):
        if idx not in self.smooth_landmarks:
            self.smooth_landmarks[idx] = new_pt
            return new_pt
        prev = self.smooth_landmarks[idx]
        smoothed = (new_pt[0] * self.alpha_pose + prev[0] * (1 - self.alpha_pose),
                    new_pt[1] * self.alpha_pose + prev[1] * (1 - self.alpha_pose))
        self.smooth_landmarks[idx] = smoothed
        return smoothed

    def filter_angle(self, key, new_angle):
        if key not in self.angle_history:
            self.angle_history[key] = new_angle
            return new_angle
        smoothed = (new_angle * self.alpha_angle + self.angle_history[key] * (1 - self.alpha_angle))
        self.angle_history[key] = smoothed
        return smoothed

    def ai_worker(self):
        while self.running:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue
            
            frame = self.latest_frame.copy()
            h, w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            pose_results = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
            
            ball_results = self.ball_model.predict(frame, classes=[32], conf=0.15, verbose=False)
            best_ball = None
            if len(ball_results) > 0 and len(ball_results[0].boxes) > 0:
                box = ball_results[0].boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_ball = ((int((x1+x2)/2), int((y1+y2)/2)), int((x2-x1)/2))

            with self.data_lock:
                self.processed_data["ball"] = best_ball
                self.processed_data["pose"] = pose_results

    def render_frame(self, frame):
        self.fps_counter += 1
        if time.time() - self.fps_start_time > 1:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()

        h, w, _ = frame.shape
        knees_bent = False # FIX: Initialize before pose detection block
        
        with self.data_lock:
            ball_data = self.processed_data["ball"]
            pose_data = self.processed_data["pose"]

        # Ball Logic
        if ball_data:
            self.ball_history.append(ball_data[0])
            self.last_radius = ball_data[1]
            self.missed_frames = 0
        else:
            self.missed_frames += 1

        if self.missed_frames < self.MAX_MISSED and self.ball_history:
            avg_x = int(np.mean([p[0] for p in self.ball_history]))
            avg_y = int(np.mean([p[1] for p in self.ball_history]))
            self.ball_coords = (avg_x/w, avg_y/h)
            cv2.circle(frame, (avg_x, avg_y), self.last_radius, (0, 165, 255), 2)
        else:
            self.ball_coords = None

        # Pose & Biometrics
        if pose_data and pose_data.pose_landmarks:
            raw_lms = pose_data.pose_landmarks[0]
            lms = {}
            for i in range(len(raw_lms)):
                lms[i] = self.filter_point(i, (raw_lms[i].x, raw_lms[i].y))

            for c1, c2 in SKELETON_CONNECTIONS:
                pt1 = (int(lms[c1][0]*w), int(lms[c1][1]*h))
                pt2 = (int(lms[c2][0]*w), int(lms[c2][1]*h))
                cv2.line(frame, pt1, pt2, (200, 200, 200), 1)

            joints = [
                ("L-Elbow", 11, 13, 15, (255, 255, 0)),
                ("R-Elbow", 12, 14, 16, (0, 255, 0)),
                ("L-Knee", 23, 25, 27, (255, 0, 255)),
                ("R-Knee", 24, 26, 28, (0, 255, 255))
            ]
            
            for name, a, b, c, color in joints:
                ang = self.get_angle(lms[a], lms[b], lms[c])
                ang = self.filter_angle(name, ang)
                
                bx, by = int(lms[b][0]*w), int(lms[b][1]*h)
                cv2.circle(frame, (bx, by), 5, color, -1)
                cv2.putText(frame, str(int(ang)), (bx+5, by-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
                if "Knee" in name and ang < 160: knees_bent = True

            # Shot Logic
            hands_on = 0
            if self.ball_coords:
                # Check distances between ball and hand/wrist landmarks
                hand_pts = [lms[15], lms[16], lms[19], lms[20]]
                if any(self.get_dist(self.ball_coords, lms[i]) < 0.22 for i in [15, 19]): hands_on += 1
                if any(self.get_dist(self.ball_coords, lms[i]) < 0.22 for i in [16, 20]): hands_on += 1

            if hands_on == 2 and knees_bent:
                self.is_gathering = True
            elif hands_on < 2 and self.ball_coords and self.ball_coords[1] > lms[11][1]:
                # If hands move away and ball is below shoulders, stop gathering
                self.is_gathering = False

            if self.is_gathering:
                cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 8)
                cv2.putText(frame, "GATHERING", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        # HUD
        cv2.rectangle(frame, (5, 5), (280, 80), (0,0,0), -1)
        cv2.putText(frame, f"FPS: {self.current_fps} | ACCURACY: FULL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"KNEES: {'BENT' if knees_bent else 'STRAIGHT'}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if knees_bent else (0,0,255), 1)
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    tracker = BasketballAnalysisSystem()
    thread = threading.Thread(target=tracker.ai_worker, daemon=True)
    thread.start()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        tracker.latest_frame = frame
        processed = tracker.render_frame(frame)
        cv2.imshow('Stable AI Coach', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tracker.running = False
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()