import cv2
import time
import numpy as np
import math
import os
import urllib.request
import sys
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- TASK-BASED CONFIGURATION ---
MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# Comprehensive Skeleton Map including Hands and Feet
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24),                   # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),         # Legs
    (27, 29), (29, 31), (27, 31),                   # Left Foot
    (28, 30), (30, 32), (28, 32),                   # Right Foot
    (15, 17), (15, 19), (15, 21), (17, 19),         # Left Hand
    (16, 18), (16, 20), (16, 22), (18, 20)          # Right Hand
]

class BasketballAnalysisSystem:
    def __init__(self):
        # 1. Initialize YOLO11
        try:
            self.ball_model = YOLO('yolo11n.pt') 
            print("--- YOLO11 Engine Loaded ---")
        except Exception as e:
            print(f"YOLO Load Error: {e}")
            sys.exit()

        # 2. Download and Initialize MediaPipe Tasks
        self.download_pose_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        print("--- MediaPipe Tasks API Loaded ---")
        
        # --- SMOOTHING CONFIGURATION ---
        # We use a deque to store the last N positions for a moving average
        self.ball_history = deque(maxlen=5) 
        self.ball_coords = None # Actual (smoothed) tracking center
        
        # Performance Tracking
        self.fps_start_time = 0
        self.fps_counter = 0
        self.current_fps = 0

    def download_pose_model(self):
        if not os.path.exists(MODEL_PATH):
            print("Downloading Pose Task Model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    def get_dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def get_angle(self, a, b, c):
        """Calculates the angle at point b given points a, b, and c."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(rad * 180.0 / np.pi)
        return 360-angle if angle > 180 else angle

    def process_frame(self, frame):
        self.fps_counter += 1
        if time.time() - self.fps_start_time > 1:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()

        h, w, _ = frame.shape

        # 1. YOLO11 Ball Detection
        ball_results = self.ball_model.predict(frame, classes=[32], conf=0.15, verbose=False)
        
        raw_ball_center = None
        current_radius = 20 # Default fallback
        
        for r in ball_results:
            for box in r.boxes:
                # Get raw detection coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_ball_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                current_radius = int(max(x2 - x1, y2 - y1) / 2)
                break # Only track the first/best detection

        # --- SMOOTHING LOGIC ---
        if raw_ball_center:
            self.ball_history.append(raw_ball_center)
            
            # Calculate mean of history to smooth jitter
            avg_x = int(np.mean([p[0] for p in self.ball_history]))
            avg_y = int(np.mean([p[1] for p in self.ball_history]))
            
            # Update public coords (normalized for distance logic)
            self.ball_coords = (avg_x / w, avg_y / h)
            
            # Draw Smoothed Ball UI
            cv2.circle(frame, (avg_x, avg_y), current_radius, (0, 165, 255), 2)
            cv2.circle(frame, (avg_x, avg_y), 2, (0, 0, 255), -1) # Center dot
            cv2.putText(frame, "Basketball", (avg_x - 30, avg_y - current_radius - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        else:
            # If no detection, we don't clear the history immediately to handle brief flickering
            # but we set ball_coords to None if it's been lost for a while
            if len(self.ball_history) > 0:
                self.ball_history.popleft()
            if not self.ball_history:
                self.ball_coords = None

        # 2. MediaPipe Pose Tasks Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if results.pose_landmarks:
            for landmarks in results.pose_landmarks:
                # Draw Skeleton
                for conn in SKELETON_CONNECTIONS:
                    p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), 
                             (int(p2.x*w), int(p2.y*h)), (255, 255, 255), 1)
                
                # Biometric Analysis (Angles)
                joints = [
                    ("L-Elbow", 11, 13, 15, (255, 255, 0)),
                    ("R-Elbow", 12, 14, 16, (0, 255, 0)),
                    ("L-Shoulder", 23, 11, 13, (255, 100, 0)),
                    ("R-Shoulder", 24, 12, 14, (100, 255, 0)),
                    ("L-Hip", 11, 23, 25, (0, 100, 255)),
                    ("R-Hip", 12, 24, 26, (0, 255, 100)),
                    ("L-Knee", 23, 25, 27, (255, 0, 255)),
                    ("R-Knee", 24, 26, 28, (0, 255, 255))
                ]

                for name, idx_a, idx_b, idx_c, color in joints:
                    try:
                        p_a = [landmarks[idx_a].x, landmarks[idx_a].y]
                        p_b = [landmarks[idx_b].x, landmarks[idx_b].y]
                        p_c = [landmarks[idx_c].x, landmarks[idx_c].y]
                        angle = self.get_angle(p_a, p_b, p_c)
                        joint_px = (int(p_b[0]*w), int(p_b[1]*h))
                        cv2.circle(frame, joint_px, 6, color, -1)
                        cv2.putText(frame, f"{int(angle)}", (joint_px[0]+10, joint_px[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    except: continue

                # Set Point Logic (Wrist Tracking)
                if self.ball_coords:
                    r_wrist = [landmarks[16].x, landmarks[16].y]
                    dist_to_ball = self.get_dist(self.ball_coords, r_wrist)
                    if dist_to_ball < 0.15 and self.ball_coords[1] < landmarks[0].y:
                        cv2.putText(frame, "SET POINT REACHED", (w//2-100, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Global HUD
        cv2.rectangle(frame, (5, 5), (350, 60), (0,0,0), -1)
        cv2.putText(frame, f"AI COACH", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {self.current_fps} | Filter: Moving Average (N=5)", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        return frame

def main():
    cap = cv2.VideoCapture(0) 
    tracker = BasketballAnalysisSystem()
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        processed_frame = tracker.process_frame(frame)
        cv2.imshow('AI Basketball Coach', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()