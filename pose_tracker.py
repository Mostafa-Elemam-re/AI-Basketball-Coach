import cv2
import sys
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import filedialog
from collections import deque
import os
import math

# -- MODEL AVAILABILITY --
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Error: Ultralytics (YOLO) not installed. Run 'pip install ultralytics'")

class BasketballColorTracker:
    def __init__(self):
        # Initial Setup
        self.VIDEO_PATH = self._get_video_file()
        
        # Models
        self.ball_model = None
        self.pose_model = None
        
        if YOLO_AVAILABLE:
            try:
                # Explicitly using YOLO26x High-Precision architecture
                print("--- Loading YOLO26x High-Precision Models ---")
                self.ball_model = YOLO("yolo26x.pt")
                self.pose_model = YOLO("yolo26x-pose.pt") 
                print("--- YOLO26x Models Loaded Successfully ---")
            except Exception as e:
                print(f"--- Model Loading Failed: {e} ---")
        
        # Tracking State
        self.path_history = deque(maxlen=120)
        self.processed_replay_buffer = [] 
        self.running = True
        self.mode = "PROCESSING"
        self.processing_progress = 0
        self.ridx = 0
        self.paused = True 

        # YOLO26x FULL SKELETON MAP
        self.FULL_SKELETON_EDGES = [
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9),   (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    def _get_video_file(self):
        root = tk.Tk()
        root.withdraw() 
        file_path = filedialog.askopenfilename(
            title="Select Basketball Video", 
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not file_path: sys.exit()
        return file_path

    def calculate_angle(self, a, b, c):
        """Calculates the angle at point B given points A, B, and C."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360-angle
        return int(angle)

    def get_resize_params(self, w, h, max_w=1280, max_h=720):
        scale = min(max_w/w, max_h/h)
        return scale, (int(w * scale), int(h * scale))

    def process_video(self):
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0 # Fallback
        
        temp_replay = []
        
        for i in range(total_frames):
            if not self.running: break
            success, frame = cap.read()
            if not success: break
            
            self.processing_progress = int((i / total_frames) * 100)
            display_frame = frame.copy()
            
            # --- TIMER LOGIC ---
            elapsed_seconds = i / fps
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            milliseconds = int((elapsed_seconds % 1) * 1000)
            timer_text = f"TIME: {minutes:02d}:{seconds:02d}.{milliseconds:03d}"
            
            # Draw Timer Overlay (Top Left)
            cv2.rectangle(display_frame, (10, 10), (280, 50), (0,0,0), -1)
            cv2.putText(display_frame, timer_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 1. Ball Detection
            current_ball_pos = None
            ball_bbox = None
            if self.ball_model:
                ball_results = self.ball_model.predict(frame, classes=[32], conf=0.3, verbose=False)
                if len(ball_results[0].boxes) > 0:
                    box = ball_results[0].boxes[0].xyxy[0].cpu().numpy()
                    bx, by = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    current_ball_pos = (bx, by)
                    ball_bbox = box 

            # 2. Hand Contact & Skeletal Logic
            active_skeletons = [] 
            if self.pose_model and ball_bbox is not None:
                pose_results = self.pose_model.predict(frame, verbose=False)
                margin = 35
                bx1, by1, bx2, by2 = ball_bbox
                bx1, by1, bx2, by2 = bx1-margin, by1-margin, bx2+margin, by2+margin
                for r in pose_results:
                    if r.keypoints is not None and len(r.keypoints.xy[0]) > 0:
                        kpts = r.keypoints.xy[0].cpu().numpy()
                        lx, ly = kpts[9]
                        rx, ry = kpts[10]
                        if (bx1 < lx < bx2 and by1 < ly < by2) or (bx1 < rx < bx2 and by1 < ry < by2):
                            active_skeletons.append(kpts)

            # 3. Rendering Trajectory
            if current_ball_pos:
                self.path_history.appendleft(current_ball_pos)
                cv2.circle(display_frame, current_ball_pos, 22, (0, 255, 255), 2)
            
            for j in range(1, len(self.path_history)):
                thickness = max(1, int(8 * (1 - j/len(self.path_history))))
                cv2.line(display_frame, self.path_history[j-1], self.path_history[j], (0, 255, 255), thickness)

            # 4. Rendering Full Skeleton & Angles
            for kpts in active_skeletons:
                for p1, p2 in self.FULL_SKELETON_EDGES:
                    pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                    pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                    if pt1 != (0,0) and pt2 != (0,0):
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)

                angles_to_compute = [
                    (5, 7, 9, "L ELBOW"), (6, 8, 10, "R ELBOW"),
                    (11, 13, 15, "L KNEE"), (12, 14, 16, "R KNEE"),
                    (7, 5, 11, "L SHOULDER"), (8, 6, 12, "R SHOULDER")
                ]

                for p1, p2, p3, label in angles_to_compute:
                    k1, k2, k3 = kpts[p1], kpts[p2], kpts[p3]
                    if all(k[0] > 0 for k in [k1, k2, k3]):
                        angle = self.calculate_angle(k1, k2, k3)
                        pos = (int(k2[0]), int(k2[1]))
                        cv2.putText(display_frame, f"{angle}deg", (pos[0]+2, pos[1]-8), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        cv2.putText(display_frame, f"{angle}deg", (pos[0], pos[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                for joint_idx in range(5, 17):
                    pt = (int(kpts[joint_idx][0]), int(kpts[joint_idx][1]))
                    if pt != (0,0):
                        cv2.circle(display_frame, pt, 4, (255, 255, 255), -1)

            w_orig, h_orig = frame.shape[1], frame.shape[0]
            _, screen_dim = self.get_resize_params(w_orig, h_orig)
            temp_replay.append(cv2.resize(display_frame, screen_dim))
        
        cap.release()
        self.processed_replay_buffer = temp_replay
        self.mode = "REPLAY"

    def run(self):
        threading.Thread(target=self.process_video, daemon=True).start()
        cv2.namedWindow("AI Basketball Tracker", cv2.WINDOW_NORMAL)
        while self.running:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): self.running = False
            if key == ord('p'): self.paused = not self.paused

            if self.mode == "PROCESSING":
                bg = np.zeros((400, 700, 3), dtype=np.uint8)
                cv2.putText(bg, f"SYNCHRONIZING TIMER: {self.processing_progress}%", (100, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("AI Basketball Tracker", bg)
            elif self.mode == "REPLAY":
                if not self.paused:
                    if self.ridx < len(self.processed_replay_buffer):
                        cv2.imshow("AI Basketball Tracker", self.processed_replay_buffer[self.ridx])
                        self.ridx += 1
                        time.sleep(0.03)
                    else: self.ridx = 0 
                else:
                    if self.ridx < len(self.processed_replay_buffer):
                        cv2.imshow("AI Basketball Tracker", self.processed_replay_buffer[self.ridx])
            
            if cv2.getWindowProperty("AI Basketball Tracker", cv2.WND_PROP_VISIBLE) < 1: break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    BasketballColorTracker().run()
