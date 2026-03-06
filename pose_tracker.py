import cv2
import sys
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import filedialog
from collections import deque
import os

# --- MODEL AVAILABILITY CHECKS ---
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: Ultralytics (YOLO) not installed. Run 'pip install ultralytics'")

class BasketballColorTracker:
    def __init__(self):
        # Initial Setup
        self.VIDEO_PATH = self._get_video_file()
        self.device = "cpu" # Default to CPU; YOLO26 is highly optimized for CPU inference
        
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                # UPGRADE: Switching to YOLO26 Nano (End-to-End Architecture)
                # YOLO26 provides STAL (Small-Target-Aware Labeling) which is 
                # ideal for tracking a basketball across a full court.
                self.yolo_model = YOLO("yolo26x.pt") 
                print(f"--- YOLO26 (Latest Generation) Loaded ---")
            except Exception as e:
                print(f"--- YOLO Init Failed: {e} ---")
        
        # Tracking State
        self.last_known_pos = None
        self.path_history = deque(maxlen=90)
        self.processed_replay_buffer = [] 
        self.running = True
        self.mode = "PROCESSING"
        self.processing_progress = 0
        self.ridx = 0

    def _get_video_file(self):
        root = tk.Tk()
        root.withdraw() 
        file_path = filedialog.askopenfilename(
            title="Select Basketball Video", 
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not file_path: sys.exit()
        return file_path

    def get_resize_params(self, w, h, max_w=1280, max_h=720):
        scale = min(max_w/w, max_h/h)
        return scale, (int(w * scale), int(_h := h * scale)) # Python 3.8+ walrus for brevity

    def process_video(self):
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_replay = []
        
        for i in range(total_frames):
            if not self.running: break
            success, frame = cap.read()
            if not success: break
            
            self.processing_progress = int((i / total_frames) * 100)
            current_ball_pos = None
            
            # 1. Detection Logic (YOLO26 Inference)
            if self.yolo_model:
                # Class 32 is 'sports ball'. 
                # YOLO26's NMS-free head eliminates box jitter during overlaps.
                results = self.yolo_model.predict(frame, classes=[32], conf=0.3, verbose=False)
                
                if len(results[0].boxes) > 0:
                    # Take the highest confidence detection
                    box = results[0].boxes[0].xyxy[0].cpu().numpy()
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    current_ball_pos = (center_x, center_y)

            # 2. Rendering & State Update
            display_frame = frame.copy()
            if current_ball_pos:
                self.last_known_pos = current_ball_pos
                self.path_history.appendleft(current_ball_pos)
                
                # Draw marker
                cv2.circle(display_frame, current_ball_pos, 15, (0, 255, 0), 3)
            
            # Draw Trajectory "Yellow Trail"
            for j in range(1, len(self.path_history)):
                thickness = max(1, int(6 * (1 - j/len(self.path_history))))
                cv2.line(display_frame, self.path_history[j-1], self.path_history[j], (0, 255, 255), thickness)
            
            # Resize for display
            w_orig, h_orig = frame.shape[1], frame.shape[0]
            _, screen_dim = self.get_resize_params(w_orig, h_orig)
            temp_replay.append(cv2.resize(display_frame, screen_dim))
        
        cap.release()
        self.processed_replay_buffer = temp_replay
        self.mode = "REPLAY"

    def run(self):
        # Start processing in a background thread
        threading.Thread(target=self.process_video, daemon=True).start()

        cv2.namedWindow("AI Basketball Tracker", cv2.WINDOW_NORMAL)
        while self.running:
            if self.mode == "PROCESSING":
                bg = np.zeros((400, 700, 3), dtype=np.uint8)
                cv2.putText(bg, f"ANALYZING VIDEO (YOLO26): {self.processing_progress}%", (140, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("AI Basketball Tracker", bg)
            
            elif self.mode == "REPLAY":
                if self.ridx < len(self.processed_replay_buffer):
                    cv2.imshow("AI Basketball Tracker", self.processed_replay_buffer[self.ridx])
                    self.ridx += 1
                    if cv2.waitKey(30) & 0xFF == ord('p'):
                        cv2.waitKey(-1)
                else: 
                    self.ridx = 0 
            
            if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty("AI Basketball Tracker", cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    BasketballColorTracker().run()