import cv2
import sys
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import filedialog
from collections import deque
import os

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

        # YOLO26x Pose Connection Map (Targeting Hand/Arm segments)
        # 5-7-9 is Left Arm, 6-8-10 is Right Arm
        self.HAND_SKELETON_EDGES = [
            (5, 7), (7, 9),  # Left shoulder to elbow, elbow to wrist
            (6, 8), (8, 10)  # Right shoulder to elbow, elbow to wrist
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

    def get_resize_params(self, w, h, max_w=1280, max_h=720):
        scale = min(max_w/w, max_h/h)
        return scale, (int(w * scale), int(h * scale))

    def process_video(self):
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_replay = []
        
        for i in range(total_frames):
            if not self.running: break
            success, frame = cap.read()
            if not success: break
            
            self.processing_progress = int((i / total_frames) * 100)
            display_frame = frame.copy()
            current_ball_pos = None
            ball_bbox = None

            # 1. Ball Detection (YOLO26x)
            if self.ball_model:
                ball_results = self.ball_model.predict(frame, classes=[32], conf=0.3, verbose=False)
                if len(ball_results[0].boxes) > 0:
                    box = ball_results[0].boxes[0].xyxy[0].cpu().numpy()
                    bx, by = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    current_ball_pos = (bx, by)
                    ball_bbox = box # [x1, y1, x2, y2]

            # 2. Hand Contact & Skeletal Logic
            hand_count = 0
            active_skeletons = [] # Stores (kpts, side) where side is 0 for Left, 1 for Right

            if self.pose_model and ball_bbox is not None:
                pose_results = self.pose_model.predict(frame, verbose=False)
                
                # Expand ball box slightly to detect fingers/wrists entering
                margin = 25
                bx1, by1, bx2, by2 = ball_bbox
                bx1, by1, bx2, by2 = bx1-margin, by1-margin, bx2+margin, by2+margin

                for r in pose_results:
                    if r.keypoints is not None and len(r.keypoints.xy[0]) > 0:
                        kpts = r.keypoints.xy[0].cpu().numpy()
                        
                        # YOLO26x Pose: 9=Left Wrist, 10=Right Wrist
                        # Check Left Hand
                        lx, ly = kpts[9]
                        if lx > bx1 and lx < bx2 and ly > by1 and ly < by2:
                            hand_count += 1
                            active_skeletons.append((kpts, 'left'))
                            
                        # Check Right Hand
                        rx, ry = kpts[10]
                        if rx > bx1 and rx < bx2 and ry > by1 and ry < by2:
                            hand_count += 1
                            active_skeletons.append((kpts, 'right'))

            # 3. Rendering
            # Draw Trajectory
            if current_ball_pos:
                self.path_history.appendleft(current_ball_pos)
                cv2.circle(display_frame, current_ball_pos, 20, (0, 255, 255), 2)
                
                # Hand Count Status
                label = f"HANDS: {hand_count}"
                cv2.rectangle(display_frame, (current_ball_pos[0]-60, current_ball_pos[1]-65), 
                             (current_ball_pos[0]+60, current_ball_pos[1]-35), (0,0,0), -1)
                cv2.putText(display_frame, label, (current_ball_pos[0]-55, current_ball_pos[1]-45), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            # Draw trajectory path
            for j in range(1, len(self.path_history)):
                thickness = max(1, int(6 * (1 - j/len(self.path_history))))
                cv2.line(display_frame, self.path_history[j-1], self.path_history[j], (0, 255, 255), thickness)

            # Draw ONLY the active hand skeletons
            for kpts, side in active_skeletons:
                # Filter edges based on which hand is touching
                edges_to_draw = []
                if side == 'left':
                    edges_to_draw = [(5, 7), (7, 9)] # Shoulder-Elbow-Wrist (Left)
                else:
                    edges_to_draw = [(6, 8), (8, 10)] # Shoulder-Elbow-Wrist (Right)
                
                for p1, p2 in edges_to_draw:
                    pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                    pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                    if pt1[0] > 0 and pt2[0] > 0:
                        # Draw glowing green skeleton for active hand
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), 4)
                        cv2.circle(display_frame, pt2, 6, (255, 255, 255), -1) # Highlight Wrist

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
                cv2.putText(bg, f"SCANNING HAND CONTACT: {self.processing_progress}%", (100, 200), 
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
