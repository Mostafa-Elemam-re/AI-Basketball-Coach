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
import csv
import matplotlib.pyplot as plt

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

        # Data Logging (Every Frame)
        self.angle_logs = []
        
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

    def save_data_and_plot(self):
        if not self.angle_logs:
            print("No data recorded to save.")
            return
        
        # 1. Save CSV
        filename = "shooting_analysis.csv"
        keys = self.angle_logs[0].keys()
        try:
            with open(filename, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.angle_logs)
            print(f"--- Data successfully saved to {filename} ---")
        except Exception as e:
            print(f"Error saving CSV: {e}")

        # 2. Generate Graph
        print("Generating Analysis Graph...")
        timestamps = [log["Timestamp"] for log in self.angle_logs]
        
        plt.figure(figsize=(12, 6))
        joints = ["L_ELBOW", "R_ELBOW", "L_KNEE", "R_KNEE", "L_SHOULDER", "R_SHOULDER"]
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1', '#33FFF2']
        
        for joint, color in zip(joints, colors):
            angles = [log[joint] for log in self.angle_logs]
            
            # Filter out None values for plotting (interpolation)
            clean_times = [t for t, a in zip(timestamps, angles) if a is not None]
            clean_angles = [a for a in angles if a is not None]
            
            if clean_angles:
                plt.plot(clean_times, clean_angles, label=joint.replace("_", " "), 
                         color=color, linewidth=2, marker='o', markersize=3, alpha=0.8)

        plt.title("Biomechanical Shot Analysis: Joint Angles vs Time", fontsize=14, fontweight='bold')
        plt.xlabel("Time (Seconds)", fontsize=12)
        plt.ylabel("Angle (Degrees)", fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plt.savefig('joint_angles_over_time.png', dpi=300)
        print("Graph saved as 'joint_angles_over_time.png'")
        plt.show()

    def process_video(self):
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        
        temp_replay = []
        
        for i in range(total_frames):
            if not self.running: break
            success, frame = cap.read()
            if not success: break
            
            self.processing_progress = int((i / total_frames) * 100)
            display_frame = frame.copy()
            
            elapsed_seconds = round(i / fps, 3)
            current_log_entry = {
                "Timestamp": elapsed_seconds,
                "L_ELBOW": None, "R_ELBOW": None,
                "L_KNEE": None, "R_KNEE": None,
                "L_SHOULDER": None, "R_SHOULDER": None
            }

            # 1. Detection
            ball_bbox = None
            if self.ball_model:
                ball_results = self.ball_model.predict(frame, classes=[32], conf=0.3, verbose=False)
                if len(ball_results[0].boxes) > 0:
                    box = ball_results[0].boxes[0].xyxy[0].cpu().numpy()
                    ball_bbox = box
                    # Trajectory dot
                    bx, by = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    self.path_history.appendleft((bx, by))
                    cv2.circle(display_frame, (bx, by), 22, (0, 255, 255), 2)

            # 2. Pose & Angles
            if self.pose_model and ball_bbox is not None:
                pose_results = self.pose_model.predict(frame, verbose=False)
                bx1, by1, bx2, by2 = ball_bbox
                margin = 40
                
                for r in pose_results:
                    if r.keypoints is not None and len(r.keypoints.xy[0]) > 0:
                        kpts = r.keypoints.xy[0].cpu().numpy()
                        # Only log if wrists are near the ball
                        lx, ly = kpts[9]
                        rx, ry = kpts[10]
                        if (bx1-margin < lx < bx2+margin and by1-margin < ly < by2+margin) or \
                           (bx1-margin < rx < bx2+margin and by1-margin < ry < by2+margin):
                            
                            # Render Skeleton
                            for p1, p2 in self.FULL_SKELETON_EDGES:
                                pt1, pt2 = (int(kpts[p1][0]), int(kpts[p1][1])), (int(kpts[p2][0]), int(kpts[p2][1]))
                                if pt1 != (0,0) and pt2 != (0,0):
                                    cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)

                            angles_to_compute = [
                                (5, 7, 9, "L_ELBOW"), (6, 8, 10, "R_ELBOW"),
                                (11, 13, 15, "L_KNEE"), (12, 14, 16, "R_KNEE"),
                                (7, 5, 11, "L_SHOULDER"), (8, 6, 12, "R_SHOULDER")
                            ]

                            for p1, p2, p3, label in angles_to_compute:
                                k1, k2, k3 = kpts[p1], kpts[p2], kpts[p3]
                                if all(k[0] > 0 for k in [k1, k2, k3]):
                                    angle = self.calculate_angle(k1, k2, k3)
                                    current_log_entry[label] = angle
                                    # Render angle text
                                    pos = (int(k2[0]), int(k2[1]))
                                    cv2.putText(display_frame, f"{angle}deg", (pos[0], pos[1]-10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Always add entry to ensure x-axis (time) is complete
            self.angle_logs.append(current_log_entry)

            # Re-render path
            for j in range(1, len(self.path_history)):
                cv2.line(display_frame, self.path_history[j-1], self.path_history[j], (0, 255, 255), 2)

            # Finalize Frame
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
            if key == ord('q'): 
                self.running = False
                self.save_data_and_plot()
            if key == ord('p'): self.paused = not self.paused

            if self.mode == "PROCESSING":
                bg = np.zeros((400, 700, 3), dtype=np.uint8)
                cv2.putText(bg, f"GENERATING KINEMATIC DATA: {self.processing_progress}%", (80, 200), 
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
