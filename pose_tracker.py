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
        self.OUTPUT_VIDEO_PATH = "analyzed_output.mp4"
        
        # Models
        self.ball_model = None
        self.pose_model = None

        if YOLO_AVAILABLE:
            try:
                print("--- Loading YOLO26x High-Precision Models ---")
                self.ball_model = YOLO("yolo26x.pt")
                self.pose_model = YOLO("yolo26x-pose.pt")
                print("--- YOLO26x Models Loaded Successfully ---")
            except Exception as e:
                print(f"--- Model Loading Failed: {e} ---")

        # Tracking State
        self.path_history = deque(maxlen=30)
        self.processed_replay_buffer = [] 
        self.running = True
        self.mode = "PROCESSING"
        self.processing_progress = 0
        self.ridx = 0
        self.paused = False # Set to False so it plays immediately after processing
        self.fps = 60.0 

        # Persistence Logic
        self.active_player_id = None # Tracks the index of the shooter
        self.last_active_kpts = None # Stores keypoints to keep rendering if person is briefly lost

        # Data Logging
        self.angle_logs = []

        self.FULL_SKELETON_EDGES = [
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9), (6, 8), (8, 10),
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
        a, b, c = np.array(a), np.array(b), np.array(c)
        if np.all(a == 0) or np.all(b == 0) or np.all(c == 0):
            return None
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360-angle
        return int(angle)

    def get_resize_params(self, w, h, max_w=1280, max_h=720):
        scale = min(max_w/w, max_h/h)
        return scale, (int(w * scale), int(h * scale))

    def save_data_and_plot(self):
        if not self.angle_logs: return

        filename = "shooting_analysis.csv"
        keys = self.angle_logs[0].keys()
        try:
            with open(filename, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.angle_logs)
            print(f"CSV Saved: {filename}")
        except Exception as e: print(f"Error saving CSV: {e}")

        timestamps = [log["Timestamp"] for log in self.angle_logs]
        plt.figure(figsize=(12, 6))
        joints = ["L_ELBOW", "R_ELBOW", "L_KNEE", "R_KNEE", "L_SHOULDER", "R_SHOULDER"]
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1', '#33FFF2']

        has_plot_data = False
        for joint, color in zip(joints, colors):
            angles = [log[joint] for log in self.angle_logs]
            clean_times = [t for t, a in zip(timestamps, angles) if a is not None]
            clean_angles = [a for a in angles if a is not None]
            if clean_angles:
                plt.plot(clean_times, clean_angles, label=joint.replace("_", " "), color=color)
                has_plot_data = True

        if has_plot_data:
            plt.title("Biomechanical Persistence Analysis")
            plt.xlabel("Seconds")
            plt.ylabel("Degrees")
            plt.legend()
            graph_name = 'joint_angles_over_time.png'
            plt.savefig(graph_name, dpi=300)
            print(f"Graph Saved: {graph_name}")
            plt.show()

    def process_video(self):
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(self.OUTPUT_VIDEO_PATH, fourcc, self.fps, (width, height))

        temp_replay = []

        for i in range(total_frames):
             if not self.running: break
             success, frame = cap.read()
             if not success: break

             self.processing_progress = int((i / total_frames) * 100)
             display_frame = frame.copy()
             elapsed_seconds = round(i / self.fps, 3)
             current_log_entry = {"Timestamp": elapsed_seconds, "L_ELBOW": None, "R_ELBOW": None, "L_KNEE": None, "R_KNEE": None, "L_SHOULDER": None, "R_SHOULDER": None}

             #1. Detect Ball
             ball_center = None
             if self.ball_model:
                 ball_results = self.ball_model.predict(frame, classes=[32], conf=0.3, verbose=False)
                 if len(ball_results) > 0 and len(ball_results[0].boxes) > 0:
                    box = ball_results[0].boxes[0].xyxy[0].cpu().numpy()
                    ball_center = (int((box[0]+box[2])/2), int((box[1]+box[3])/2))
                    self.path_history.appendleft(ball_center)
                    cv2.circle(display_frame, ball_center, 15, (0, 255, 255), 2)
            
            # 2. Detect ALL People (Pose)
             if self.pose_model:
                pose_results = self.pose_model.predict(frame, verbose=False)
                potential_shooters = []

                # First, extract all valid human detections
                for idx, r in enumerate(pose_results):
                    if hasattr(r, 'keypoints') and r.keypoints is not None and len(r.keypoints.xy) > 0:
                        kpts = r.keypoints.xy[0].cpu().numpy()
                        if len(kpts) < 17: continue
                        potential_shooters.append({'id': idx, 'kpts': kpts})

                # Logic: Check if anyone is currently holding the ball
                if ball_center:
                    for shooter in potential_shooters:
                        k = shooter['kpts']
                        lw, rw = k[9], k[10] # Wrists
                        dist_l = math.hypot(ball_center[0]-lw[0], ball_center[1]-lw[1])
                        dist_r = math.hypot(ball_center[0]-rw[0], ball_center[1]-rw[1])

                        # If ball is near hands, update the "Last Person Holding Ball"
                        if dist_l < 60 or dist_r < 60:
                            self.active_player_id = shooter['id']

                # 3. Persistent Target Selection
                # Find the current frame's keypoints for our persistent shooter
                current_shooter_kpts = None
                for shooter in potential_shooters:
                    if shooter['id'] == self.active_player_id:
                        current_shooter_kpts = shooter['kpts']
                        self.last_active_kpts = shooter['kpts'] # Cache for fallback
                        break

                # If the active shooter is found in this frame, or we have cached data
                render_kpts = current_shooter_kpts if current_shooter_kpts is not None else self.last_active_kpts

                if render_kpts is not None:
                    # Draw the persistent skeleton
                    for p1, p2 in self.FULL_SKELETON_EDGES:
                        pt1, pt2 = (int(render_kpts[p1][0]), int(render_kpts[p1][1])), (int(render_kpts[p2][0]), int(render_kpts[p2][1]))
                        if pt1 != (0,0) and pt2 != (0,0):
                            cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)

                    # Biomechanical Calculations
                    angles_to_compute = [
                        (5, 7, 9, "L_ELBOW"), (6, 8, 10, "R_ELBOW"),
                        (11, 13, 15, "L_KNEE"), (12, 14, 16, "R_KNEE"),
                        (7, 5, 11, "L_SHOULDER"), (8, 6, 12, "R_SHOULDER")
                        ]

                    for p1, p2, p3, label in angles_to_compute:
                        angle = self.calculate_angle(render_kpts[p1], render_kpts[p2], render_kpts[p3])
                        if angle:
                            current_log_entry[label] = angle
                            pos = (int(render_kpts[p2][0]), int(render_kpts[p2][1]))
                            cv2.putText(display_frame, f"{angle}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

             self.angle_logs.append(current_log_entry)
 
             for j in range(1, len(self.path_history)):
                cv2.line(display_frame, self.path_history[j-1], self.path_history[j], (0, 255, 255), 2)

             # Write to Output File
             out_writer.write(display_frame)

             # Resize for UI Replay only
             w_orig, h_orig = frame.shape[1], frame.shape[0]
             _, screen_dim = self.get_resize_params(w_orig, h_orig)
             temp_replay.append(cv2.resize(display_frame, screen_dim))

        cap.release()
        out_writer.release()
        print(f"Video Exported Successfully: {self.OUTPUT_VIDEO_PATH}")

        self.processed_replay_buffer = temp_replay
        self.mode = "REPLAY"
        # Auto-save assets once processing is finished
        self.save_data_and_plot()

    def run(self):
        threading.Thread(target=self.process_video, daemon=True).start()
        cv2.namedWindow("AI Basketball Tracker", cv2.WINDOW_NORMAL)

        while self.running:
            frame_delay = 1.0 / self.fps
            start_time = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                self.running = False
            if key == ord('p'): self.paused = not self.paused

            if self.mode == "PROCESSING":
                bg = np.zeros((400, 700, 3), dtype=np.uint8)
                cv2.putText(bg, f"TRACING KINEMATICS: {self.processing_progress}%", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("AI Basketball Tracker", bg)
            elif self.mode == "REPLAY":
                if self.ridx < len(self.processed_replay_buffer):
                    cv2.imshow("AI Basketball Tracker", self.processed_replay_buffer[self.ridx])
                    if not self.paused: 
                        self.ridx += 1
                        time_to_wait = frame_delay - (time.time() - start_time)
                        if time_to_wait > 0:
                            time.sleep(time_to_wait)
                else: 
                    self.ridx = 0

            if cv2.getWindowProperty("AI Basketball Tracker", cv2.WND_PROP_VISIBLE) < 1: break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    BasketballColorTracker().run()
