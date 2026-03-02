import cv2
import sys
import numpy as np
import time
import threading
from collections import deque
from ultralytics import YOLO

class HighPrecisionBasketballCoach:
    def __init__(self):
        # We will try to auto-detect the camera in connect_camera()
        self.CAMERA_SOURCE = 1 
        self.RECORD_DURATION = 15.0 
        
        try:
            self.pose_model = YOLO('yolo11n-pose.pt')
            self.det_model = YOLO('yolo11n.pt')
            print("--- YOLOv11 High-Precision Engines Loaded ---")
        except Exception as e:
            print(f"Init Error: {e}")
            sys.exit()
        
        self.raw_recording_buffer = [] 
        self.processed_replay_buffer = [] 
        self.latest_raw_frame = None
        self.running = True
        self.mode = "LIVE" 
        self.processing_progress = 0
        self.replay_clock = None
        self.ridx = 0

    def connect_camera(self):
        """Robust camera connection logic."""
        # Try current index first, then fallback to 0 or 1
        sources_to_try = [self.CAMERA_SOURCE, 0, 1, 2]
        cap = None
        
        for src in sources_to_try:
            print(f"Attempting to open camera source: {src}...")
            # Try with DSHOW first (faster on Windows)
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.CAMERA_SOURCE = src
                break
            # Try without DSHOW if that fails
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                self.CAMERA_SOURCE = src
                break
                
        if cap is None or not cap.isOpened():
            print("Error: Could not find any active camera source.")
            print("Troubleshooting: 1. Is the webcam plugged in? 2. Is another app (Teams/Zoom) using it?")
            sys.exit()

        # Optimize for 720p capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Successfully connected to Camera {self.CAMERA_SOURCE}")
        return cap

    def camera_grabber(self, cap):
        while self.running:
            success, frame = cap.read()
            if success:
                self.latest_raw_frame = frame
            else:
                # If the camera disconnects during recording
                print("Warning: Camera frame lost.")
                time.sleep(0.01)
        cap.release()

    def process_session(self):
        if not self.raw_recording_buffer:
            self.mode = "FINISHED"
            return

        total_frames = len(self.raw_recording_buffer)
        temp_replay = []
        path_history = deque(maxlen=25)
        SKELETON_CONNECTIONS = [(5,6), (5,7), (7,9), (6,8), (8,10), (5,13), (6,14)]
        
        start_ts = self.raw_recording_buffer[0][0]

        for i, (ts, frame) in enumerate(self.raw_recording_buffer):
            if not self.running: break
            self.processing_progress = int((i / total_frames) * 100)
            
            results = self.det_model.predict(frame, classes=[32], conf=0.25, verbose=False)
            pose_results = self.pose_model.predict(frame, conf=0.40, verbose=False, max_det=1)

            if results and results[0].boxes:
                box = results[0].boxes[results[0].boxes.conf.argmax()]
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                path_history.appendleft(center)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)

            if pose_results and len(pose_results[0].keypoints.xy) > 0:
                kpts = pose_results[0].keypoints.xy[0]
                for s, e in SKELETON_CONNECTIONS:
                    if s < len(kpts) and e < len(kpts):
                        p1, p2 = tuple(map(int, kpts[s])), tuple(map(int, kpts[e]))
                        if p1[0] > 0 and p2[0] > 0:
                            cv2.line(frame, p1, p2, (0, 255, 0), 2)

            for j in range(1, len(path_history)):
                cv2.line(frame, path_history[j-1], path_history[j], (0, 255, 255), 2)

            temp_replay.append((ts - start_ts, frame))
        
        self.processed_replay_buffer = temp_replay
        self.mode = "REPLAY"

    def run(self):
        cap = self.connect_camera()
        threading.Thread(target=self.camera_grabber, args=(cap,), daemon=True).start()

        # UI Loop
        while self.latest_raw_frame is None: 
            print("Waiting for camera buffer to fill...")
            time.sleep(0.5)
            
        print("Starting 15s record in 3 seconds...")
        time.sleep(3)
        
        start_time = time.time()
        self.mode = "LIVE"

        while self.running:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

            if self.mode == "LIVE":
                if self.latest_raw_frame is not None:
                    self.raw_recording_buffer.append((time.perf_counter(), self.latest_raw_frame.copy()))
                    display = self.latest_raw_frame.copy()
                    elapsed = time.time() - start_time
                    cv2.circle(display, (50, 50), 15, (0, 0, 255), -1)
                    cv2.putText(display, f"RECORDING: {elapsed:.1f}s", (80, 60), 1, 1.5, (255, 255, 255), 2)
                    cv2.imshow("AI Coach", display)
                    
                    if elapsed >= self.RECORD_DURATION:
                        self.mode = "PROCESSING"
                        threading.Thread(target=self.process_session, daemon=True).start()

            elif self.mode == "PROCESSING":
                bg = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(bg, f"ANALYZING: {self.processing_progress}%", (450, 360), 1, 2, (0, 255, 0), 2)
                cv2.imshow("AI Coach", bg)

            elif self.mode == "REPLAY":
                if self.replay_clock is None: self.replay_clock = time.perf_counter()
                elapsed = time.perf_counter() - self.replay_clock
                while self.ridx < len(self.processed_replay_buffer) and self.processed_replay_buffer[self.ridx][0] <= elapsed:
                    cv2.imshow("AI Coach", self.processed_replay_buffer[self.ridx][1])
                    self.ridx += 1
                if self.ridx >= len(self.processed_replay_buffer):
                    self.mode = "FINISHED"

            elif self.mode == "FINISHED":
                fin = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(fin, "DONE. PRESS 'Q'.", (480, 360), 1, 2, (255, 255, 255), 2)
                cv2.imshow("AI Coach", fin)

        self.running = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    HighPrecisionBasketballCoach().run()