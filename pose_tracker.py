import cv2
import sys
import numpy as np
import time
import threading
from collections import deque
from ultralytics import YOLO

class SimpleBallTracker:
    def __init__(self):
        # --- CONFIGURATION ---
        self.CAMERA_SOURCE = 1 
        
        try:
            # Using yolo11n.pt - scanning full frames (640px)
            self.model = YOLO('yolo11n.pt')
            print(f"--- Full-Frame Scanning Engine Initialized ---")
        except Exception as e:
            print(f"Init Error: {e}")
            sys.exit()
        
        # State Management
        self.last_center = None
        self.path_history = deque(maxlen=30)
        self.miss_count = 0
        self.max_miss_allowed = 45 
        
        # Buffering & Timing
        self.target_delay_seconds = 5
        self.processed_buffer = deque()
        
        # Shared State
        self.new_frame_available = threading.Event()
        self.latest_raw_frame = None
        self.actual_camera_fps = 30.0
        self.running = True
        self.is_playing = False
        self.proc_fps = 0

    def connect_camera(self):
        """Connects to Windows Link and attempts to reduce motion blur via exposure."""
        cap = cv2.VideoCapture(self.CAMERA_SOURCE, cv2.CAP_DSHOW)
        if not cap.isOpened():
            sys.exit("Error: Could not open camera.")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set exposure to reduce motion blur
        cap.set(cv2.CAP_PROP_EXPOSURE, -7) 
        
        return cap

    def camera_grabber(self, cap):
        frame_count = 0
        start_time = time.time()
        while self.running:
            success, frame = cap.read()
            if success:
                self.latest_raw_frame = frame
                self.new_frame_available.set()
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    self.actual_camera_fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
            else:
                time.sleep(0.01)
        cap.release()

    def processing_worker(self):
        prev_time = time.time()
        while self.running:
            if not self.new_frame_available.wait(timeout=1.0):
                continue
            
            self.new_frame_available.clear()
            if self.latest_raw_frame is None:
                continue

            frame = self.latest_raw_frame.copy()
            
            # --- FULL FRAME DETECTION ---
            # Removed ROI padding/cropping to scan the entire screen every time.
            # Using imgsz=640 for better accuracy on full-screen scans.
            results = self.model.predict(
                frame, 
                classes=[32, 37], 
                conf=0.15, 
                verbose=False, 
                imgsz=640 
            )

            best_ball = None
            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    cur_cx, cur_cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                    
                    if best_ball is None or box.conf[0] > best_ball['conf']:
                        best_ball = {
                            'center': (cur_cx, cur_cy), 
                            'bbox': (bx1, by1, bx2, by2), 
                            'conf': float(box.conf[0])
                        }

            if best_ball:
                self.last_center = best_ball['center']
                self.path_history.appendleft(self.last_center)
                self.miss_count = 0
                bx1, by1, bx2, by2 = best_ball['bbox']
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
            else:
                self.miss_count += 1
                if self.miss_count > self.max_miss_allowed:
                    self.last_center = None

            # Render Trail
            for i in range(1, len(self.path_history)):
                thickness = max(1, int(np.sqrt(30 / float(i + 1)) * 2))
                cv2.line(frame, self.path_history[i-1], self.path_history[i], (0, 255, 255), thickness)

            # Performance Stats
            curr_time = time.time()
            self.proc_fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.putText(frame, f"DETECTION: {int(self.proc_fps)} FPS (Full Scan)", (20, 40), 1, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"CAM: {int(self.actual_camera_fps)} FPS", (20, 70), 1, 1.2, (255, 255, 0), 2)

            self.processed_buffer.append(frame)

    def run(self):
        cap = self.connect_camera()
        threading.Thread(target=self.camera_grabber, args=(cap,), daemon=True).start()
        threading.Thread(target=self.processing_worker, daemon=True).start()

        while self.running:
            required_buffer = int(self.actual_camera_fps * self.target_delay_seconds)
            if not self.is_playing:
                if len(self.processed_buffer) >= required_buffer:
                    self.is_playing = True
                else:
                    loading = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(loading, f"BUFFERING FULL SCAN: {len(self.processed_buffer)}/{required_buffer}", (350, 360), 1, 1.5, (255, 255, 255), 2)
                    cv2.imshow("Delayed Tracker Output", loading)
                    if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False
                    continue

            if self.processed_buffer:
                display_frame = self.processed_buffer.popleft()
                cv2.imshow("Delayed Tracker Output", display_frame)
                wait_time = max(1, int(1000 / max(1, self.actual_camera_fps)))
                if cv2.waitKey(wait_time) & 0xFF == ord('q'): 
                    self.running = False
            else:
                self.is_playing = False

        cv2.destroyAllWindows()

if __name__ == "__main__":
    SimpleBallTracker().run()