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
        self.RECORD_DURATION = 7.0 
        
        try:
            self.model = YOLO('yolo11n.pt')
            self.model.to('cpu') 
            print("--- Precision Timing Engine Initialized ---")
        except Exception as e:
            print(f"Init Error: {e}")
            sys.exit()
        
        # Buffers
        self.raw_recording_buffer = [] # Stores (timestamp, frame)
        self.processed_replay_buffer = [] # Stores (relative_time, frame)
        
        # Shared State
        self.latest_raw_frame = None
        self.running = True
        self.mode = "LIVE" 

    def connect_camera(self):
        cap = cv2.VideoCapture(self.CAMERA_SOURCE, cv2.CAP_DSHOW)
        if not cap.isOpened():
            sys.exit("Error: Could not open camera.")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def camera_grabber(self, cap):
        while self.running:
            success, frame = cap.read()
            if success:
                self.latest_raw_frame = frame
            else:
                time.sleep(0.001)
        cap.release()

    def process_recorded_session(self):
        """Processes frames and maps them to their original capture timing."""
        if not self.raw_recording_buffer:
            self.mode = "FINISHED"
            return

        total_frames = len(self.raw_recording_buffer)
        print(f"Applying AI to {total_frames} frames using precise capture timing...")
        
        self.processed_replay_buffer = []
        path_history = deque(maxlen=30)
        max_miss = 45
        miss_count = 0
        
        # Reference first timestamp to make relative
        start_ts = self.raw_recording_buffer[0][0]

        for ts, frame in self.raw_recording_buffer:
            rel_time = ts - start_ts
            
            results = self.model.predict(
                frame, classes=[32], conf=0.15, verbose=False, imgsz=640, device='cpu'
            )

            best_ball = None
            if results and results[0].boxes:
                boxes = results[0].boxes
                best_idx = boxes.conf.argmax()
                box = boxes[best_idx]
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                best_ball = {'center': ((bx1 + bx2) // 2, (by1 + by2) // 2), 'bbox': (bx1, by1, bx2, by2)}

            if best_ball:
                path_history.appendleft(best_ball['center'])
                miss_count = 0
                cv2.rectangle(frame, (best_ball['bbox'][0], best_ball['bbox'][1]), 
                              (best_ball['bbox'][2], best_ball['bbox'][3]), (0, 165, 255), 2)
            else:
                miss_count += 1
                if miss_count > max_miss:
                    path_history.clear()

            for j in range(1, len(path_history)):
                thickness = max(1, int(np.sqrt(30 / float(j + 1)) * 2))
                cv2.line(frame, path_history[j-1], path_history[j], (0, 255, 255), thickness)

            # Pre-apply UI
            cv2.rectangle(frame, (10, 10), (280, 55), (255, 0, 0), -1)
            cv2.putText(frame, "ANALYSIS REPLAY", (20, 42), 1, 1.5, (255, 255, 255), 2)
            
            self.processed_replay_buffer.append((rel_time, frame))
        
        self.raw_recording_buffer = [] 
        self.mode = "REPLAY"

    def run(self):
        cap = self.connect_camera()
        threading.Thread(target=self.camera_grabber, args=(cap,), daemon=True).start()

        while self.latest_raw_frame is None:
            time.sleep(4)
        
        start_record_time = time.time()
        print("Recording...")

        while self.running:
            if self.mode == "LIVE":
                if self.latest_raw_frame is not None:
                    # Capture with exact high-res timestamp
                    ts = time.perf_counter()
                    self.raw_recording_buffer.append((ts, self.latest_raw_frame.copy()))
                    
                    display = self.latest_raw_frame.copy()
                    elapsed = time.time() - start_record_time
                    remaining = max(0, self.RECORD_DURATION - elapsed)
                    cv2.rectangle(display, (10, 10), (280, 55), (0, 0, 255), -1)
                    cv2.putText(display, f"RECORDING: {remaining:.1f}s", (20, 42), 1, 1.5, (255, 255, 255), 2)
                    cv2.imshow("Tracker Feed", display)
                    
                    if elapsed >= self.RECORD_DURATION:
                        self.mode = "PROCESSING"
                        loading = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(loading, "ANALYZING...", (520, 360), 1, 2.0, (0, 255, 0), 2)
                        cv2.imshow("Tracker Feed", loading)
                        cv2.waitKey(1)
                        self.process_recorded_session()
                        
                        replay_start_clock = time.perf_counter()
                        replay_idx = 0
                
                if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False

            elif self.mode == "REPLAY":
                # SYNC TO RECORDED TIMESTAMPS
                current_replay_time = time.perf_counter() - replay_start_clock
                
                # Check if it's time to show the next frame(s)
                while replay_idx < len(self.processed_replay_buffer) and \
                      self.processed_replay_buffer[replay_idx][0] <= current_replay_time:
                    
                    frame_to_show = self.processed_replay_buffer[replay_idx][1]
                    cv2.imshow("Tracker Feed", frame_to_show)
                    replay_idx += 1
                
                if replay_idx >= len(self.processed_replay_buffer):
                    self.mode = "FINISHED"
                
                if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False

            elif self.mode == "FINISHED":
                final = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(final, "COMPLETE. PRESS 'Q' TO EXIT.", (380, 360), 1, 1.5, (255, 255, 255), 2)
                cv2.imshow("Tracker Feed", final)
                if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False

        cv2.destroyAllWindows()

if __name__ == "__main__":
    SimpleBallTracker().run()