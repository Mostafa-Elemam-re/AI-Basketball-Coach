import cv2
import sys
from ultralytics import YOLO

class SimpleBallTracker:
    def __init__(self):
        try:
            # Load BOTH models
            self.ball_model = YOLO('yolo11n.pt')
            self.pose_model = YOLO('yolo11n-pose.pt')
            print("--- Dual-Model Tracking Initialized (Optimized Scopes) ---")
        except Exception as e:
            print(f"Initialization Error: {e}")
            sys.exit()
        
        self.last_center = None
        self.padding = 150 
        self.miss_count = 0        
        self.max_miss_buffer = 10  

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            success, frame = cap.read()
            if not success:
                break

            h, w, _ = frame.shape
            
            # --- POSE ESTIMATION (Always Full Frame) ---
            # We run this on the whole 'frame' so you are tracked regardless of the ball's position
            pose_results = self.pose_model.predict(frame, conf=0.3, verbose=False, imgsz=640)
            for r in pose_results:
                if r.keypoints is not None:
                    for kpts in r.keypoints.xy:
                        for kp in kpts:
                            kx, ky = int(kp[0]), int(kp[1])
                            if kx > 0 and ky > 0:
                                cv2.circle(frame, (kx, ky), 3, (0, 255, 0), -1)

            # --- BALL DETECTION (Uses ROI/Padding Logic) ---
            search_area = frame
            offset_x, offset_y = 0, 0
            mode_text = "MODE: FULL FRAME SCAN (BALL)"
            mode_color = (0, 0, 255)

            if self.last_center is not None and self.miss_count < self.max_miss_buffer:
                cx, cy = self.last_center
                x1, y1 = max(0, cx - self.padding), max(0, cy - self.padding)
                x2, y2 = min(w, cx + self.padding), min(h, cy + self.padding)
                
                search_area = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
                
                mode_text = f"MODE: ROI SEARCH (BALL) - Missed: {self.miss_count}"
                mode_color = (0, 255, 255) if self.miss_count > 0 else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

            cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

            # Detect the ball within the restricted search_area
            ball_results = self.ball_model.predict(search_area, classes=[32], conf=0.15, verbose=False, imgsz=640)

            found_this_frame = False
            best_ball = None

            for r in ball_results:
                for box in r.boxes:
                    if best_ball is None or box.conf[0] > best_ball.conf[0]:
                        best_ball = box

            if best_ball is not None:
                bx1, by1, bx2, by2 = map(int, best_ball.xyxy[0])
                abs_cx = (bx1 + bx2) // 2 + offset_x
                abs_cy = (by1 + by2) // 2 + offset_y
                radius = (bx2 - bx1) // 2
                
                self.last_center = (abs_cx, abs_cy)
                self.padding = int(radius * 2.5) + 80 # Tighter padding since it doesn't need to fit the person
                
                found_this_frame = True
                self.miss_count = 0 
                cv2.circle(frame, (abs_cx, abs_cy), radius, (0, 165, 255), 3)
                
                conf_val = float(best_ball.conf[0])
                cv2.putText(frame, f"Ball: {conf_val:.2f}", (bx1 + offset_x, by1 + offset_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            if not found_this_frame:
                self.miss_count += 1
                if self.miss_count >= self.max_miss_buffer:
                    self.last_center = None
                    self.miss_count = 0 

            cv2.imshow("Optimized Ball & Pose Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SimpleBallTracker()
    tracker.run()