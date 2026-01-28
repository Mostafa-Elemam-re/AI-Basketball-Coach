import cv2
import sys
import numpy as np
from ultralytics import YOLO

class SimpleBallTracker:
    def __init__(self):
        try:
            # Load BOTH models
            self.ball_model = YOLO('yolo11n.pt')
            self.pose_model = YOLO('yolo11n-pose.pt')
            print("--- Dual-Model Tracking Initialized (With Skeletons & Angles) ---")
        except Exception as e:
            print(f"Initialization Error: {e}")
            sys.exit()
        
        self.last_center = None
        self.padding = 150 
        self.miss_count = 0        
        self.max_miss_buffer = 10  

    def calculate_angle(self, a, b, c):
        """Calculates the angle at point b given points a, b, and c."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return int(angle)

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
            pose_results = self.pose_model.predict(frame, conf=0.3, verbose=False, imgsz=640)
            for r in pose_results:
                if r.keypoints is not None:
                    for kpts in r.keypoints.xy:
                        pts = kpts.cpu().numpy()
                        
                        # Skeleton connections (index pairs for drawing limbs)
                        skeleton_links = [
                            (5, 7), (7, 9), (6, 8), (8, 10),     # Arms
                            (5, 6), (5, 11), (6, 12), (11, 12), # Torso
                            (11, 13), (13, 15), (12, 14), (14, 16) # Legs
                        ]

                        # Draw limbs
                        for i1, i2 in skeleton_links:
                            p1, p2 = pts[i1], pts[i2]
                            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

                        # Draw all joints
                        for kp in pts:
                            kx, ky = int(kp[0]), int(kp[1])
                            if kx > 0 and ky > 0:
                                cv2.circle(frame, (kx, ky), 4, (0, 0, 255), -1)

                        # Angle logic for elbows and knees
                        joint_triplets = [
                            (5, 7, 9, "L-Elbow"), (6, 8, 10, "R-Elbow"),
                            (11, 13, 15, "L-Knee"), (12, 14, 16, "R-Knee")
                        ]

                        for i1, i2, i3, label in joint_triplets:
                            if all(pts[i][0] > 0 for i in [i1, i2, i3]):
                                p1, p2, p3 = pts[i1], pts[i2], pts[i3]
                                angle = self.calculate_angle(p1, p2, p3)
                                cv2.putText(frame, f"{angle}deg", (int(p2[0]) + 15, int(p2[1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
            ball_results = self.ball_model.predict(search_area, classes=[32], conf=0.25, verbose=False, imgsz=640)

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
                self.padding = int(radius * 2.5) + 80 
                
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