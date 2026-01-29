import cv2
import sys
import numpy as np
import time
from ultralytics import YOLO

class SimpleBallTracker:
    def __init__(self):
        try:
            # Load BOTH models
            self.ball_model = YOLO('yolo11n.pt')
            self.pose_model = YOLO('yolo11n-pose.pt')
            print("--- AI Models Loaded Successfully ---")
        except Exception as e:
            print(f"Initialization Error: {e}")
            sys.exit()
        
        self.camera_index = None
        # FPS calculation variables
        self.prev_time = 0
        self.fps = 0

    def find_correct_camera(self):
        """Checks indices 0-4 and lets the user pick the one that shows the S24."""
        print("\n--- Camera Scanner Started ---")
        print("I will show a preview for each camera found.")
        print("Press 'Y' if you see your phone, or 'N' to skip to the next camera.")
        
        for idx in range(5):
            print(f"Checking Camera Index {idx}...")
            cap = cv2.VideoCapture(idx)
            
            start_time = time.time()
            while time.time() - start_time < 10:
                success, frame = cap.read()
                if success and frame is not None:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"INDEX: {idx}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 'Y' if this is the S24", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, "Press 'N' to skip", (50, 140), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("Camera Scanner", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('y') or key == ord('Y'):
                        print(f"Success! Locked onto Index {idx}")
                        cv2.destroyWindow("Camera Scanner")
                        return cap, idx
                    if key == ord('n') or key == ord('N'):
                        break
                else:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
        print("No camera was selected. Please ensure your phone is linked and try again.")
        sys.exit()

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return int(angle)

    def run(self):
        cap, self.camera_index = self.find_correct_camera()
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"Tracking started on Index {self.camera_index}. Press 'Q' to quit.")

        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            # --- FPS CALCULATION ---
            current_time = time.time()
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                self.fps = 1 / time_diff
            self.prev_time = current_time

            # --- POSE ESTIMATION ---
            pose_results = self.pose_model.predict(frame, conf=0.3, verbose=False)
            for r in pose_results:
                if r.keypoints is not None:
                    for kpts in r.keypoints.xy:
                        pts = kpts.cpu().numpy()
                        links = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
                        for i1, i2 in links:
                            if i1 < len(pts) and i2 < len(pts):
                                p1, p2 = pts[i1], pts[i2]
                                if p1[0] > 0 and p2[0] > 0:
                                    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
                        
                        for i1, i2, i3 in [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]:
                            if all(pts[i][0] > 0 for i in [i1, i2, i3]):
                                angle = self.calculate_angle(pts[i1], pts[i2], pts[i3])
                                cv2.putText(frame, f"{angle}d", (int(pts[i2][0]), int(pts[i2][1])), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- BALL DETECTION ---
            ball_results = self.ball_model.predict(frame, classes=[32], conf=0.25, verbose=False)
            for r in ball_results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    cv2.circle(frame, ((bx1+bx2)//2, (by1+by2)//2), (bx2-bx1)//2, (0, 165, 255), 3)

            # --- DISPLAY FPS ---
            cv2.putText(frame, f"FPS: {int(self.fps)}", (1100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("AI Basketball Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SimpleBallTracker()
    tracker.run()