import cv2
import sys
from ultralytics import YOLO

class SimpleBallTracker:
    def __init__(self):
        try:
            # Load the lightweight YOLO11 model
            self.model = YOLO('yolo11n.pt')
            print("--- Simple Ball Detection with Dynamic Padding Initialized ---")
        except Exception as e:
            print(f"Initialization Error: {e}")
            sys.exit()
        
        # Variables to store the position and size for the padding display
        self.last_center = None
        self.padding = 100 # Starting default padding

        # --- CONSISTENCY SETTINGS ---
        self.miss_count = 0 # Counter for consecutive missed frames
        self.max_miss_buffer = 25 # Number of frames to stay in ROI mode before giving up

    def run(self):
        # Open the webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press 'q' to quit.")

        while True:
            success, frame = cap.read()
            if not success:
                break

            h, w, _ = frame.shape
            
            # --- ROI LOGIC SETUP ---
            # Default to full frame
            search_area = frame
            offset_x, offset_y = 0, 0
            mode_text = "MODE: FULL FRAME SCAN"
            mode_color = (0, 0, 255) # Red for full scan

            # --- DISPLAY PADDING BOX & SET SEARCH AREA ---
            if self.last_center is not None and self.miss_count < self.max_miss_buffer:
                cx, cy = self.last_center
                
                # Calculate coordinates and ensure they stay within the frame boundaries
                x1 = max(0, cx - self.padding)
                y1 = max(0, cy - self.padding)
                x2 = min(w, cx + self.padding)
                y2 = min(h, cy + self.padding)
                
                # Set the search area to the cropped padding box
                search_area = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
                mode_text = f"MODE: ROI SEARCH (Missed: {self.miss_count})"
                mode_color = (0, 255, 255) if self.miss_count > 0 else (0, 255, 0)# Green for ROI
                
                # Draw a light gray rectangle to show the "Search Area"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(frame, f"Padding: {self.padding}px (ROI Mode)", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Display the HUD Mode Text
            cv2.putText(frame, mode_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

            # Run detection on the 'search_area'
            # Added augment=True and agnostic_nms=True to improve detection at edges
            results = self.model.predict(search_area, classes=[32], conf=0.15, verbose=False, augment=True, agnostic_nms=True, imgsz=640)

            found_this_frame = False
            for r in results:
                if len(r.boxes) > 0:
                    box = r.boxes[0]
                    x1_det, y1_det, x2_det, y2_det = map(int, box.xyxy[0])
                    
                    # Add offsets to translate crop coordinates back to full frame coordinates
                    center_x = (x1_det + x2_det) // 2 + offset_x
                    center_y = (y1_det + y2_det) // 2 + offset_y
                    radius = (x2_det - x1_det) // 2
                    
                    # Update the center for the next frame
                    self.last_center = (center_x, center_y)
                    
                    # Update padding based on the radius of the ball
                    self.padding = int(radius * 1.5) + 50
                    
                    found_this_frame = True
                    self.miss_count = 0 # Reset miss counter on success

                    # Draw detection using the absolute coordinates
                    cv2.circle(frame, (center_x, center_y), radius, (0, 165, 255), 3)
                    
                    conf = float(box.conf[0])
                    cv2.putText(frame, f"Ball: {conf:.2f}", (x1_det + offset_x, y1_det + offset_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

            # If no ball was found, we clear the center so it goes back to full frame scan
            if not found_this_frame:
                self.miss_count += 1
                if self.miss_count >= self.max_miss_buffer:
                    self.last_center = None
                    self.miss_count = 0

            # Display the resulting frame
            cv2.imshow("Simple Ball Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SimpleBallTracker()
    tracker.run()