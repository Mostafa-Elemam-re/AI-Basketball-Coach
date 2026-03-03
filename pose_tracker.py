import cv2
import sys
import numpy as np
import time
import threading
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from collections import deque

class BasketballColorTracker:
    def __init__(self):
        # Configuration
        self.VIDEO_PATH = self._get_video_file()
        
        try:
            # Using YOLOv11n for global object localization
            self.model = YOLO('yolo11n.pt')
            print(f"--- Engine Loaded: Background-Aware Predictive Engine ---")
        except Exception as e:
            print(f"Init Error: {e}")
            sys.exit()
        
        # Tracking & Color State
        self.ball_hsv_bounds = None 
        self.is_calibrated = False
        self.reference_radius = None 
        self.last_known_pos = None
        self.velocity = np.array([0.0, 0.0]) # [dx, dy]
        
        # Background Subtraction Component
        # varThreshold: higher = more stable background, lower = more sensitive to slow motion
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        
        # State Management
        self.processed_replay_buffer = [] 
        self.running = True
        self.mode = "PROCESSING" 
        self.processing_progress = 0
        self.ridx = 0

    def _get_video_file(self):
        root = tk.Tk()
        root.withdraw() 
        file_path = filedialog.askopenfilename(
            title="Select Basketball Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not file_path:
            sys.exit()
        return file_path

    def resize_to_screen(self, image, max_width=1280, max_height=720):
        h, w = image.shape[:2]
        scaling_factor = min(max_width/w, max_height/h)
        if scaling_factor < 1:
            new_size = (int(w * scaling_factor), int(h * scaling_factor))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def _calibrate_all(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2-x1, y2-y1
        self.reference_radius = (w + h) / 4.0
        
        roi = frame[y1+int(h*0.3):y2-int(h*0.3), x1+int(w*0.3):x2-int(w*0.3)]
        if roi.size == 0: return None
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_mean = np.median(hsv_roi[:,:,0])
        s_mean = np.median(hsv_roi[:,:,1])
        v_mean = np.median(hsv_roi[:,:,2])
        
        lower = np.array([max(0, h_mean-25), max(25, s_mean-90), max(25, v_mean-90)])
        upper = np.array([min(180, h_mean+25), 255, 255])
        
        return (lower, upper)

    def _find_arcs_and_estimate_circles(self, frame, hsv_frame, fg_mask, search_area=None):
        """
        Modified Arc Detection: 
        Uses the Foreground Mask (fg_mask) to ignore static background noise.
        """
        if not self.reference_radius or not self.is_calibrated:
            return None

        if search_area:
            sx1, sy1, sx2, sy2 = search_area
            sx1, sy1 = max(0, sx1), max(0, sy1)
            sx2, sy2 = min(frame.shape[1], sx2), min(frame.shape[0], sy2)
            roi_frame = frame[sy1:sy2, sx1:sx2]
            roi_hsv = hsv_frame[sy1:sy2, sx1:sx2]
            roi_fg = fg_mask[sy1:sy2, sx1:sx2] # Crop the foreground mask
            offset = (sx1, sy1)
        else:
            roi_frame = frame
            roi_hsv = hsv_frame
            roi_fg = fg_mask
            offset = (0, 0)

        if roi_frame.size < 100: return None

        # INTERSECTION: We only care about edges that are ALSO moving
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 20, 80)
        
        # MASK THE EDGES: Only look at edges that belong to moving foreground
        moving_edges = cv2.bitwise_and(edges, roi_fg)
        
        contours, _ = cv2.findContours(moving_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            if len(cnt) < 8: continue 
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            
            ratio = radius / self.reference_radius
            if 0.5 < ratio < 1.5:
                dist_from_center = [np.sqrt((p[0][0]-x)**2 + (p[0][1]-y)**2) for p in cnt]
                radius_std = np.std(dist_from_center)
                
                if radius_std < (radius * 0.25):
                    sample_points = cnt[::2]
                    matches = 0
                    for pt in sample_points:
                        px, py = pt[0]
                        if 0 <= py < roi_hsv.shape[0] and 0 <= px < roi_hsv.shape[1]:
                            pixel_hsv = roi_hsv[py, px]
                            if (self.ball_hsv_bounds[0] <= pixel_hsv).all() and \
                               (pixel_hsv <= self.ball_hsv_bounds[1]).all():
                                matches += 1
                    
                    color_score = matches / len(sample_points) if len(sample_points) > 0 else 0
                    if color_score > 0.3:
                        candidates.append({
                            'pos': (int(x + offset[0]), int(y + offset[1])),
                            'radius': radius,
                            'score': color_score * 2 + (1.0 - (radius_std/radius))
                        })
        
        return max(candidates, key=lambda x: x['score']) if candidates else None

    def process_video(self):
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        path_history = deque(maxlen=45)
        temp_replay = []
        
        for i in range(total_frames):
            if not self.running: break
            success, frame = cap.read()
            if not success: break
            
            self.processing_progress = int((i / total_frames) * 100)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # --- NEW: BACKGROUND SUBTRACTION LAYER ---
            # Update the background model and get the foreground mask
            fg_mask = self.back_sub.apply(frame)
            # Clean up the mask (remove small noise points)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            current_ball_pos = None
            detect_method = "NONE"

            predicted_pos = None
            if self.last_known_pos is not None:
                predicted_pos = (int(self.last_known_pos[0] + self.velocity[0]), 
                                 int(self.last_known_pos[1] + self.velocity[1]))

            # --- LAYER 1: YOLO ---
            results = self.model.predict(frame, classes=[32], conf=0.25, verbose=False)
            if results and len(results[0].boxes) > 0:
                box = results[0].boxes[0].xyxy[0].cpu().numpy()
                if not self.is_calibrated:
                    self.ball_hsv_bounds = self._calibrate_all(frame, box)
                    self.is_calibrated = True
                current_ball_pos = (int((box[0]+box[2])/2), int((box[1]+box[3])/2))
                detect_method = "YOLO"

            # --- LAYER 2: TARGETED ARC (Now Background-Filtered) ---
            if current_ball_pos is None:
                s_roi = None
                if predicted_pos:
                    r = int(self.reference_radius * 4) if self.reference_radius else 100
                    s_roi = [predicted_pos[0]-r, predicted_pos[1]-r, predicted_pos[0]+r, predicted_pos[1]+r]
                
                # Pass the foreground mask to the arc detector
                arc = self._find_arcs_and_estimate_circles(frame, hsv_frame, fg_mask, s_roi)
                if arc:
                    current_ball_pos = arc['pos']
                    detect_method = "ARC"

            # --- VELOCITY UPDATE ---
            if current_ball_pos and self.last_known_pos:
                new_v = np.array([current_ball_pos[0] - self.last_known_pos[0], 
                                  current_ball_pos[1] - self.last_known_pos[1]])
                self.velocity = self.velocity * 0.6 + new_v * 0.4
            elif current_ball_pos is None and predicted_pos:
                self.velocity *= 0.96

            # --- VISUALIZATION ---
            display_frame = frame.copy()
            # Overlay foreground mask at 30% opacity for debugging/visual effect
            mask_rgb = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_rgb, 0.3, 0)

            if current_ball_pos:
                self.last_known_pos = current_ball_pos
                path_history.appendleft(current_ball_pos)
                col = (0, 255, 0) if detect_method == "YOLO" else (255, 0, 255)
                cv2.circle(display_frame, current_ball_pos, int(self.reference_radius or 15), col, 2)
                cv2.putText(display_frame, detect_method, (current_ball_pos[0]-20, current_ball_pos[1]-25), 1, 0.8, col, 1)
            elif predicted_pos:
                cv2.circle(display_frame, predicted_pos, int(self.reference_radius or 15), (100, 100, 100), 1)

            for j in range(1, len(path_history)):
                cv2.line(display_frame, path_history[j-1], path_history[j], (0, 255, 255), 2)

            # HUD
            cv2.rectangle(display_frame, (0, 0), (480, 40), (0, 0, 0), -1)
            cv2.putText(display_frame, f"BACKGROUND-FILTERED AI ACTIVE", (15, 25), 1, 1.0, (0, 255, 0), 2)
            temp_replay.append(self.resize_to_screen(display_frame))
        
        cap.release()
        self.processed_replay_buffer = temp_replay
        self.mode = "REPLAY"

    def run(self):
        threading.Thread(target=self.process_video, daemon=True).start()
        while self.running:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if self.mode == "PROCESSING":
                bg = np.zeros((400, 700, 3), dtype=np.uint8)
                cv2.putText(bg, f"LEARNING BACKGROUND: {self.processing_progress}%", (100, 200), 1, 1.2, (0, 255, 0), 2)
                cv2.imshow("AI Basketball Tracker", bg)
            elif self.mode == "REPLAY":
                if self.ridx < len(self.processed_replay_buffer):
                    cv2.imshow("AI Basketball Tracker", self.processed_replay_buffer[self.ridx])
                    self.ridx += 1
                    time.sleep(0.03) 
                else: self.mode = "FINISHED"
            elif self.mode == "FINISHED":
                fin = np.zeros((400, 700, 3), dtype=np.uint8)
                cv2.putText(fin, "ANALYSIS COMPLETE", (250, 200), 1, 1.2, (0, 255, 0), 2)
                cv2.imshow("AI Basketball Tracker", fin)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    BasketballColorTracker().run()