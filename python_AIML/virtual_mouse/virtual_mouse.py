import ctypes
import os
import sys
import time

# =====================================================================
# 🛠️ WINDOWS MEDIAPIPE BUG PATCH
# =====================================================================
orig_getattr = ctypes.CDLL.__getattr__
def patched_getattr(self, name):
    if name == 'free':
        try:
            return ctypes.CDLL('msvcrt').free
        except Exception:
            pass
    return orig_getattr(self, name)
ctypes.CDLL.__getattr__ = patched_getattr
# =====================================================================

import cv2
import mediapipe as mp
import pyautogui
import math

# Automatically detect paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
ABS_MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(ABS_MODEL_PATH):
    print(f"\n❌ ERROR: 'hand_landmarker.task' was NOT found in this folder:\n👉 {SCRIPT_DIR}\n")
    sys.exit(1)

# Global variables to pass data between background AI thread and main loop
latest_result = None

def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Initialize modern MediaPipe Tasks API in LIVE_STREAM mode
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ABS_MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=1
)

# Screen and webcam configuration
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)
frame_w, frame_h = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

# Configuration tweaks
frame_reduction = 100  
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  

# Smoothing variables
cloc_x, cloc_y = 0, 0
ploc_x, ploc_y = 0, 0
smoothing = 4  

# Click State Machine (Prevents runaway spam clicks)
left_clicked = False
right_clicked = False

# Open the asynchronous AI landmarker instance
with HandLandmarker.create_from_options(options) as landmarker:
    print("\n🚀 Virtual Mouse with Right Click Active! Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)
        
        # Draw active boundary box
        cv2.rectangle(frame, (frame_reduction, frame_reduction), 
                      (frame_w - frame_reduction, frame_h - frame_reduction), 
                      (255, 0, 0), 2)

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Extract Landmarks: 4=Thumb, 8=Index, 12=Middle
                thumb_pt = hand_landmarks[4]
                index_pt = hand_landmarks[8]
                middle_pt = hand_landmarks[12]
                
                # Convert to pixel coordinates
                thumb_x, thumb_y = int(thumb_pt.x * frame_w), int(thumb_pt.y * frame_h)
                index_x, index_y = int(index_pt.x * frame_w), int(index_pt.y * frame_h)
                middle_x, middle_y = int(middle_pt.x * frame_w), int(middle_pt.y * frame_h)
                
                # Draw tracker dots on fingertips
                cv2.circle(frame, (index_x, index_y), 6, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, (thumb_x, thumb_y), 6, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (middle_x, middle_y), 6, (255, 255, 0), cv2.FILLED)
                
                # Map coordinates (Inverted X)
                x_mapped = int(screen_width - ((index_x - frame_reduction) * screen_width / (frame_w - 2 * frame_reduction)))
                y_mapped = int((index_y - frame_reduction) * screen_height / (frame_h - 2 * frame_reduction))
                
                # Apply smoothing filters
                cloc_x = ploc_x + (x_mapped - ploc_x) / smoothing
                cloc_y = ploc_y + (y_mapped - ploc_y) / smoothing
                
                final_x = max(0, min(screen_width - 1, int(cloc_x)))
                final_y = max(0, min(screen_height - 1, int(cloc_y)))
                
                # Move cursor
                pyautogui.moveTo(final_x, final_y)
                ploc_x, ploc_y = cloc_x, cloc_y
                
                # Calculate pinch distances
                left_pinch_dist = math.hypot(index_x - thumb_x, index_y - thumb_y)
                right_pinch_dist = math.hypot(index_x - middle_x, index_y - middle_y)
                
                # ---- LEFT CLICK CONTROLS ----
                if left_pinch_dist < 25:
                    cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 0, 255), 3)
                    if not left_clicked:
                        pyautogui.click(button='left')
                        left_clicked = True  # Lock clicking until fingers open up
                else:
                    left_clicked = False  # Reset lock when fingers are opened
                    
                # ---- RIGHT CLICK CONTROLS ----
                if right_pinch_dist < 25:
                    cv2.line(frame, (index_x, index_y), (middle_x, middle_y), (0, 165, 255), 3)
                    if not right_clicked:
                        pyautogui.click(button='right')
                        right_clicked = True  # Lock right click until fingers open up
                else:
                    right_clicked = False  # Reset lock when fingers are opened
                    
        cv2.imshow('Virtual AI Mouse', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
