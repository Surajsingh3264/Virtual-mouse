import ctypes
import os
import sys

# =====================================================================
# 🛠️ WINDOWS MEDIAPIPE BUG PATCH
# Manually supplies the missing 'free' function to prevent ctypes crash
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

# Safety Check: Stop execution early if file is missing to provide a clear error
if not os.path.exists(ABS_MODEL_PATH):
    print(f"\n❌ ERROR: 'hand_landmarker.task' was NOT found in this folder:")
    print(f"👉 {SCRIPT_DIR}\n")
    print("Please download it and place it in the folder next to this script.")
    sys.exit(1)

# Initialize modern MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ABS_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Screen and webcam configuration
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)
frame_w, frame_h = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

# Configuration variables
frame_reduction = 100  # Boundary padding box
pyautogui.FAILSAFE = False

# Open the AI landmarker instance
with HandLandmarker.create_from_options(options) as landmarker:
    print("\n🚀 Virtual Mouse Started Successfully! Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Flip the image horizontally for natural mirroring
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert OpenCV frame to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run detection
        detection_result = landmarker.detect(mp_image)
        
        # Draw active boundary box (visual mouse tracking region)
        cv2.rectangle(frame, (frame_reduction, frame_reduction), 
                      (frame_w - frame_reduction, frame_h - frame_reduction), 
                      (255, 0, 0), 2)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # MediaPipe landmark list structure: Landmark 8 is Index Tip, Landmark 4 is Thumb Tip
                index_pt = hand_landmarks[8]
                thumb_pt = hand_landmarks[4]
                
                # Convert normalized coordinates (0.0 - 1.0) to pixel values
                index_x = int(index_pt.x * frame_w)
                index_y = int(index_pt.y * frame_h)
                thumb_x = int(thumb_pt.x * frame_w)
                thumb_y = int(thumb_pt.y * frame_h)
                
                # Draw visual dots and line connecting index and thumb
                cv2.circle(frame, (index_x, index_y), 8, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 2)
                
                # Map coordinates to screen size with boundary limits
                x_mapped = int((index_x - frame_reduction) * screen_width / (frame_w - 2 * frame_reduction))
                y_mapped = int((index_y - frame_reduction) * screen_height / (frame_h - 2 * frame_reduction))
                
                # Clamp within display resolution limits
                x_mapped = max(0, min(screen_width - 1, x_mapped))
                y_mapped = max(0, min(screen_height - 1, y_mapped))
                
                # Move screen cursor dynamically
                pyautogui.moveTo(x_mapped, y_mapped, duration=0)
                
                # Calculate Euclidean pinch distance for clicking
                distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
                
                # If fingertips touch (close proximity), trigger left click
                if distance < 25:
                    cv2.circle(frame, (index_x, index_y), 15, (0, 0, 255), cv2.FILLED)
                    pyautogui.click()
                    pyautogui.sleep(0.2)  # Delay prevents click-loops
                    
        # Render the tracking display window
        cv2.imshow('Virtual AI Mouse', frame)
        
        # Press 'q' to break execution loop and exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
