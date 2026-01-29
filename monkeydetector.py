import cv2
import mediapipe as mp
import os
import numpy as np
from collections import Counter

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)

# --- IMAGE LOADING ---
image_folder = 'monkey'
# "plain" is our neutral state
gestures = ["flipping", "peace", "idea", "thumbsup", "shock", "handsonear", "plain"]
monkey_pics = {}

for g in gestures:
    for ext in [".jpg", ".png"]:
        img_path = os.path.join(image_folder, f"{g}{ext}")
        img = cv2.imread(img_path)
        if img is not None:
            monkey_pics[g] = img
            break

# --- STABILITY & DEFAULT ---
history = []
BUFFER_SIZE = 5 
default_gesture = "plain"
current_img_key = default_gesture

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Start each frame as 'plain'
    detected_now = default_gesture 

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        
        # Helper function for finger state
        def is_open(tip_idx, pip_idx):
            return lm[tip_idx].y < lm[pip_idx].y - 0.02

        index_open = is_open(8, 6)
        mid_open   = is_open(12, 10)
        ring_open  = is_open(16, 14)
        pinky_open = is_open(20, 18)
        thumb_open = lm[4].y < lm[5].y - 0.03

        tips_x = [lm[4].x, lm[8].x, lm[12].x, lm[16].x, lm[20].x]
        
        # --- GESTURE LOGIC ---
        
        # 1. Shock (All fingers extended)
        if index_open and mid_open and ring_open and pinky_open:
            detected_now = "shock"

        # 2. Peace (Index and Middle extended)
        elif index_open and mid_open and not ring_open and not pinky_open:
            detected_now = "peace"
            
        # 3. Flipping (Only Middle extended)
        elif mid_open and not index_open and not ring_open and not pinky_open:
            detected_now = "flipping"
            
        # 4. Idea (Only Index extended)
        elif index_open and not mid_open and not ring_open and not pinky_open:
            detected_now = "idea"

        # 5. Thumbs Up (Only Thumb extended)
        elif thumb_open and not index_open and not mid_open and not ring_open:
            detected_now = "thumbsup"
        
        # 6. Hands on Ear (Positional check)
        at_side_edge = any(x < 0.1 or x > 0.9 for x in tips_x)
        at_head_height = all(l.y < 0.4 for l in lm[4:21:4]) # Check tips height
        if at_side_edge and at_head_height:
            detected_now = "handsonear"

    # --- SMOOTHING ---
    history.append(detected_now)
    if len(history) > BUFFER_SIZE: history.pop(0)
    counts = Counter(history).most_common(1)
    if counts: current_img_key = counts[0][0]

    # --- DISPLAY ---
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w] = frame
    
    # Apply the plain monkey image if it exists in the folder
    if current_img_key in monkey_pics:
        m_img = cv2.resize(monkey_pics[current_img_key], (w, h))
        combined[:, w:] = m_img
    else:
        # Visual fallback if plain.jpg is missing
        cv2.putText(combined, "Missing plain.jpg", (w + 50, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(combined, f"Status: {current_img_key}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Monkey Tracker", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()