import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
import os
from datetime import datetime

# Settings
DATA_PATH = "data/raw_landmarks.csv"
CLASSES = list("ABCDEFGHIKLMNOPQRSTUVXYZ")  # ASL letters w/o J, W? (J and Z are motion; add later if you want)
# Note: You can include W too if you want: it is static. Z is motion; J is motion.

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def landmarks_to_row(hand_landmarks):
    # 21 points, each has (x, y, z). We’ll flatten to 63 numbers.
    row = []
    for lm in hand_landmarks.landmark:
        row += [lm.x, lm.y, lm.z]
    return row

def ensure_header():
    if not os.path.exists(DATA_PATH):
        cols = []
        for i in range(21):
            cols += [f"x{i}", f"y{i}", f"z{i}"]
        df = pd.DataFrame(columns=["label"] + cols)
        df.to_csv(DATA_PATH, index=False)

def main():
    ensure_header()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access webcam.")
        return

    print("Instructions:")
    print("- Make the handshape for a letter (e.g., A).")
    print("- Press that key on your keyboard (A-Z) to save a sample.")
    print("- Press 'Q' to quit.")
    print("- Try many angles/distances for each letter to help the model learn!")
    saved = 0

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            # Convert color for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Draw & show
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            cv2.putText(frame, "Press letter key to save sample. Q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Collect Data - Sign Language", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break

            if key != 255:  # some key pressed
                char = chr(key).upper()
                if char in CLASSES:
                    # Save current landmarks if present
                    if result.multi_hand_landmarks:
                        hand_landmarks = result.multi_hand_landmarks[0]
                        row = landmarks_to_row(hand_landmarks)
                        df_row = pd.DataFrame([[char] + row])
                        df_row.to_csv(DATA_PATH, mode="a", header=False, index=False)
                        saved += 1
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved sample for '{char}'. Total saved: {saved}")
                    else:
                        print("No hand detected. Try again.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
