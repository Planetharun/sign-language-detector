import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import os
import time

MODEL_PATH = "data/model.joblib"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def landmarks_to_row(hand_landmarks):
    row = []
    for lm in hand_landmarks.landmark:
        row += [lm.x, lm.y, lm.z]
    return np.array(row, dtype=np.float32)

def normalize(sample):
    # wrist (0) and middle finger tip (12)
    x0, y0, z0 = sample[0], sample[1], sample[2]
    x12, y12, z12 = sample[36], sample[37], sample[38]
    scale = np.linalg.norm([x12 - x0, y12 - y0, z12 - z0]) + 1e-6
    return (sample - np.array([x0, y0, z0] * 21)) / scale

def main():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] No model found at", MODEL_PATH, "→ run train_model.py first.")
        return

    bundle = load(MODEL_PATH)
    clf = bundle["model"]
    classes = list(clf.classes_)
    print("[INFO] Loaded model. Classes:", classes)

    # Try default DirectShow backend on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[WARN] Camera index 0 failed. Trying index 1...")
        cap.release()
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not access any webcam. Close other apps and check permissions.")
        return

    print("[INFO] Camera opened.")
    cv2.namedWindow("Sign Detector (DEBUG)", cv2.WINDOW_NORMAL)

    last_pred = "..."
    last_time = time.time()
    last_landmark_time = 0
    frame_count = 0
    t0 = time.time()

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t0
                fps = frame_count / max(elapsed, 1e-6)
                print(f"[DEBUG] FPS ~ {fps:.1f}")

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            found = False
            if result.multi_hand_landmarks:
                found = True
                last_landmark_time = time.time()
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                row = landmarks_to_row(hand_landmarks)
                row = normalize(row).reshape(1, -1)

                # predict + top-3
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(row)[0]
                    top_idx = np.argsort(probs)[::-1][:3]
                    top_labels = [classes[i] for i in top_idx]
                    top_probs = [probs[i] for i in top_idx]
                    last_pred = " / ".join([f"{l}:{p*100:.0f}%" for l, p in zip(top_labels, top_probs)])
                else:
                    pred = clf.predict(row)[0]
                    last_pred = str(pred)
            else:
                # If no hand for >1s, show "No hand"
                if time.time() - last_landmark_time > 1.0:
                    last_pred = "No hand"

            cv2.putText(frame, f"Prediction: {last_pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(frame, "Q=quit | Camera idx tried: 0,1", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Sign Detector (DEBUG)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
