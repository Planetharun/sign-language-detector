import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import os

DATA_PATH = "data/raw_landmarks.csv"
MODEL_PATH = "data/model.joblib"

def main():
    if not os.path.exists(DATA_PATH):
        print("ERROR: No data found. Run collect_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        print("ERROR: 'label' column missing. Did you modify the CSV?")
        return

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Simple normalization: center & scale by hand size using wrist-to-middle-finger tip distance
    # (This makes distances more consistent across camera zoom.)
    # If you want: leave it out initially; RandomForest is usually robust. But let’s include.
    # Wrist is landmark 0, middle finger tip is 12 (x12,y12,z12).
    def normalize(sample):
        x0, y0, z0 = sample[0], sample[1], sample[2]
        x12, y12, z12 = sample[36], sample[37], sample[38]
        scale = np.linalg.norm([x12 - x0, y12 - y0, z12 - z0]) + 1e-6
        return (sample - np.array([x0, y0, z0] * 21)) / scale

    X_norm = np.apply_along_axis(normalize, 1, X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    dump({"model": clf}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
