import cv2

backends = [
    ("CAP_MSMF",  cv2.CAP_MSMF),   # Modern Windows backend
    ("CAP_DSHOW", cv2.CAP_DSHOW),  # Legacy DirectShow
    ("CAP_ANY",   cv2.CAP_ANY),    # Let OpenCV choose
]

def try_open(index, backend):
    name, api = backend
    cap = cv2.VideoCapture(index, api)  # <-- pass the INT constant
    ok = cap.isOpened()
    if ok:
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"[OK] index={index} backend={name} size={w}x{h}")
            cap.release()
            return True
    cap.release()
    print(f"[FAIL] index={index} backend={name}")
    return False

def main():
    found = False
    for idx in range(0, 6):             # try indexes 0..5
        for b in backends:
            if try_open(idx, b):
                found = True
    if not found:
        print("\nNo working camera found via OpenCV. "
              "Close other apps, recheck Windows Camera permissions, and update drivers.")

if __name__ == "__main__":
    main()
