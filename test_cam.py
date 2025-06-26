# pylint: disable=no-member, missing-module-docstring

import cv2

# List of possible camera indices and OpenCV backends
camera_indices = [0, 1, 2]
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, 0]

cap = None
opened = False

# Try all combinations of camera index and backend
for index in camera_indices:
    for backend in backends:
        print(f"🔄 Trying camera index {index} with backend: {backend}")
        cap = cv2.VideoCapture(index, backend) if isinstance(backend, int) else cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"✅ Opened webcam at index {index} using backend: {backend}")
            opened = True
            break
    if opened:
        break

if not opened or not cap or not cap.isOpened():
    print("❌ Failed to open webcam with any index/backend.")
    exit()

# Optional: Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("📷 Webcam stream started — press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("❌ Frame not received.")
        continue

    print(f"✅ Frame shape: {frame.shape}")
    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
