from ultralytics import YOLO
import cv2
import pyttsx3

# Load your trained model (adjust path if needed)
model = YOLO(r'D:\Project\sign_language_yolo\sign_language_project\yolov8n_asl_5k2\weights\best.pt')

# Initialize text-to-speech
engine = pyttsx3.init()
spoken = ""

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("✅ Running sign detection — press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict with YOLOv8
    results = model.predict(source=frame, show=False, conf=0.5)
    boxes = results[0].boxes
    names = model.names
    annotated_frame = results[0].plot()

    if boxes and boxes.cls.numel() > 0:
        class_id = int(boxes.cls[0])
        label = names[class_id]

        # Show label
        cv2.putText(annotated_frame, f"Detected: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Speak once if changed
        if spoken != label:
            engine.say(label)
            engine.runAndWait()
            spoken = label
    else:
        spoken = ""

    # Show webcam output
    cv2.imshow("Sign Language Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
