import cv2
import time
from ultralytics import YOLO

def main():
    # ✅ Use a lightweight model (change this to your custom model if needed)
    model_path = r"C:\Users\PRATHIKSHA\Downloads\best.pt"  # Replace with yolov8n.pt for speed if needed
    model = YOLO(model_path)

    # ✅ Set confidence threshold
    conf_threshold = 0.5

    # ✅ Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ✅ Reduce webcam resolution for better speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        start_time = time.time()

        # ✅ Use async streaming inference
        results = model(frame, conf=conf_threshold, stream=True)

        # ✅ Loop through results (usually one)
        for r in results:
            annotated_frame = r.plot()

        # ✅ FPS calculation
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ✅ Show the frame
        cv2.imshow("Target Detection", annotated_frame)

        # ✅ Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
