import cv2
import time
from ultralytics import YOLO

def main():
    # ✅ Load the trained YOLOv8 model
    model_path = r"C:\Users\PRATHIKSHA\Downloads\best (1).pt" # Replace with yolov8n.pt if needed
    model = YOLO(model_path)

    # ✅ Set confidence threshold for detection
    conf_threshold = 0.5

    # ✅ Open the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ✅ Reduce webcam resolution for better speed (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    print("Press 'q' to quit.")

    while True:
        # Capture each frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        start_time = time.time()

        # ✅ Perform inference on the captured frame
        results = model(frame, conf=conf_threshold, stream=True)

        # ✅ Loop through results (usually one result per frame)
        annotated_frame = frame  # Initialize annotated_frame as the original frame
        for r in results:
            annotated_frame = r.plot()  # Plot the bounding boxes and annotations

        # ✅ FPS calculation
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ✅ Show the annotated frame in a window
        cv2.imshow("Live Target Detection", annotated_frame)

        # ✅ Exit the loop on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()