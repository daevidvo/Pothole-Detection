import cv2
import tensorflow as tf
import numpy as np
import os


video = os.path.join(os.getcwd(), "videos/example.mp4")

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    print("video feed not found")
    exit()

# Load TensorFlow model
model = tf.saved_model.load("./model_data")
infer = model.signatures["serving_default"]

# Detect model input size automatically
input_signature = list(infer.structured_input_signature[1].values())[0]
input_height, input_width = input_signature.shape[1:3]

def find_pothole_regions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 180)

    # Find contours (possible pothole regions)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 10000:  # filter out noise and huge blobs
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
    return boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    resized = cv2.resize(frame, (input_width, input_height))
    inputImage = tf.convert_to_tensor(resized, dtype=tf.float32)
    inputImage = inputImage / 255.0
    inputImage = tf.expand_dims(inputImage, axis=0)

    # Run inference
    predictions = infer(inputs=inputImage)
    output = predictions["output_0"].numpy()[0]
    conf = float(output) if output.size == 1 else float(np.max(output))

    if conf > 0.5:
        # Use contour detection to highlight likely potholes
        boxes = find_pothole_regions(frame)
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(frame, f"POTHOLE DETECTED ({conf:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"No Pothole ({conf:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Pothole Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
