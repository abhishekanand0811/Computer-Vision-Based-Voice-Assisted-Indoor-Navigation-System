import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import math
from ultralytics import YOLO

# --- Pseudo Depth Estimation ---
def calculate_depth(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.GaussianBlur(gray_image, (15, 15), 0)
    depth_map = 255 - depth_map
    depth_map = depth_map / 255.0
    return depth_map

# Load YOLO model
yolo_model = YOLO("yolo11m.pt")  # Replace with your model

# Open live camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

scaling_factor = 100  # Adjust this for real-world distance scaling

print("Live video processing. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Get pseudo-depth
    depth_map = calculate_depth(frame)

    # YOLO prediction
    results = yolo_model.predict(source=frame, save=False, verbose=False)

    # Initialize visualization
    output_image = frame.copy()
    frame_center_x = frame.shape[1] // 2 
    frame_center_y = frame.shape[0] // 2

    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    classes = results[0].boxes.cls
    class_names = yolo_model.names

    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(cls)]

        object_center_x = (x1 + x2) // 2
        object_center_y = (y1 + y2) // 2

        dx = object_center_x - frame_center_x
        dy = object_center_y - frame_center_y

        distance_from_center_px = math.sqrt(dx**2 + dy**2)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        object_depth_region = depth_map[y1:y2, x1:x2]
        non_zero_depth_values = object_depth_region[object_depth_region > 0]

        if len(non_zero_depth_values) > 0:
            depth_value = np.median(non_zero_depth_values)
            real_distance = depth_value * scaling_factor
        else:
            real_distance = -1.0

        label = f"{class_name} ({score:.2f}, {real_distance:.2f} cm)"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.circle(output_image, (object_center_x, object_center_y), 4, (0, 0, 255), -1)
        cv2.line(output_image, (frame_center_x, frame_center_y),
                 (object_center_x, object_center_y), (255, 0, 0), 1)

    # Show the frame
    cv2.imshow("Live Detection", output_image)

    # Exit with ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
