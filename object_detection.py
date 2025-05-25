import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Import your depth estimation functions
from depth_estimation import calculate_depth # type: ignore

def perform_object_detection(frame, yolo_model, scaling_factor):
    """Detect objects and calculate distances."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform YOLO object detection
    results = yolo_model.predict(source=frame, save=False)

    # Extract depth map using your custom depth estimation
    depth_map = calculate_depth(frame)

    objects = []
    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        scores = result.boxes.conf
        class_names = yolo_model.names

        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            object_name = class_names[int(cls)]
            
            # Calculate distance based on depth estimation
            object_depth_region = depth_map[y1:y2, x1:x2]
            depth_value = np.median(object_depth_region[object_depth_region > 0])  # Filter valid depth values
            real_distance = depth_value * scaling_factor

            objects.append((object_name, score, real_distance))

            # Draw bounding boxes and labels on the frame
            label = f"{object_name} ({score:.2f}, {real_distance:.2f} cm)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return objects, depth_map
