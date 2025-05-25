import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
import math
import pyttsx3
import ollama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Pseudo Depth Estimation ---
def calculate_depth(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.GaussianBlur(gray_image, (15, 15), 0)
    depth_map = 255 - depth_map
    depth_map = depth_map / 255.0
    return depth_map

# --- Scene Description Generator ---
def generate_scene_description(detections):
    description = []
    for obj in detections:
        label = obj['label']
        distance = obj['distance_cm']
        position = obj['position']

        distance_m = distance / 100
        description.append(f"There is a {label} about {distance_m:.1f} meters to your {position}.")
    
    return " ".join(description)

# --- Local LLM via Ollama ---
def get_llm_guidance_ollama(scene_description):
    prompt = (
        "You are helping a blind person navigate their surroundings. "
        "Here’s what their camera sees:\n"
        f"{scene_description}\n"
        "Give short, clear walking guidance in three sentences. Do not make up anything that is not given in the scene description."
    )

    response = ollama.chat(model='llama3:8b', messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content'].strip()

# --- Text-to-Speech ---
def speak_guidance(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# --- Load YOLO Model ---
yolo_model = YOLO("yolo11m.pt")  # Use your model

# --- Open Camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press SPACE to get navigation guidance or ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    cv2.imshow("Live Camera Feed", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to quit
        break
    elif key == 32:  # SPACE to process current frame
        captured_image = frame.copy()
        print("Processing captured frame...")

        depth_map = calculate_depth(captured_image)
        results = yolo_model.predict(source=captured_image, save=False)

        boxes = results[0].boxes.xyxy
        scores = results[0].boxes.conf
        classes = results[0].boxes.cls
        class_names = yolo_model.names

        scaling_factor = 100
        output_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

        frame_center_x = captured_image.shape[1] // 2
        left_bound = captured_image.shape[1] // 3
        right_bound = 2 * captured_image.shape[1] // 3

        scene_data = []

        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls)]

            object_center_x = (x1 + x2) // 2

            # Determine position (left/center/right)
            if object_center_x < left_bound:
                position = "left"
            elif object_center_x > right_bound:
                position = "right"
            else:
                position = "center"

            object_depth_region = depth_map[y1:y2, x1:x2]
            non_zero_depth_values = object_depth_region[object_depth_region > 0]

            if len(non_zero_depth_values) > 0:
                depth_value = np.median(non_zero_depth_values)
                real_distance = depth_value * scaling_factor
            else:
                real_distance = -1.0

            label = f"{class_name} ({score:.2f}, {real_distance:.2f} cm)"
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            scene_data.append({
                "label": class_name,
                "distance_cm": real_distance,
                "position": position
            })

        output_path = "output_with_distances.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        print(f"\nImage saved to {output_path}")

        scene_text = generate_scene_description(scene_data)
        print("\nScene Description for LLM:")
        print(scene_text)

        guidance = get_llm_guidance_ollama(scene_text)
        print("\nLLM Guidance:")
        print(guidance)

        speak_guidance(guidance)

# Clean up
cap.release()
cv2.destroyAllWindows()