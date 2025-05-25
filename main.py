import json
import os
import cv2
import numpy as np
import torch
import face_recognition 
from ultralytics import YOLO
from utils import speak   # Shared text-to-speech functionality
from object_detection import perform_object_detection
from face_recognition_module import register_new_face, recognize_face

# File to store face data
faces_data_file = "faces_data.json"

# Load existing face data if available
if os.path.exists(faces_data_file):
    with open(faces_data_file, "r") as f:
        faces_data = json.load(f)
else:
    faces_data = {"embeddings": [], "names": []}

def save_faces_data():
    """Save face data to a JSON file."""
    with open(faces_data_file, "w") as f:
        json.dump(faces_data, f)

def main():
    # Load YOLO model for object detection
    yolo_model = YOLO("yolo11m.pt")  # Replace with your YOLO model file

    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera is live. Press 'ESC' to exit.")
    scaling_factor = 0.05  # Adjust based on your depth calculation needs

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Resize frame for faster processing (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Perform object detection and depth estimation
        objects, depth_map = perform_object_detection(small_frame, yolo_model, scaling_factor)

        # Check if a face is detected in the frame
        face_locations = face_recognition.face_locations(small_frame)
        if face_locations:
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(
                    [np.array(embedding) for embedding in faces_data["embeddings"]],
                    face_encoding,
                    tolerance=0.6,
                )

                top, right, bottom, left = [int(coord * 2) for coord in face_location]  # Scale back to original frame

                if True in matches:
                    match_index = matches.index(True)
                    person_name = faces_data["names"][match_index]
                    speak(f"Hello, you have seen this person before. It's {person_name}.")
                else:
                    speak("This is a new person. Registering face.")
                    new_name = f"Person_{len(faces_data['names']) + 1}"
                    register_new_face(frame, (top, right, bottom, left), new_name, faces_data)
                    save_faces_data()

                # Optional: Draw box around face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the video feed with bounding boxes
        cv2.imshow("Camera", frame)

        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

