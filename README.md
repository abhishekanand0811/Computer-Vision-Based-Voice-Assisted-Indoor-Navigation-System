# 🦯 Computer-Vision-Based-Voice-Assisted-Indoor-Navigation-System
An AI-powered assistive system designed to help **visually impaired individuals** navigate indoor environments using **Object Detection**, **Depth Estimation**, and **Voice Assistance**. The system combines computer vision and large language models (LLMs) to identify obstacles, estimate distances, and provide real-time voice feedback.
## 📦 Repository Structure

This repository contains the core components of the project:

- 🔍 Object Detection (YOLOv8)
- 🌊 Depth Estimation (Monocular Estimation)
- 🧠 LLM-based Contextual Guidance (via Ollama)
- 🗣️ Voice Assistance (Text-to-Speech using `pyttsx3`)
- 🎦 Live Camera Feed & Real-Time Processing

👉 **Facial Recognition** has been modularized and moved to a separate repository. Access it here:  
[🔗 Facial Recognition Module Repository]([https://github.com/yourusername/facial-recognition-module](https://github.com/abhishekanand0811/Facial-Recognition-Module-for-Indoor-Navigation-System))

---

## 🚀 Features

- Real-time Object Detection using YOLOv8
- Monocular Depth Estimation for spatial awareness
- Context-aware voice assistance using LLMs
- Live camera feed for real-time analysis
- Modular design with separate facial recognition system

---

## 🛠️ Tech Stack

| Layer       | Technologies Used |
|------------|-------------------|
| **Backend**  | Python, OpenCV, PyTorch, YOLOv8 |
| **LLM Integration** | Ollama (local LLM inference) |
| **Voice Assistance** | pyttsx3 (Text-to-Speech) |
| **Hardware** | Laptop webcam (Intel i5, Intel Iris Xe) |

---

## 📁 Project Structure

indoor-navigation/
├── main.py # Integration point for all modules
├── object_detection.py # YOLOv8 detection logic
├── depth_estimation.py # Monocular depth estimation
├── llm_assistant.py # LLM integration for guidance
├── voice_output.py # Text-to-speech logic
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/indoor-navigation.git
   cd indoor-navigation
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the main script:

bash
Copy
Edit
python main.py
📷 Sample Output
Real-time object detection + depth estimation with audio feedback.
(Insert a screenshot or GIF here)

🔒 Facial Recognition Module
To enable user identification and personalization, refer to our Facial Recognition Repository. This module provides:

Face detection using Haar cascades

Real-time facial recognition with FaceNet

User authentication & context-aware assistance

🔮 Future Work
Multi-language voice output

Path guidance using AR overlays

GPS + Indoor beacon hybrid support

Enhanced low-light performance

🤝 Contributing
We welcome contributions! Feel free to open issues, suggest features, or raise pull requests.

📜 License
MIT License.
© 2025 Abhishek Anand & Team, Amrita Vishwa Vidyapeetham.

🙌 Acknowledgments
Ultralytics YOLOv8

Intel ISL MiDaS for depth estimation

OpenAI & Ollama for LLM integration

OpenCV, PyTorch, Streamlit









