# Surakshit Sadak AI 🚦

![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Tech Stack](https://img.shields.io/badge/Tech-Python_|_Streamlit_|_YOLOv8-blue)
![Architecture](https://img.shields.io/badge/Architecture-Edge_AI_Cascade-purple)

**Surakshit Sadak AI** is an interactive, full-stack traffic enforcement and analytics engine. Designed as a proactive solution to road safety, this system processes live or uploaded video feeds to automatically detect high-risk violations, calculate vehicle speeds, and analyze rider safety in real-time.

Originally built as a local edge-inference API, the system has been refactored into a fully interactive **Streamlit** web application with hardware auto-detection, allowing it to run seamlessly on Apple Silicon (MPS), Nvidia GPUs (CUDA), or Cloud CPUs.

🔗 **[Live Demo: Surakshit Sadak AI on Streamlit Cloud]**(https://5beitsa7cbxqqkvnq94cbb.streamlit.app)

## 🧠 Core Architecture & Features

### 1. Dual-Model Cascade Inference
To maximize processing speed without sacrificing accuracy, the system uses a cascade object detection methodology:
* **The Traffic Tracker:** A generalized YOLOv8n model scans the environment for vehicles (Cars, Motorcycles, Buses, Trucks).
* **The Violation Classifier:** When a motorcycle is detected, the bounding box is dynamically cropped and passed to a **custom-trained 100-epoch neural network** specifically designed to detect helmet usage.

### 2. Spatial Velocity Tracking
* Utilizes **ByteTrack** for high-fidelity object tracking across frames.
* Employs perspective transformation (homography matrix mapping) to convert 2D video pixels into a 3D topographical grid, allowing the system to calculate real-world speed (km/h) based on a 5-frame temporal memory buffer.

### 3. Automated Enforcement Engine
The system autonomously identifies and labels multiple high-risk violations on the live dashboard:
* **Speed Tracking:** Calculates live km/h using temporal math.
* **Wrong-Way Detection:** Uses directional vector math on the Y-axis to detect vehicles moving against the flow of traffic.
* **Automated Helmet Checking:** Triggers the secondary cascade model only on two-wheelers to save compute resources.

### 4. Interactive Web Command Center
* Built entirely in **Streamlit** for a seamless, HTML-free frontend.
* Features local `.mp4` video uploading and processing directly in the browser.
* Dynamic hardware auto-routing (`torch.backends.mps` / `cuda` / `cpu`) ensures the app runs efficiently regardless of the deployment environment.


## 🚀 Quick Start Guide

### Prerequisites
* Python 3.9+
* Git

### Local Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/mohit7739/surakshit-sadak-ai.git](https://github.com/mohit7739/surakshit-sadak-ai.git)
   cd surakshit-sadak-ai
