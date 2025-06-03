README.txt – MVSR LAB Project Submission

Project Title
6D Object Pose Estimation with YOLOv11s and FoundationPose

Repository
GitHub: https://github.com/Tilcho/mvsr

1. Project Description
This project implements a 6D object pose estimation. It includes:
- Object detection with YOLOv11s
- Pose estimation using FoundationPose (via Docker)
- Python-based preprocessing and visualization scripts
- Rendered training data and classification pipeline (never got it to work tho...)

2. Environment Setup

A. Python Environment (env)
- Base Python code runs under a virtual environment:
  python -m venv env
  source env/bin/activate
  pip install -r requirements.txt

B. YOLOv11s Environment (env_yolo)
- For YOLOv11s:
  python -m venv env_yolo
  source env_yolo/bin/activate
  pip install -r requirements_yolo.txt

C. FoundationPose via Docker
- The FoundationPose model is run in a Docker container that mirrors the original repository. I added a folder called /morobots, that needs to be in the main folder as well as the adapted datareader.py and runPoseEstimation.py 
- To set it up:
  cd docker_foundationpose
  docker build -t foundationpose .
  docker run --rm -it -v $(pwd):/workspace foundationpose

to run it: python runPoseEstimation.py -img <0...9> 

3. File Structure Overview
mvsr/
├── data/                  # Dataset & CAD models
├── docker_foundationpose/ # Dockerfile and scripts for FoundationPose
├── env/                   # Python virtualenv (not included)
├── env_yolo/              # YOLO environment (not included)
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Scripts for rendering, detection, pose estimation
├── slides_lab.pdf         # Project presentation
├── requirements.txt       # Base Python dependencies
├── requirements_yolo.txt  # YOLO-specific dependencies
├── README.txt             # This file

4. Running the Applications

Step-by-Step

1. Clone the Repository
   git clone https://github.com/Tilcho/mvsr
   cd mvsr

2. Activate Python Environment
   Follow the instructions in section 2.

3. Run Detection
   YOLO detection is triggered via:
   python3 scripts/<filetorun>.py

4. Run Pose Estimation
   Execute FoundationPose inside Docker as described.


5. Used Methods
- Object Detection: YOLOv11s (Dataset created by ChatGPT (somewhat) and labeled with Label Studio)
- Pose Estimation: FoundationPose
- Matching: PnP + Keypoint detection

6. Results Visualization
- Output images and pose overlays are saved in results/
- Includes 3D bounding boxes overlaid on RGB-D inputs and Yolo classification images

7. Sources
- YOLOv11s GitHub: https://github.com/ultralytics/ultralytics
  Edje Electronics https://www.youtube.com/watch?v=r0RspiLG260&t=405s
- FoundationPose GitHub: https://github.com/NVlabs/FoundationPose
- Comments and Code snippets: OpenAI ChatGPT
