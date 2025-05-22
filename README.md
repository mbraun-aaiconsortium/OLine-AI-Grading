Open Powershell, run as administrator
Navigate to desired directory 

git clone https://github.com/mbraun-aaiconsortium/oline-ai-grading
cd Oline-Ai-Grading

Install Python 3.11

py -3.11 -m venv venv

venv\Scripts\activate
pip install -r requirements.txt

Place video into videos directory, name practice_video.mp4

python main_pipeline.py

Run label-studio.exe

Annotation can be done in Label-Studio, goal is to get a ground_truth.json file into the system

python evaluate.py → compare against your ground_truth.json from Label Studio

On the line above... the ground_truth.json file needs to be generated from label-studio... work is needed to determine how

---Chat GPT Output Below---

🏈 Football AI Grading Platform
This project uses computer vision (YOLOv8-Pose) to automatically analyze football practice film frame-by-frame, grading offensive linemen based on coach-defined technical criteria.

Core Features:
Detects players & pose keypoints from video footage.

Detects ball snap timing to mark play start.

Classifies play type (Run / Pass) and specific run plays.

Grades each offensive lineman's first & second step (placement, direction, timing).

Tracks OL body positioning every 0.5s post-snap.

Detects technical errors (e.g., slow step, pad level too high).

Generates annotated video with overlays.

Exports grading reports in CSV format for coaches.

📂 Project Structure
bash
Copy
Edit
Football_AI_Grading/
├── videos/               # Input practice videos & test images
├── models/               # YOLOv8n-pose model
├── modules/              # Modular components (detection, grading, etc.)
├── outputs/              # Annotated videos & grading reports
├── venv/                 # Python virtual environment
├── main_pipeline.py      # Main orchestrator script
└── README.md             # This file
🚀 How to Run
1. Clone the Repository:
bash
Copy
Edit
git clone https://github.com/yourusername/Football_AI_Grading.git
cd Football_AI_Grading
2. Set up Python Environment:
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt
3. Download YOLOv8-Pose Model:
Place yolov8n-pose.pt inside /models/ directory.

You can get it from Ultralytics Releases.

4. Run the Grading Pipeline:
bash
Copy
Edit
python main_pipeline.py
🛠️ Dependencies
Python 3.9+

Ultralytics YOLOv8

OpenCV

NumPy

Pandas

✅ Current Status:
 YOLOv8-Pose detection working on video frames.

 Snap detection module.

 OL step grading logic.

 Position tracking per 0.5s.

 Visual overlays.

 CSV report exports.

 Advanced play classification (in progress).

 Team-wide grading scalability.

🏗️ How it Works (Simplified Flow)
Detect players & pose keypoints per video frame.

Identify play start by detecting ball snap.

Classify play type (Run/Pass & Run variant).

Grade OL first/second steps (angle, length, timing).

Track OL posture, hip height, base width every 0.5s.

Detect technical errors based on coach schemas.

Output annotated video & grading report.

🤝 Contributions
PRs are welcome. Please fork the repo, create a feature branch, and submit pull requests.

📜 License
MIT License.

👷 Author
Joshua Braun
github.com/yourusername


