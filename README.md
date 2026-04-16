# **🏀 AI Basketball Coach**

An advanced computer vision system designed to analyze basketball shooting mechanics, track ball trajectory, and provide biomechanical feedback using YOLO26x.


# **📽️ Project Demo**

<div align="center">
<img src="OneShot.gif">

<i>Tracking ball trajectory and skeletal biomechanics during a shot</i>

<img src="Feedback-Results.png">

<i>Biomechanical Feedback Report</i>
</div>


# **🚀 Key Features**

High-Precision Detection: Utilizes YOLO26x for state-of-the-art object detection, ensuring consistent tracking of the basketball even at high velocities.

Advanced Pose Estimation: Integrated with YOLO26x-pose for ultra-accurate 2D skeletal landmarking, providing a more robust foundation for biomechanical analysis than standard models.


# **⚙️ Installation & Setup**

1. Navigate to your working folder

```bash
cd [replace this with the folder name]
```

2. Clone github repository

```bash
git clone [https://github.com/Mostafa-Elemam-re/AI-Basketball-Coach.git](https://github.com/Mostafa-Elemam-re/AI-Basketball-Coach.git)
```

3. Navigate to Repository Folder

```bash
cd AI-Basketball-Coach
```

4. Create Environment

```bash
python -m venv [replace this with the name of environment]
```

5. Activate Environment

```bash
[name of environment]\Scripts\activate
```

6. Install Library Requirements

```bash
pip install -r requirements.txt
```

7. Run Model

```bash
python pose_tracker.py
```
