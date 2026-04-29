# **🏀 AI Basketball Coach**

An advanced computer vision system designed to analyse basketball shooting mechanics, track ball trajectory, and provide biomechanical feedback using YOLO26x.

---

# **📽️ Project Demo**

<div align="center">
<video 
    src="https://youtube.com/shorts/8NrbtUO-7lg?feature=share"
    controls
    style="max-width: 100%; border-radius: 0px;" 
    muted="muted"
    autoplay 
    loop>
  </video>
    
<i>Tracking ball trajectory and skeletal biomechanics during a shot</i>

<img src="Feedback-Results.png">

<i>Biomechanical Feedback Report</i>
</div>

---

# **🚀 Key Features**

- **High-Precision Detection**: Utilizes YOLO26x for state-of-the-art object detection, ensuring consistent tracking of the basketball even at high velocities.

- **Advanced Pose Estimation**: Integrated with YOLO26x-pose for ultra-accurate 2D skeletal landmarking, providing a more robust foundation for biomechanical analysis than standard models.

---

# **⚙️ Installation & Setup**

**1. Navigate to Working Directory**

```bash
cd [replace this with the directory name]
```

**2. Clone Github Repository**

```bash
git clone https://github.com/Mostafa-Elemam-re/AI-Basketball-Coach.git
```

**3. Navigate to Cloned Repository**

```bash
cd AI-Basketball-Coach
```

**4. Create Environment**

```bash
python -m venv Coach
```

**5. Activate Environment**

```bash
Coach\Scripts\activate
```

**6. Install Library Requirements**

```bash
pip install -r requirements.txt
```

**7. Run Model**

```bash
python pose_tracker.py
```

**8. Run Feedback Generator**

```bash
python feedback-generator.py
```
