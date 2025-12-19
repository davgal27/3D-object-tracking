# POVa-ball-tracking in 3D using multiple cameras
![My GitHub GIF](github_gif.gif)

This project tracks a ball in 3D space using stereo cameras.
---

## Setup

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd pova-ball-tracking
```

2. **Create and activate a virtual environment:**

```powershell
# Windows PowerShell
python -m venv .env
.\.env\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**

```
numpy>=1.26
opencv-python>=4.8
matplotlib>=3.8
pypylon>=2.1
```

---

## How to Run

The workflow consists of three main steps (after calibration):

1. **Detection**  
   Run the detection script to process video frames and extract 3D ball coordinates:

```bash
python src/detection.py
```

2. **Plotting**  
   Visualize the detected coordinates in a static plot:

```bash
python src/plot_xyz.py
```

3. **Animation**  
   Animate the tracked ball trajectory:

```bash
python src/animate_plot.py
```
