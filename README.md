# FormIQ — AI Gym Exercise Form Assessment

> Real-time pose estimation, biomechanical error detection, and progressive overload tracking.  
> Built for the University of Johannesburg Biometrics (IT28X07) research project.

---

## Project Structure

```
gym_aq/
├── app.py               ← Glassmorphic Streamlit frontend (run this)
├── main.py              ← CLI live webcam mode
├── requirements.txt
├── modules/
│   ├── __init__.py
│   ├── extractor.py     ← Module 1: MediaPipe pose keypoint extraction
│   ├── segmentor.py     ← Module 2: Rep segmentation via hip trajectory
│   ├── features.py      ← Module 3: Biomechanical feature computation
│   ├── classifier.py    ← Module 4: SVM error classification & scoring
│   └── tracker.py       ← Module 5 (NEW): Session tracker & overload signals
├── models/              ← Trained .pkl classifier files (place here after training)
├── logs/                ← Session JSON exports
└── data/                ← Fitness-AQA dataset clips
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Streamlit app (demo mode — no camera/model required)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Use **"Simulate Rep"** in the sidebar to generate demo data.

### 3. Live webcam mode (CLI)

```bash
python main.py --exercise BackSquat --athlete yourname
```

Press **Q** to quit. Session summary is saved to `logs/`.

---

## Pipeline

| Module                  | Description                                                                    |
| ----------------------- | ------------------------------------------------------------------------------ |
| **1 – Extractor**       | MediaPipe Pose · 33 3D landmarks · visibility filtering                        |
| **2 – Segmentor**       | Hip midpoint trajectory · `scipy.signal.find_peaks` · Savitzky-Golay smoothing |
| **3 – Features**        | Joint angles · symmetry ratios · depth · temporal smoothness                   |
| **4 – Classifier**      | Per-error SVM (RBF) · Fitness-AQA labels · 0–100 quality score                 |
| **5 – Tracker** _(NEW)_ | Rep history · trend regression · overload readiness · session JSON export      |

---

## Training on Fitness-AQA

1. Request dataset access: https://github.com/ParitoshParmar/Fitness-AQA
2. Place clips in `data/`
3. Extract features from annotated clips using `modules/features.py`
4. Train and save:

```python
from modules.classifier import FormClassifier
clf = FormClassifier("BackSquat")
clf.train(X_train, y_dict)   # y_dict = {error_name: binary_array}
clf.save("models/backsquat_svm.pkl")
```

5. Load at runtime:

```python
clf.load("models/backsquat_svm.pkl")
```

Without a trained model, the system falls back to heuristic .

---

## Module 5 — Session Tracker (New Feature)

The tracker adds **longitudinal intelligence** beyond single-rep scoring:

- **Trend detection**: Linear regression over rep scores → flags fatigue-induced form breakdown
- **Overload readiness**: 3 consecutive reps ≥80 → system signals it is safe to increase load
- **Plateau detection**: Scores clustering within ±5 pts → suggests technique cue or load change
- **Error frequency map**: Identifies your most consistent mistake across the session
- **Coaching tip generator**: Maps top error to a targeted corrective drill
- **JSON export**: Full session log with per-rep data, trend stats, and coaching tip

---

## Evaluation Protocol (Fitness-AQA)

| Metric                              | Target            |
| ----------------------------------- | ----------------- |
| Accuracy (Knees Inward – BackSquat) | >95%              |
| F1 per error class                  | >0.80             |
| Inference latency                   | <30ms/frame (CPU) |
| Rep segmentation error              | ±1 frame          |

Baselines: OpenPose-TDM and HMR-TDM from Parmar et al. (ECCV 2022).

---

## References

1. Parmar et al. (2022). _Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment._ ECCV.
2. Yin et al. (2025). _FLEX: A Large-Scale Multimodal Dataset for Fitness Action Quality Assessment._ arXiv:2506.03198.
3. Lugaresi et al. (2019). _MediaPipe: A Framework for Building Perception Pipelines._ arXiv:1906.08172.
4. Cao et al. (2019). _OpenPose: Realtime Multi-Person 2D Pose Estimation._ IEEE TPAMI.

---

_Student: Clarence Obini Dinkaa Ebebe · 222086329 · University of Johannesburg · IT28X07_
