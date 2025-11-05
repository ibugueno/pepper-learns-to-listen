# Pepper Learns to Listen
Multimodal perception pipeline for **Student-Directed Speech Recognition** in **Educational Human–Robot Interaction (HRI)**.  
This repository provides three core perception modules that enable Pepper to:
1. **Detect who is speaking**
2. **Understand whether speech is addressed to the robot**
3. **Estimate a student’s engagement level**
4. **Adjust interaction behavior accordingly**

The system is designed for **real-time deployment** on Pepper with the **Jetson Orin** add-on (PEPPER-XP), but can also run on desktop systems.

---

## Repository Structure

```
pepper-learns-to-listen/
│
├── head_pose/                  # Head Pose Estimation (Biwi Kinect Head Pose)
│   ├── train_biwi_vit.py       # ViT-based regression model
│   ├── datasets/               # (user-provided) Biwi RGB-D frames
│   └── manifests/              # CSV listing images and labels
│
├── lip_activity/               # Active Speaker / Lip Activity Detection (AVA)
│   ├── train_ava_activespeaker.py
│   ├── audio_video_utils.py
│   ├── manifests/
│   └── preprocessing/          # Faces crops + aligned audio extraction pipeline
│
└── engagement_detection/       # Student Engagement Recognition (DIPSER)
    ├── train_engagement_dipser_noaudio.py
    ├── manifests/
    └── datasets/               # (user-provided) frame sequences + frame-level labels
```

---

## Module Overview

| Module | Input | Output | Dataset | Purpose |
|-------|-------|--------|---------|---------|
| **Head Pose** | RGB face crops | (yaw, pitch, roll) | Biwi Kinect Head Pose | Estimates **attention direction** and whether the student is facing the robot. |
| **Lip / Active Speaker** | Face frames + (optional) audio | speaking / not speaking | AVA-ActiveSpeaker | Determines **who is speaking** at each moment. |
| **Engagement Detection** | Face sequences | low / medium / high | DIPSER | Measures **interaction readiness, attention and affective involvement**. |

These three signals jointly support **student-directed interaction**:
- Pepper listens **only when** the student is facing it *and* speaking.
- Pepper adapts **interaction strategy** depending on engagement state.

---

## Installation

```
git clone https://github.com/<your-username>/pepper-learns-to-listen.git
cd pepper-learns-to-listen

conda create -n pepper-listen python=3.9 -y
conda activate pepper-listen

pip install torch torchvision torchaudio timm
pip install numpy opencv-python tqdm
```

If using Jetson Orin:
```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51     torch torchvision torchaudio
```

---

## 1) Head Pose Estimation

```
python head_pose/train_biwi_vit.py   --train-manifest head_pose/manifests/train.csv   --val-manifest   head_pose/manifests/val.csv   --img-size 224   --pretrained   --epochs 40   --out-dir head_pose/runs/vit
```

Output: `yaw`, `pitch`, `roll` in degrees per frame.

---

## 2) Active Speaker / Lip Activity Detection

```
python lip_activity/train_ava_activespeaker.py   --train-manifest lip_activity/manifests/train.csv   --val-manifest   lip_activity/manifests/val.csv   --num-frames 32   --temporal gru   --epochs 30   --out-dir lip_activity/runs/gru
```

If no audio is provided, the model automatically falls back to **vision-only temporal cues**.

---

## 3) Engagement Detection (DIPSER)

```
python engagement_detection/train_engagement_dipser.py   --train-manifest engagement_detection/manifests/train.csv   --val-manifest   engagement_detection/manifests/val.csv   --num-frames 32   --temporal gru   --epochs 30   --out-dir engagement_detection/runs/gru
```

Output: `{low, medium, high}` per frame.

---

## Real-Time Fusion (Runtime)

Each module outputs:
- `/head_pose`: `{yaw, pitch, roll, facing_robot_prob}`
- `/active_speaker`: `{person_id, speaking_prob}`
- `/engagement`: `{engagement, trend}`

The **fusion policy** decides when Pepper:
- listens
- asks for clarification
- adapts tone and gestures

---

## License
MIT License (code). Dataset licenses follow their original terms.
