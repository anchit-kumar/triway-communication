# ASL Communication System

> An inclusive real-time communication device that bridges deaf, blind, and non-impaired users — built on an NVIDIA Jetson Orin Nano.

---

## Overview

Three users with different accessibility needs share a single conversation:

| User | Input | Output |
|------|-------|--------|
| **Deaf** | ASL hand signs via webcam | Text committed to shared chat log |
| **Blind** | USB keyboard with TTS feedback | Hears new messages via espeak-ng |
| **Non-impaired** | USB keyboard | Text displayed on 20×4 LCD |

All messages are written to a shared `src/app/chat.txt` log. LCD displays refresh automatically when new messages arrive.

### ASL Recognition Pipeline

```
Camera (1280×720 @ 30fps)
  → MediaPipe HandLandmarker   (21 keypoints)
  → Normalize landmarks        (63-dim feature vector)
  → Keras MLP (256→256→128→28) (letter classification)
  → Majority-vote window       (12 frames, need 10 to agree)
  → Committed letter
```

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Platform | NVIDIA Jetson Orin Nano (aarch64) |
| GPU | Jetson integrated GPU, CUDA 12.6 |
| Camera | USB camera at `/dev/video0` (1280×720, 30 fps) |
| Keyboards | 2× USB keyboards (evdev) |
| I2C expanders | 2× MCP23017 on I2C bus 7 (addresses `0x23`, `0x21`) |
| LCD displays | 2× 20×4 character displays (4-bit parallel via MCP23017) |
| Audio | Speaker via PulseAudio (`espeak-ng` + `paplay`) |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/anchit-kumar/triway-communication.git
cd SciencePrj25
git lfs pull   # downloads the ML model files
```

### 2. Install dependencies

```bash
bash src/scripts/setup.sh
```

This creates `venv/tf_train/` and installs all packages from `requirements.txt`.

> **Jetson note:** TensorFlow must be the NVIDIA-patched build (`tensorflow==2.16.1+nv24.8`).
> Standard `pip install tensorflow` will **not** work on Jetson.
> Install it from [NVIDIA's Jetson Python packages](https://developer.nvidia.com/embedded/downloads).

### 3. Activate the environment

```bash
source src/scripts/env.sh
```

### 4. Run the application

```bash
cd src/app
python main.py
```

---

## Web Demo (no hardware required)

Test the camera + ASL model without any I2C hardware:

```bash
source src/scripts/env.sh
cd src/demo
python test_aslLive.py --host 0.0.0.0 --port 8080
```

Open `http://<jetson-ip>:8080` in a browser to see a live MJPEG stream with ASL predictions overlaid.

---

## Training the Model

The classifier is a 4-layer MLP trained on hand landmark keypoints extracted from the ASL Alphabet Dataset.

```bash
# 1. Download the dataset (requires Kaggle API credentials)
python src/training/downloadataset.py

# 2. Convert images to landmark CSV
python src/training/convert_data.py \
  --dataset data/ASL_Alphabet_Dataset \
  --out_csv src/training/outputs/asl_landmarks.csv

# 3. Open and run the training notebook
jupyter notebook src/training/train_scratch.ipynb
```

> **Note:** A CNN-based approach was tried first and did not generalize well.
> The keypoint-based MLP is the working approach.

---

## Project Structure

```
SciencePrj25/
├── src/
│   ├── app/                    # Main hardware application
│   │   ├── main.py             # Entry point + configuration
│   │   ├── camera.py           # ASL recognition (MediaPipe + TF)
│   │   ├── keyboard.py         # USB keyboard input (evdev)
│   │   ├── lcd.py              # 20×4 I2C LCD control
│   │   ├── speaker.py          # TTS (espeak-ng)
│   │   ├── gpioExpander.py     # MCP23017 I2C driver
│   │   └── chat.py             # Shared chat log helpers
│   ├── training/               # ML training & data pipeline
│   │   ├── train_scratch.ipynb # Main training notebook
│   │   ├── convert_data.py     # Images → landmark CSV
│   │   ├── downloadataset.py   # Kaggle dataset downloader
│   │   └── outputs/            # Generated CSVs / plots
│   ├── models/                 # ML model files (Git LFS)
│   │   ├── hand_landmarker.task
│   │   ├── mlp_best.keras
│   │   └── labels.json
│   ├── demo/                   # Web demo (no hardware needed)
│   │   └── test_aslLive.py     # Flask MJPEG server
│   ├── assets/                 # Reference images
│   └── scripts/
│       ├── setup.sh            # One-time environment setup
│       └── env.sh              # Activate venv + set env vars
├── data/                       # Dataset (gitignored — see data/README.md)
├── requirements.txt
├── LICENSE
└── CLAUDE.MD                   # Developer reference
```

---

## Configuration

All tunable constants live at the top of `src/app/main.py`:

| Constant | Default | Purpose |
|---|---|---|
| `ASL_CONF_MIN` | `0.80` | Min ML confidence to accept a prediction |
| `ASL_WINDOW` | `12` | Sliding window size (frames) |
| `ASL_NEED` | `10` | Frames in window that must agree |
| `ASL_NOHAND_SECONDS` | `0.7` | Idle time before ASL state reset |
| `ASL_SEND_SECONDS` | `5` | Auto-send after 5 s of no hand |
| `CHAT_POLL_SECONDS` | `0.25` | Chat file poll interval |

---

## License

MIT — see [LICENSE](LICENSE).
