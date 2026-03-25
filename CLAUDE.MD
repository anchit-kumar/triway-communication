# ASL Communication System

## Purpose

This project is an inclusive real-time communication device that bridges three groups of users with different abilities:

- **Deaf users** — Sign language (ASL) captured via webcam is recognized in real-time and converted to text
- **Blind users** — USB keyboard input with character/word text-to-speech feedback via espeak-ng
- **Non-impaired users** — USB keyboard input displayed on an LCD screen

All three users share a common chat log (`src/app/chat.txt`). The system runs on an NVIDIA Orin Jetson Nano with two USB keyboards, two 20×4 I2C LCD displays (driven via MCP23017 I2C GPIO expanders), a USB camera, and a speaker.

---

## Hardware Specifications

| Component | Details |
|-----------|---------|
| **Platform** | NVIDIA Jetson Orin Nano (aarch64) |
| **Kernel** | Linux 5.15.148-tegra (JetPack R36.4.7, Sept 2025) |
| **CPU** | 6-core ARMv8 Processor rev 1 (v8l) |
| **RAM** | 7.4 GB unified CPU/GPU memory |
| **Swap** | 3.7 GB |
| **GPU** | NVIDIA Orin (nvgpu), integrated — CUDA 12.6 |
| **SSD** | 3.6 TB NVMe mounted at `/ssd` |
| **Camera** | USB camera at `/dev/video0` (1280×720, 30 fps, MJPEG) |
| **Keyboards** | 2× USB keyboards (evdev input layer) |
| **I2C Expanders** | 2× MCP23017 on I2C bus 7 — addresses `0x23` and `0x21` |
| **LCD Displays** | 2× 20×4 character displays (4-bit parallel, via MCP23017) |
| **Audio** | PulseAudio speaker (espeak-ng TTS → paplay) |

**Specific keyboard names (used for evdev auto-discovery)**:
- Non-impaired: `"LiteOn Lenovo New Calliope USB Keyboard"`
- Blind: `"LITE-ON Technology USB NetVista Full Width Keyboard"`

---

## Software Environment

| Item | Details |
|------|---------|
| **Python** | 3.10.12 |
| **TensorFlow** | 2.16.1+nv24.8 (NVIDIA-patched build for Jetson) |
| **CUDA** | 12.6 (`/usr/local/cuda-12.6`) |
| **OpenCV** | 4.11.0.86 |
| **NumPy** | 1.26.4 |
| **Virtual env** | `/ssd/SciencePrj25/venv/tf_train/` |
| **Env setup script** | `src/scripts/env.sh` |

**Always activate the environment before running anything:**
```bash
source src/scripts/env.sh
```
This activates the venv, sets `LD_LIBRARY_PATH` for Tegra/CUDA libraries, and exports `MODELS_DIR` / `HANDPOSE_MODEL`.

---

## Project Structure

```
SciencePrj25/
├── src/
│   ├── app/                        # Main hardware application
│   │   ├── main.py                 # Entry point + top-level configuration
│   │   ├── camera.py               # ASLCamera: MediaPipe + TensorFlow inference
│   │   ├── keyboard.py             # KeyboardReader: evdev USB keyboard input
│   │   ├── lcd.py                  # LCD: I2C 20×4 display control
│   │   ├── speaker.py              # Speaker: espeak-ng TTS + audio playback
│   │   ├── gpioExpander.py         # MCP23017 I2C GPIO expander driver
│   │   ├── chat.py                 # Shared chat.txt read/write helpers
│   │   └── test.py                 # Hardware test utilities
│   ├── training/                   # ML training & data pipeline
│   │   ├── train_scratch.ipynb     # Main training notebook (MLP classifier)
│   │   ├── train_scratch_old.py    # Legacy CNN attempt (did not work well)
│   │   ├── convert_data.py         # Converts image dataset → hand landmark CSV
│   │   ├── convert_to_onnx.py      # ONNX export utility
│   │   ├── downloadataset.py       # Kaggle dataset downloader
│   │   ├── test_image_ASL.py       # Single-image ASL test script
│   │   ├── test_image_keypoints.py # Single-image keypoint visualizer
│   │   └── outputs/
│   │       └── asl_landmarks_norm.csv
│   ├── models/                     # ML model files (Git LFS)
│   │   ├── hand_landmarker.task    # MediaPipe hand landmark model (7.8 MB)
│   │   ├── mlp_best.keras          # Trained ASL MLP classifier (1.5 MB)
│   │   └── labels.json             # Class label mappings (A–Z + del + space)
│   ├── demo/                       # Web demo (no hardware required)
│   │   ├── test_aslLive.py         # Flask MJPEG server with ASL overlay
│   │   └── camera.py               # Camera module for demo
│   ├── assets/
│   │   └── images/                 # Reference annotated images
│   └── scripts/
│       ├── setup.sh                # One-time: create venv + install deps
│       └── env.sh                  # Activate venv + set env vars (source this)
├── data/                           # Dataset (gitignored — see data/README.md)
├── requirements.txt
├── LICENSE
├── .gitignore
└── .gitattributes                  # Git LFS tracking for *.keras and *.task
```

---

## How to Use

### Run the main application (requires all hardware)

```bash
source src/scripts/env.sh
cd src/app
python main.py
```

### Run the web demo (no hardware needed — camera only)

```bash
source src/scripts/env.sh
cd src/demo
python test_aslLive.py --host 0.0.0.0 --port 8080
# Visit http://10.10.10.2:8080 (or whatever the Jetson IP is)
```

### Train the ASL model

```bash
source src/scripts/env.sh

# 1. Convert dataset images to landmarks CSV
python src/training/convert_data.py \
  --dataset data/ASL_Alphabet_Dataset \
  --out_csv src/training/outputs/asl_landmarks.csv

# 2. Open and run src/training/train_scratch.ipynb in Jupyter
```

### Download the dataset

```bash
python src/training/downloadataset.py
# Requires kagglehub + valid Kaggle API credentials
```

---

## ML Models

### MediaPipe HandLandmarker
- **Path**: `src/models/hand_landmarker.task` (7.8 MB, Git LFS)
- **Purpose**: Detect 21 hand keypoints (x, y, z) from each camera frame
- **Config**: Video mode, 1 hand, 0.5 detection/tracking confidence
- **Output**: 63-dim normalized feature vector (wrist-centered, scale-normalized)

### Keras MLP Classifier
- **Path**: `src/models/mlp_best.keras` (1.5 MB, Git LFS)
- **Architecture**: `Input(63) → Dense(256, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(28, Softmax)`
- **Classes**: 28 — A through Z plus `"del"` and `"space"`
- **Framework**: TensorFlow 2.16.1+nv24.8 on CUDA 12.6
- **Training notes**: CNN-based approaches failed; keypoint-based MLP works well. Mixed-precision (float16) training used for GPU efficiency.

### Label file
- **Path**: `src/models/labels.json`
- Maps integer class indices to letter strings

---

## Configuration Reference

All tunable constants are at the top of `src/app/main.py`.

| Constant | Default | Purpose |
|---|---|---|
| `ASL_CONF_MIN` | `0.80` | Minimum ML confidence to accept a prediction |
| `ASL_WINDOW` | `12` | Sliding window size (frames) for majority voting |
| `ASL_NEED` | `10` | Frames in window that must agree to commit a letter |
| `ASL_NOHAND_SECONDS` | `0.7` | Idle time (s) before resetting ASL state |
| `ASL_SEND_SECONDS` | `5` | Auto-send word after 5 s of no hand detected |
| `CHAT_POLL_SECONDS` | `0.25` | Interval for checking chat file updates |
| `LCD_W` / `LCD_H` | `20` / `4` | LCD display dimensions (characters) |
| `CAMERA_CFG.device` | `/dev/video0` | Camera device path |
| `CAMERA_CFG.width/height` | `1280 / 720` | Camera resolution |
| `CAMERA_CFG.fps` | `30` | Camera framerate |

**I2C wiring** (`src/app/main.py`):
```python
bus = smbus2.SMBus(7)           # I2C bus 7
lcdNonI = LCD(bus, 0x23, 'B')  # Non-impaired LCD: expander 0x23, bank B
lcdDeaf  = LCD(bus, 0x21, 'A') # Deaf LCD:         expander 0x21, bank A
```

---

## Code Conventions & Style

### Naming
- `snake_case` — functions and variables
- `UPPER_CASE` — module-level constants
- `PascalCase` — class names (e.g., `ASLCamera`, `KeyboardReader`, `LCD`)
- Common abbreviations: `NonI` (non-impaired), `ASL`, `LCD`, `TTS`

### Module organization
Each hardware component has its own file in `src/app/`. `main.py` orchestrates all components and holds the shared configuration block at the top.

### Threading model
All I/O (camera, keyboards, speaker) runs in **daemon threads**. The main loop polls results. Shared state is protected with `threading.Lock()`. Never block the main thread with I/O.

### ASL recognition logic
A sliding window of `ASL_WINDOW` frames is maintained. A letter is committed only when `ASL_NEED` or more frames in the window agree on the same prediction above `ASL_CONF_MIN` confidence.

### Chat system
`src/app/chat.txt` is the shared message log. `chat.py` provides `readMsg()`, `readMsgLCD(width, height)`, and `writeMsg(msg, person_id)` helpers. The main loop monitors via `os.stat` polling at `CHAT_POLL_SECONDS` intervals.

---

## Dependencies

### Python packages (in venv)

```
tensorflow==2.16.1+nv24.8   # NVIDIA Jetson build — do not replace with stock TF
mediapipe
opencv-python==4.11.0.86
numpy==1.26.4
pandas==2.2.2
scipy==1.15.3
flask
evdev
smbus2
kagglehub
```

> **Important**: Do not `pip install tensorflow` — this will break CUDA/Tegra compatibility.

### System software

- `espeak-ng` — text-to-speech engine
- `paplay` — PulseAudio player
- `i2c-tools` — I2C bus utilities (`i2cdetect` for debugging)
- Linux kernel with I2C support (bus 7 active)
