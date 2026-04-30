# ASL Communication System

This is the durable project context for Codex. Read it before changing code in this
repo, and preserve the hardware and Jetson-specific constraints unless the user
explicitly asks to change them.

## Project Purpose

This project is an inclusive real-time communication device for three groups:

- Deaf users: ASL is captured by webcam, recognized in real time, and converted
  to text.
- Blind users: USB keyboard input receives character/word feedback through
  `espeak-ng` text to speech.
- Non-impaired users: USB keyboard input is displayed on a 20x4 LCD.

All users share a common chat log at `src/app/chat.txt`. The system targets an
NVIDIA Jetson Orin Nano with two USB keyboards, two 20x4 I2C LCD displays via
MCP23017 expanders, a USB camera, and a speaker.

## Codex Working Rules

- Prefer small, targeted changes that match the existing Python style.
- Activate the project environment before running Python commands:

  ```bash
  source src/scripts/env.sh
  ```

- Do not install or replace TensorFlow with stock `pip install tensorflow`.
  This project needs the NVIDIA Jetson build: `tensorflow==2.16.1+nv24.8`.
- Be careful with hardware entry points. `src/app/main.py` expects Jetson
  hardware, I2C bus 7, evdev keyboards, camera `/dev/video0`, and audio output.
  Running it on a non-Jetson or without devices may fail.
- Avoid destructive changes to model files, datasets, `src/app/chat.txt`, or
  hardware configuration unless the user specifically requests them.
- If testing hardware behavior, prefer the smallest relevant script or function
  first. Report when a test could not be run because hardware is unavailable.
- Keep generated data and large datasets out of git. Model files are tracked via
  Git LFS.

## Hardware Specifications

| Component | Details |
| --- | --- |
| Platform | NVIDIA Jetson Orin Nano (aarch64) |
| Kernel | Linux 5.15.148-tegra (JetPack R36.4.7, Sept 2025) |
| CPU | 6-core ARMv8 Processor rev 1 (v8l) |
| RAM | 7.4 GB unified CPU/GPU memory |
| Swap | 3.7 GB |
| GPU | NVIDIA Orin integrated GPU, CUDA 12.6 |
| SSD | 3.6 TB NVMe mounted at `/ssd` |
| Camera | USB camera at `/dev/video0` (1280x720, 30 fps, MJPEG) |
| Keyboards | 2 USB keyboards through evdev |
| I2C Expanders | 2 MCP23017 expanders on I2C bus 7 at `0x23` and `0x21` |
| LCD Displays | 2 20x4 character displays, 4-bit parallel via MCP23017 |
| Audio | PulseAudio speaker, `espeak-ng` TTS piped to `paplay` |

Keyboard names used for evdev auto-discovery:

- Non-impaired: `LiteOn Lenovo New Calliope USB Keyboard`
- Blind: `LITE-ON Technology USB NetVista Full Width Keyboard`

## Software Environment

| Item | Details |
| --- | --- |
| Python | 3.10.12 |
| TensorFlow | 2.16.1+nv24.8, NVIDIA-patched Jetson build |
| CUDA | 12.6 at `/usr/local/cuda-12.6` |
| OpenCV | 4.11.0.86 |
| NumPy | 1.26.4 |
| Virtual env | `/ssd/SciencePrj25/venv/tf_train/` |
| Env setup script | `src/scripts/env.sh` |

`src/scripts/env.sh` activates the venv, sets Tegra/CUDA library paths, and
exports `MODELS_DIR` / `HANDPOSE_MODEL`.

## Project Structure

```text
SciencePrj25/
+-- src/
|   +-- app/                        # Main hardware application
|   |   +-- main.py                 # Entry point + top-level configuration
|   |   +-- camera.py               # ASLCamera: MediaPipe + TensorFlow inference
|   |   +-- keyboard.py             # KeyboardReader: evdev USB keyboard input
|   |   +-- lcd.py                  # LCD: I2C 20x4 display control
|   |   +-- speaker.py              # Speaker: espeak-ng TTS + audio playback
|   |   +-- gpioExpander.py         # MCP23017 I2C GPIO expander driver
|   |   +-- chat.py                 # Shared chat.txt read/write helpers
|   |   +-- test.py                 # Hardware test utilities
|   +-- training/                   # ML training and data pipeline
|   +-- models/                     # ML model files tracked with Git LFS
|   +-- demo/                       # Web demo, camera only
|   +-- assets/
|   +-- scripts/
+-- Data/                           # Local dataset directory in this checkout
+-- requirements.txt
+-- README.md
+-- AGENTS.md                       # Codex project memory
+-- CLAUDE.MD                       # Legacy Claude project memory
```

## Common Commands

Run the main hardware application:

```bash
source src/scripts/env.sh
cd src/app
python main.py
```

Run the web demo without LCD/I2C hardware:

```bash
source src/scripts/env.sh
cd src/demo
python test_aslLive.py --host 0.0.0.0 --port 8080
```

Train or refresh the ASL dataset landmarks:

```bash
source src/scripts/env.sh
python src/training/convert_data.py \
  --dataset Data/ASL_Alphabet_Dataset/asl_alphabet_train \
  --out_csv src/training/outputs/asl_landmarks.csv
```

Download the dataset:

```bash
python src/training/downloadataset.py
```

This requires `kagglehub` and valid Kaggle credentials.

## ML Models

MediaPipe HandLandmarker:

- Path: `src/models/hand_landmarker.task`
- Purpose: detect 21 hand keypoints from each frame.
- Config: video mode, one hand, 0.5 detection/tracking confidence.
- Output: 63-dimensional wrist-centered, scale-normalized feature vector.

Keras MLP classifier:

- Path: `src/models/mlp_best.keras`
- Architecture: `Input(63) -> Dense(256) -> Dense(256) -> Dense(128) -> Dense(28)`
- Classes: 28, letters A-Z plus `del` and `space`.
- Framework: TensorFlow 2.16.1+nv24.8 on CUDA 12.6.
- Training note: CNN-based approaches did not work well; keypoint-based MLP is
  the working approach. Mixed precision was used for GPU efficiency.

Labels:

- Path: `src/models/labels.json`
- Maps integer class indices to letter strings.

## Main Configuration

Most tunable constants live at the top of `src/app/main.py`.

| Constant | Default | Purpose |
| --- | --- | --- |
| `ASL_CONF_MIN` | `0.80` | Minimum confidence to accept a prediction |
| `ASL_WINDOW` | `12` | Sliding window size in frames |
| `ASL_NEED` | `10` | Frames in window that must agree before commit |
| `ASL_NOHAND_SECONDS` | `0.7` | Idle time before resetting ASL state |
| `ASL_SEND_SECONDS` | `5` | Auto-send word after no hand is detected |
| `CHAT_POLL_SECONDS` | `0.25` | Interval for checking chat file updates |
| `LCD_W` / `LCD_H` | `20` / `4` | LCD character dimensions |
| `CAMERA_CFG.device` | `/dev/video0` | Camera device path |
| `CAMERA_CFG.width/height` | `1280 / 720` | Camera resolution |
| `CAMERA_CFG.fps` | `30` | Camera framerate |

I2C wiring in `src/app/main.py`:

```python
bus = smbus2.SMBus(7)
lcdNonI = LCD(bus, 0x23, 'B')
lcdDeaf = LCD(bus, 0x21, 'A')
```

## Code Conventions

- `snake_case` for functions and variables.
- `UPPER_CASE` for module-level constants.
- `PascalCase` for classes such as `ASLCamera`, `KeyboardReader`, and `LCD`.
- Common abbreviations: `NonI` means non-impaired; `ASL`, `LCD`, and `TTS` are
  standard project abbreviations.
- Hardware components are separated into modules under `src/app/`.
- `main.py` orchestrates components and owns shared configuration.

## Runtime Architecture

- Camera, keyboards, and speaker run in daemon threads.
- The main loop polls component state and chat-file changes.
- Shared mutable state should be protected with `threading.Lock()`.
- Do not block the main thread with long-running I/O.

ASL recognition uses a sliding window. A letter is committed only when
`ASL_NEED` or more frames in the `ASL_WINDOW` agree on the same prediction above
`ASL_CONF_MIN`.

`src/app/chat.txt` is the shared message log. `src/app/chat.py` provides:

- `readMsg()`
- `readMsgLCD(width, height)`
- `writeMsg(msg, person_id)`

The main loop watches the chat log with `os.stat` polling at
`CHAT_POLL_SECONDS`.

## Dependencies

Python packages include:

```text
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

TensorFlow is intentionally installed separately on Jetson as
`tensorflow==2.16.1+nv24.8`; it is documented in `requirements.txt` but not
installed from PyPI by that file.

System software:

- `espeak-ng`
- `paplay`
- `i2c-tools`
- Linux kernel with active I2C bus 7 support
