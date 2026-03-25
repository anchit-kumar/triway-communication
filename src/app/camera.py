import json
import time
import threading
from pathlib import Path

_DEFAULT_LABELS = str(Path(__file__).parent.parent / "models" / "labels.json")

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def normalize(hand_landmarks, handedness_name: str):
    is_left = (handedness_name or "").lower().startswith("left")

    w0 = hand_landmarks[0]
    wrist_x = float(1.0 - w0.x) if is_left else float(w0.x)
    wrist_y = float(w0.y)
    wrist_z = float(w0.z)

    pts = []
    for lm in hand_landmarks:
        x = float(1.0 - lm.x) if is_left else float(lm.x)
        y = float(lm.y)
        z = float(lm.z)
        pts.append((x - wrist_x, y - wrist_y, z - wrist_z))

    max_dist = 0.0
    for (dx, dy, dz) in pts:
        d = (dx*dx + dy*dy + dz*dz) ** 0.5
        if d > max_dist:
            max_dist = d
    scale = max_dist if max_dist > 1e-6 else 1.0

    out = []
    for (dx, dy, dz) in pts:
        out.extend([dx/scale, dy/scale, dz/scale])
    return np.array(out, dtype=np.float32)


class ASLCamera:
    def __init__(
        self,
        keypoints_model_path: str,
        asl_model_path: str,
        labels_path: str = _DEFAULT_LABELS,
        device: str = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        print_every: int = 0,   # 0 = no terminal prints
    ):
        
        #Constructor vars init
        self.keypoints_model_path = keypoints_model_path
        self.asl_model_path = asl_model_path
        self.labels_path = labels_path

        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.print_every = print_every
        

        #ASL Model init
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        with open(self.labels_path, "r") as f:
            self.classes = json.load(f)["classes"]

        self.model = tf.keras.models.load_model(self.asl_model_path)

        base_options = python.BaseOptions(model_asset_path=self.keypoints_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.keypoints_model = vision.HandLandmarker.create_from_options(options)
        
        #Setup camera
        self.cam = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open camera: {self.device}")
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)

        #Threading setup
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        
        #Class vars init
        self.latest = None      # (label, prob) or None
        self.latest_top5 = None # list[(label, prob)] or None
        self.latest_ts_ms = 0
        self.latest_err = None
        self.latest_frame = None


        #Start thread
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        frame_i = 0
        last_ts = -1

        while not self.stop_evt.is_set():
            ok, bgr = self.cam.read()
            if not ok or bgr is None:
                continue

            with self.lock:
                self.latest_frame = bgr


            ts_ms = int(time.monotonic() * 1000)
            if ts_ms <= last_ts:
                ts_ms = last_ts + 1
            last_ts = ts_ms

            try:
                #Keypoint detection
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                result = self.keypoints_model.detect_for_video(mp_image, ts_ms)

                pred = None
                top5_list = None

                #From keypoints to ASL class prediction
                if result.hand_landmarks and len(result.hand_landmarks) == 1:
                    handed_name = None
                    if result.handedness and len(result.handedness[0]) > 0:
                        handed_name = result.handedness[0][0].category_name

                    x63 = normalize(result.hand_landmarks[0], handed_name)
                    probs = self.model.predict(x63.reshape(1, 63), verbose=0)[0]
                    top5 = np.argsort(probs)[::-1][:5]

                    top1_id = int(top5[0])
                    pred = (self.classes[top1_id], float(probs[top1_id]))

                    top5_list = [(self.classes[int(i)], float(probs[int(i)])) for i in top5]

                with self.lock:
                    self.latest = pred
                    self.latest_top5 = top5_list
                    self.latest_ts_ms = ts_ms
                    self.latest_err = None

                frame_i += 1
                """
                if self.print_every and (frame_i % self.print_every == 0):
                    if pred is None:
                        print(f"\nFrame {frame_i}: nothing")
                    else:
                        print(f"\nFrame {frame_i}: TOP1 {pred[0]} {pred[1]:.4f}")
                        if top5_list:
                            print("TOP5:", "  ".join([f"{c}:{p:.2f}" for c, p in top5_list]))
                """

            except Exception as e:
                with self.lock:
                    self.latest_err = repr(e)
                time.sleep(0.01)

    def predict_cur_letter(self):
        with self.lock:
            return self.latest

    def get_top5(self):
        with self.lock:
            return self.latest_top5

    def get_status(self):
        with self.lock:
            return {
                "ts_ms": self.latest_ts_ms,
                "err": self.latest_err,
                "has_pred": self.latest is not None,
            }

    def close(self):
        self.stop_evt.set()
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)
        try:
            self.cam.release()
        except Exception:
            pass
        try:
            self.keypoints_model.close()
        except Exception:
            pass

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

