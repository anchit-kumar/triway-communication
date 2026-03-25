#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_header():
    header = [
        "rel_path",        # path relative to dataset root
        "class_name",
        "filename",
        "img_w",
        "img_h",
        "handedness",      # "Left"/"Right" (best guess)
        "handedness_score" # confidence for that label
    ]
    # 21 landmarks * (x,y,z) = 63 columns
    for i in range(21):
        header += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
    return header


def iter_images(dataset_root: Path):
    # Each subfolder is a class
    for class_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                yield class_dir.name, img_path


def detect_first_hand(landmarker, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None  # no hands

    # first hand only
    hand_landmarks = result.hand_landmarks[0]

    handedness = ""
    handedness_score = ""
    if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
        handedness = result.handedness[0][0].category_name
        handedness_score = result.handedness[0][0].score

    return hand_landmarks, handedness, handedness_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/ssd/SciencePrj25/Data/asl_alphatbet_train", help="Dataset root")
    ap.add_argument("--model", default="src/models/hand_landmarker.task", help="Path to .task model")
    ap.add_argument("--out_csv", default="src/training/outputs/asl_landmarks_norm.csv", help="Output CSV path")
    ap.add_argument("--num_hands", type=int, default=1, help="We only keep first hand anyway")
    ap.add_argument("--min_hand_det_conf", type=float, default=0.5)
    ap.add_argument("--min_hand_pres_conf", type=float, default=0.5)
    ap.add_argument("--min_tracking_conf", type=float, default=0.5)
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_det_conf,
        min_hand_presence_confidence=args.min_hand_pres_conf,
        min_tracking_confidence=args.min_tracking_conf,
    )

    header = build_header()
    total = 0
    kept = 0
    skipped_no_hand = 0
    skipped_read_fail = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for class_name, img_path in iter_images(dataset_root):
                total += 1

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    skipped_read_fail += 1
                    continue

                h, w = bgr.shape[:2]
                det = detect_first_hand(landmarker, bgr)
                if det is None:
                    skipped_no_hand += 1
                    continue

                hand_landmarks, handedness, handedness_score = det

                rel_path = str(img_path.relative_to(dataset_root))
                row = [
                    rel_path,
                    class_name,
                    img_path.name,
                    w,
                    h,
                    handedness,
                    handedness_score,
                ]

                # exactly 21 landmarks expected; if not, pad/truncate safely
                lms = list(hand_landmarks)
                if len(lms) < 21:
                    lms += [type(lms[0])(x=0.0, y=0.0, z=0.0)] * (21 - len(lms))
                lms = lms[:21]

                for lm in lms:
                    row += [lm.x, lm.y, lm.z]

                writer.writerow(row)
                kept += 1

                if kept % 500 == 0:
                    print(f"[progress] kept={kept} / total_seen={total}")

    print("\nDone.")
    print(f"  total_seen:      {total}")
    print(f"  kept:            {kept}")
    print(f"  skipped_no_hand: {skipped_no_hand}")
    print(f"  skipped_read:    {skipped_read_fail}")
    print(f"  wrote:           {out_csv}")


if __name__ == "__main__":
    main()
