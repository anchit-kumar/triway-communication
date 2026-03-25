#!/usr/bin/env python3
import argparse
import os
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


# 21 landmark names in MediaPipe Hands order
LM_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def draw_landmarks_bgr(bgr, hand_landmarks_list):
    """Draw landmarks & connections on a BGR image."""
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    for hand_landmarks in hand_landmarks_list:
        # Convert to the proto that drawing_utils expects
        landmark_list_proto = landmark_pb2.NormalizedLandmarkList(
            landmark=[
                landmark_pb2.NormalizedLandmark(
                    x=lm.x, y=lm.y, z=lm.z
                )
                for lm in hand_landmarks
            ]
        )

        mp_drawing.draw_landmarks(
            image=bgr,
            landmark_list=landmark_list_proto,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
        )
    return bgr


def print_keypoints(result):
    if not result.hand_landmarks:
        print("No hands detected.")
        return

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        # handedness: list per hand -> list of categories (usually best is [0])
        handed = None
        hand_score = None
        if result.handedness and len(result.handedness) > i and len(result.handedness[i]) > 0:
            handed = result.handedness[i][0].category_name
            hand_score = result.handedness[i][0].score

        print(f"\nHand {i}:")
        print(f"  handedness={handed}  score={hand_score}")
        print(f"  num_keypoints={len(hand_landmarks)}")

        # Print all 21 in order (index is the order)
        for idx, lm in enumerate(hand_landmarks):
            name = LM_NAMES[idx] if idx < len(LM_NAMES) else f"LM_{idx}"
            print(f"  {idx:02d} {name:>17s}: x={lm.x:.6f} y={lm.y:.6f} z={lm.z:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to hand_landmarker.task")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--out", required=True, help="Output annotated image path")
    ap.add_argument("--max_hands", type=int, default=1)
    ap.add_argument("--min_hand_det_conf", type=float, default=0.5)
    ap.add_argument("--min_hand_pres_conf", type=float, default=0.5)
    ap.add_argument("--min_tracking_conf", type=float, default=0.5)
    args = ap.parse_args()

    # Read image (BGR)
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_hand_det_conf,
        min_hand_presence_confidence=args.min_hand_pres_conf,
        min_tracking_confidence=args.min_tracking_conf,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    # Print keypoints + counts
    print_keypoints(result)

    # Draw (if any)
    if result.hand_landmarks:
        bgr = draw_landmarks_bgr(bgr, result.hand_landmarks)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, bgr)
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
