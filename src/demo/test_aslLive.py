#!/usr/bin/env python3
import argparse
import time

import cv2
from flask import Flask, Response

from camera import ASLCamera


def draw_overlay(bgr, pred):
    if pred is None:
        txt = "TOP1: nothing"
    else:
        lbl, prob = pred
        txt = f"TOP1: {lbl} ({prob:.2f})"
    cv2.putText(bgr, txt, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return bgr


def mjpeg_generator(cam: ASLCamera, jpeg_q: int):
    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        pred = cam.predict_cur_letter()
        frame = draw_overlay(frame, pred)

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--jpeg_q", type=int, default=80)

    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--hand_task", default="/ssd/SciencePrj25/src/models/hand_landmarker.task")
    ap.add_argument("--model", default="/ssd/SciencePrj25/src/models/mlp_best.keras")
    ap.add_argument("--labels", default="/ssd/SciencePrj25/src/models/labels.json")
    ap.add_argument("--print_every", type=int, default=10)
    args = ap.parse_args()

    cam = ASLCamera(
        keypoints_model_path=args.hand_task,
        asl_model_path=args.model,
        labels_path=args.labels,
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        print_every=args.print_every,
    )

    app = Flask(__name__)

    @app.route("/")
    def index():
        return (
            "<html><body style='background:#111;color:#eee;font-family:Arial;'>"
            "<h2>ASL Live Stream</h2>"
            "<img src='/stream' style='max-width:95vw;border:2px solid #444;'/>"
            "</body></html>"
        )

    @app.route("/stream")
    def stream():
        return Response(
            mjpeg_generator(cam, args.jpeg_q),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=False)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
