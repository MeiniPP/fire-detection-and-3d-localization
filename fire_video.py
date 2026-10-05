#!/usr/bin/env python3
"""Detect fire in ordinary video files without estimating depth."""

import argparse
import os
import time
from pathlib import Path

import cv2

from detector import DEFAULT_MODEL, FireDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True, help='Input video file.')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--nms', type=float, default=0.4)
    parser.add_argument('--provider', choices=('cpu', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--hsv', action='store_true', help='Apply the camera pipeline\'s HSV filter.')
    parser.add_argument('--headless', action='store_true', help='Process without opening a display window.')
    parser.add_argument('--output', type=Path, help='Optional annotated MP4 output.')
    parser.add_argument('--max-frames', type=int, help='Optional frame limit for a quick smoke test.')
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or not 0 < args.nms <= 1:
        parser.error('Confidence and NMS thresholds must be in (0, 1].')
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive.')
    if args.output and os.path.normcase(os.path.abspath(args.output)) == os.path.normcase(os.path.abspath(args.video)):
        parser.error('Output must differ from the input video.')

    detector = FireDetector(args.model, args.confidence, args.nms, args.provider)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit('Cannot open video: {}'.format(args.video))
    writer = None
    frames = 0
    total_detections = 0
    started = time.perf_counter()
    try:
        while args.max_frames is None or frames < args.max_frames:
            success, frame = cap.read()
            if not success:
                break
            tick = time.perf_counter()
            detections = detector.detect(frame, hsv_filter=args.hsv)
            elapsed = time.perf_counter() - tick
            for (left, top, right, bottom), score, _ in detections:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, 'fire:{:.2f}'.format(score), (left, max(15, top - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, 'Processing FPS: {:.2f}'.format(1.0 / max(elapsed, 1e-9)),
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if args.output:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not 0 < fps < 1000:
                        fps = 30.0
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create output video: {}'.format(args.output))
                writer.write(frame)
            frames += 1
            total_detections += len(detections)
            if not args.headless:
                cv2.imshow('Fire detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
    if frames == 0:
        raise SystemExit('The video contains no readable frames.')
    print('Processed {} frames; {} detections; {:.2f} seconds; providers: {}'.format(
        frames, total_detections, time.perf_counter() - started, detector.net.get_providers()))


if __name__ == '__main__':
    main()
