#!/usr/bin/env python3
"""Detect fire and estimate camera-relative coordinates from ROS RGB-D images."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from detector import DEFAULT_MODEL, FireDetector

DEFAULT_CONFIG = Path(__file__).absolute().parent / 'config' / 'camera.json'


def compute_3d_point(u, v, depth, camera_matrix):
    """Project an aligned color-image pixel and depth in meters into X, Y, Z."""
    if not np.isfinite(depth) or depth <= 0:
        return None
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    return np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth])


def depth_in_meters(value, encoding):
    """Interpret common ROS depth-image encodings explicitly."""
    if encoding == '16UC1':
        return float(value) / 1000.0
    if encoding == '32FC1':
        return float(value)
    raise ValueError('Expected depth encoding 16UC1 (millimeters) or 32FC1 (meters).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--nms', type=float, default=0.4)
    parser.add_argument('--provider', choices=('cpu', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or not 0 < args.nms <= 1:
        parser.error('Confidence and NMS thresholds must be in (0, 1].')

    try:
        import rospy
        import message_filters
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge, CvBridgeError
    except ImportError as exc:
        raise SystemExit('ROS Python dependencies are missing. Source ROS Noetic and install cv_bridge, '
                         'sensor_msgs, and message_filters. See docs/ros-setup.md.') from exc

    with args.config.open(encoding='utf-8') as handle:
        config = json.load(handle)
    camera_matrix = np.asarray(config['camera_matrix'], dtype=np.float32)
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all() or min(camera_matrix[0, 0], camera_matrix[1, 1]) <= 0:
        raise ValueError('camera_matrix must be a finite 3x3 matrix with positive focal lengths.')
    detector = FireDetector(args.model, args.confidence, args.nms, args.provider)
    bridge = CvBridge()
    rospy.init_node('fire_detection_node', anonymous=True)

    def callback(rgb_msg, depth_msg):
        try:
            rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            rospy.logerr('Image conversion failed: %s', exc)
            return
        if depth.shape != rgb.shape[:2]:
            rospy.logwarn_throttle(5, 'RGB and depth dimensions differ. Use depth registered to the color image.')
            return
        if depth_msg.encoding not in ('16UC1', '32FC1'):
            rospy.logwarn_throttle(5, 'Unsupported depth encoding: ' + depth_msg.encoding)
            return
        tick = time.perf_counter()
        detections = detector.detect(rgb, hsv_filter=True)
        for (left, top, right, bottom), score, _ in detections:
            u, v = (left + right) // 2, (top + bottom) // 2
            distance = depth_in_meters(depth[v, u], depth_msg.encoding)
            point = compute_3d_point(u, v, distance, camera_matrix)
            if point is None:
                label, position = 'Too close / No depth', '3D: Invalid'
            else:
                label = 'fire:{:.2f} Z={:.2f}m'.format(score, distance)
                position = 'X:{:.2f} Y:{:.2f} Z:{:.2f}'.format(*point)
            cv2.rectangle(rgb, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(rgb, label, (left, max(15, top - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            cv2.putText(rgb, position, (left, min(rgb.shape[0] - 5, top + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            rospy.loginfo_throttle(1, 'Fire source: ' + position)
        if not args.headless:
            elapsed = time.perf_counter() - tick
            cv2.putText(rgb, 'Processing FPS: {:.2f}'.format(1.0 / max(elapsed, 1e-9)),
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Detection', rgb)
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow('Depth', depth_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown('Detection window closed by user.')

    rgb_sub = message_filters.Subscriber(config['rgb_topic'], Image)
    depth_sub = message_filters.Subscriber(config['depth_topic'], Image)
    synchronizer = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub], queue_size=config.get('sync_queue_size', 10), slop=config.get('sync_slop', 0.1))
    synchronizer.registerCallback(callback)
    rospy.loginfo('Fire detection started; providers: %s', detector.net.get_providers())
    try:
        rospy.spin()
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
