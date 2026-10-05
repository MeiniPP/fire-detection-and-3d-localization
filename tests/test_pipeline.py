"""Regression checks for geometry/NMS and a real bundled-model smoke check."""

import unittest

import numpy as np

from detector import FireDetector, post_process_opencv
from fire_cam import compute_3d_point, depth_in_meters


class PostProcessingTests(unittest.TestCase):
    def test_disjoint_boxes_at_nonzero_origin_are_not_suppressed(self):
        predictions = np.array([[105, 105, 10, 10, 0.9, 1.0],
                                [125, 105, 10, 10, 0.8, 1.0]], dtype=np.float32)
        boxes, _, _ = post_process_opencv(predictions, 320, 320, 320, 320, 0.4, 0.5)
        self.assertEqual(len(boxes), 2)
        np.testing.assert_allclose(boxes[0], [100, 100, 110, 110])

    def test_duplicate_boxes_are_suppressed(self):
        predictions = np.array([[105, 105, 10, 10, 0.9, 1.0],
                                [105, 105, 10, 10, 0.8, 1.0]], dtype=np.float32)
        boxes, scores, _ = post_process_opencv(predictions, 320, 320, 320, 320, 0.4, 0.5)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)

    def test_predictions_below_threshold_return_no_boxes(self):
        predictions = np.array([[105, 105, 10, 10, 0.1, 1.0]], dtype=np.float32)
        boxes, scores, classes = post_process_opencv(predictions, 320, 320, 320, 320, 0.4, 0.5)
        self.assertEqual((len(boxes), len(scores), len(classes)), (0, 0, 0))


class DepthTests(unittest.TestCase):
    matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

    def test_optical_axis_projects_to_zero_xy(self):
        np.testing.assert_allclose(compute_3d_point(320, 240, 2.0, self.matrix), [0, 0, 2])

    def test_projection_uses_camera_axis_units(self):
        np.testing.assert_allclose(compute_3d_point(370, 215, 2.0, self.matrix), [0.2, -0.1, 2])

    def test_invalid_depth_does_not_produce_coordinates(self):
        for value in (0, -1, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertIsNone(compute_3d_point(320, 240, value, self.matrix))

    def test_depth_units_are_equivalent(self):
        self.assertEqual(depth_in_meters(1500, '16UC1'), depth_in_meters(1.5, '32FC1'))

    def test_unsupported_depth_encoding_is_rejected(self):
        with self.assertRaises(ValueError):
            depth_in_meters(1, '8UC1')


class ModelSmokeTests(unittest.TestCase):
    def test_bundled_model_runs_on_cpu_without_ros(self):
        detector = FireDetector(provider='cpu')
        self.assertEqual(detector.net.get_inputs()[0].shape, [1, 3, 320, 320])
        self.assertEqual(detector.net.get_outputs()[0].shape, [1, 6300, 6])
        self.assertEqual(detector.detect(np.zeros((240, 426, 3), dtype=np.uint8)), [])


if __name__ == '__main__':
    unittest.main()
