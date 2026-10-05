# Provenance and publication changes

## Original work

Author: Haonan Yin. The project was developed for a 2025 undergraduate graduation thesis at Yangzhou University. The author's contribution includes the fire dataset/annotation workflow, YOLOv5 training, manual ONNX export, ROS image integration, HSV filtering, camera projection, and thesis experiments.

The publication source was selected by modification date:

- The RGB-D camera application was last modified on May 6, 2025. Identical copies existed in the two named system folders and in the development file `fire2.py`.
- The video application was last modified on February 8, 2026. Identical dated versions existed in the development folder and the named system folder.

The original archive is preserved separately. Personal documents, student identifiers, transcripts, certificates, grading forms, plagiarism reports, school templates, reference-paper PDFs, virtual environments, unrelated learning exercises, and duplicate archives are excluded.

## Model

`models/best.onnx` is the author's fire detector trained using YOLOv5 and manually exported for deployment. The thesis describes exporting `best.pt` with YOLOv5's `export.py`. The included artifact has input `images` with shape `[1, 3, 320, 320]` and output `output` with shape `[1, 6300, 6]`.

The thesis describes a 640 x 640 export, whereas the preserved deployment artifact is 320 x 320. Inference follows the actual artifact. Its raw prediction coordinates require the original three-scale anchor decoding; it is not interchangeable with an arbitrary modern YOLO export. No reconstructed training checkpoint or invented export command is supplied. The exact archived training run corresponding to this export has not been established.

The four archived copies of this ONNX model were byte-identical. This repository keeps one copy, unchanged.

## Third-party foundations

- [Ultralytics YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0): training framework, pretrained initialization, and export tooling. The local v7.0 archive carries GPL-3.0; that license is retained for this project code. The upstream framework is not copied wholesale into this repository.
- [Orbbec ROS Astra camera driver](https://github.com/orbbec/ros_astra_camera): camera acquisition. It is installed separately and retains its own license.
- [OpenCV](https://opencv.org/), [ONNX Runtime](https://onnxruntime.ai/), and [NumPy](https://numpy.org/): inference and numerical dependencies, installed separately under their respective licenses.
- The thesis records a fire dataset labeled using Roboflow, but a dataset redistribution license and exact source manifest were not retained. Raw datasets are therefore not distributed. An older development note references the DeepQuestAI Fire-Smoke-Dataset; this does not establish that the final model used that dataset.

The author's exported weights are included with the author's publication authorization. Dataset images and third-party video footage are excluded. The model is not presented as trained from exclusively self-photographed images.

## Meaningful changes for publication

1. Extracted the original common inference functions into `detector.py`; retained resizing, RGB normalization, anchor decoding, objectness confidence, and HSV thresholds.
2. Corrected OpenCV NMS inputs to use `[x, y, width, height]`; retained `[left, top, right, bottom]` for drawing and ROI extraction. Flattened returned NMS indices for OpenCV compatibility.
3. Made video mode operate on an ordinary complete video frame without depth calculation, following the author's confirmed intended use. Removed the inherited stereo-calibration and SGBM branch from this entry point.
4. Added configurable input/model/output paths, CPU/CUDA selection, headless operation, frame limits, and explicit input/output errors. Default model/config paths are relative to the scripts rather than a personal working directory.
5. Moved the camera's original topics and intrinsics into `config/camera.json`. Retained approximate RGB/depth synchronization and center-pixel pinhole projection. Added clipped box bounds and explicit ROS depth units to prevent invalid indexing or silently misinterpreted depth.
6. Added English documentation, a dependency snapshot, regression/smoke tests, and publication exclusions. No weights were retrained, no historical measurements were changed, and no new accuracy benchmark is claimed.

The original display logged every detected point; the cleaned camera application logs at most once per second to keep the terminal usable. The on-screen FPS label measures detection processing, while the video CLI summary includes the overall processing loop. Neither is equated with the historical thesis benchmark.
