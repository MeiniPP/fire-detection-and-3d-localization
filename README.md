# Fire Detection and 3D Localization

A graduation project by Haonan Yin for detecting fire with a trained YOLOv5 model and estimating camera-relative coordinates from RGB-D images. The project provides ordinary video detection and a ROS depth-camera application.

## Overview

The original project combined YOLOv5 training, manual ONNX export, HSV color filtering, synchronized RGB/depth acquisition, and pinhole-camera projection. This repository contains a runnable inference application and English documentation of the original work.

- **Video mode:** detects fire in an ordinary video file. It does not estimate distance or split frames into stereo images.
- **Camera mode:** subscribes to ROS color and registered depth images, applies detection and HSV filtering, and displays the fire source's X, Y, and Z coordinates.
- **Deployment:** includes the author's trained ONNX model, CPU inference by default, and optional CUDA execution.

## Approach

Frames are resized to the bundled model's 320 x 320 input, converted from BGR to RGB, normalized, and passed to ONNX Runtime. The original three-scale YOLOv5 anchor decoding and objectness-based confidence threshold are retained. Non-maximum suppression removes overlapping boxes. Camera mode checks the color content inside each candidate box using the original HSV thresholds.

For a detected box center `(u, v)` with aligned depth `Z`, the camera application computes:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

Coordinates are in meters in the color camera frame: X points right, Y points down, and Z points forward. The depth image must be registered to the color image, and the intrinsics must match its resolution.

## Project structure

```text
.
├── fire_video.py          # Video detection entry point
├── fire_cam.py            # ROS RGB-D detection and localization
├── detector.py            # Shared ONNX decoding, NMS, and HSV filtering
├── models/best.onnx       # Author-trained, manually exported model
├── config/camera.json     # ROS topics and original camera intrinsics
├── requirements.txt      # Inference dependencies
├── requirements-tested.txt
├── tests/                # NMS, depth, projection, and model smoke checks
├── docs/ros-setup.md      # Hardware setup and calibration instructions
├── docs/results.md       # Thesis results and verification boundaries
├── docs/provenance.md    # Model origin, third-party credit, and cleanup changes
└── LICENSE
```

## Installation

For video inference, use Python 3.9 or later. The publication checks used Python 3.9.13, OpenCV 4.11.0, NumPy 2.0.2, and ONNX Runtime 1.19.2 on Windows.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For the exact inference dependency versions checked during publication, install `requirements-tested.txt` instead. These desktop versions are separate from the original ROS Noetic environment.

## Usage

Run from the repository root with a local video file:

```bash
python fire_video.py --video path/to/fire-video.mp4
```

Press `q` to close the display. For a server or a quick smoke check:

```bash
python fire_video.py --video path/to/fire-video.mp4 --headless --max-frames 30
```

Save an annotated video:

```bash
python fire_video.py --video path/to/fire-video.mp4 --headless --output outputs/detection.mp4
```

Optional flags include `--confidence 0.5`, `--nms 0.4`, `--hsv`, and `--model models/best.onnx`. Video mode preserves the original detector's behavior without HSV filtering unless `--hsv` is supplied. Camera mode always applies the original HSV check.

Videos and raw datasets are not redistributed because their original redistribution terms were not recorded. Provide a local video you are permitted to use. The trained model is included, so retraining is unnecessary for inference.

### Depth camera

The original hardware configuration was Ubuntu 20.04, ROS Noetic, and a LeTMC-520 identified by the driver as an Orbbec Astra Pro. See [ROS setup](docs/ros-setup.md) for dependencies, camera launch, depth registration, and intrinsics.

After starting the camera driver and sourcing ROS:

```bash
python3 fire_cam.py --config config/camera.json
```

Use `--headless` to process and log coordinates without display windows. This application displays and logs coordinates; it does not publish a separate ROS coordinate topic.

### Optional GPU inference

Install a CUDA/cuDNN-compatible `onnxruntime-gpu` package in place of `onnxruntime`, following the [official CUDA provider instructions](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), then use:

```bash
python fire_video.py --video path/to/fire-video.mp4 --provider cuda
```

`--provider cuda` fails if CUDA cannot initialize; it does not silently claim GPU execution. `--provider auto` selects CUDA when available and otherwise uses CPU.

## Results

The graduation thesis reports **76.9 FPS**, approximately **13 ms per frame**, on an NVIDIA GeForce RTX 2060. It describes the camera's depth measurement range as **0.6-8 m**. These are historical thesis results and hardware specifications, rather than new benchmarks of this cleaned repository or a claim that flames were tested throughout that range.

The thesis includes near-range candle/lighter tests and screen-based flame-video experiments at longer distances. For screen-based tests, the measured depth is the screen's position. See [results and limitations](docs/results.md) for the experimental conditions and the archived training curve.

The publication checks load the included model and exercise CPU video inference. Real camera operation and the historical GPU speed require the original hardware/software environment.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests cover the actual ONNX input/output contract, a blank-frame inference smoke check, NMS rectangle conversion, no-detection handling, invalid depth values, depth units, and pinhole projection. They do not measure fire-detection accuracy or replace a camera calibration experiment.

During publication, the offline application processed all 2,975 frames of a local video and produced an annotated output that reopened successfully. See the [verification record](docs/verification.md).

## Course context and contribution

This work was completed as an undergraduate graduation project in automation at Yangzhou University in 2025. Haonan Yin built the dataset/annotation workflow, trained the YOLOv5 detector, exported the model to ONNX, integrated the ROS camera pipeline, and conducted the experiments described in the thesis. YOLOv5 and the camera driver are credited as third-party foundations; the project does not claim authorship of those frameworks.

## License and attribution

The project code is distributed under GPL-3.0; see [LICENSE](LICENSE). The original YOLOv5 v7.0 source archive used GPL-3.0. Model provenance, dataset boundaries, third-party references, and the meaningful publication fixes are documented in [provenance](docs/provenance.md).
