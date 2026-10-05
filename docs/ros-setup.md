# ROS RGB-D setup

## Original environment

The graduation project used Ubuntu 20.04, ROS Noetic, Python 3.8, and an Orbbec LeTMC-520 camera identified as Astra Pro. ROS dependencies are installed through the operating system, not through the desktop `requirements-tested.txt` file.

This page documents the original platform for reproducing historical work. Hardware setup has not been rerun during repository cleanup.

## Dependencies

On an existing ROS Noetic installation:

```bash
source /opt/ros/noetic/setup.bash
sudo apt install python3-opencv python3-numpy python3-pip python3-venv \
  ros-noetic-rospy ros-noetic-sensor-msgs ros-noetic-cv-bridge \
  ros-noetic-message-filters
```

Create an environment that can access the ROS and OpenCV system packages:

```bash
python3 -m venv --system-site-packages .venv-ros
source .venv-ros/bin/activate
python -m pip install onnxruntime==1.16.3
```

ONNX Runtime 1.16.3 provides [Python 3.8 wheels](https://pypi.org/project/onnxruntime/1.16.3/). This is a compatibility choice for the documented ROS platform, not the author's recorded original package version or a newly tested hardware environment. Keep the system NumPy/OpenCV packages in this environment; installing the desktop NumPy 2 dependency snapshot can break the older `cv_bridge` ABI.

## Camera driver

Use the [Orbbec ROS Astra driver instructions](https://github.com/orbbec/ros_astra_camera) to install the required OpenNI/libuvc dependencies, build its catkin workspace, and install its USB device rules. The original project used a workspace containing the `astra_camera` package.

After building the driver workspace:

```bash
source /opt/ros/noetic/setup.bash
source ~/ros_ws/devel/setup.bash
roslaunch astra_camera astra_pro.launch depth_align:=true
```

The driver's current [`astra_pro.launch`](https://github.com/orbbec/ros_astra_camera/blob/main/launch/astra_pro.launch) exposes `depth_align`. Driver versions and hardware firmware may differ; check the launch file installed on your machine and confirm registration rather than assuming equal image sizes prove alignment.

In another terminal, inspect the streams:

```bash
source /opt/ros/noetic/setup.bash
source ~/ros_ws/devel/setup.bash
rostopic list
rostopic hz /camera/color/image_raw
rostopic hz /camera/depth/image_raw
rostopic echo -n 1 /camera/color/camera_info
```

Default topics are `/camera/color/image_raw` and `/camera/depth/image_raw`, as in the original program. Update `config/camera.json` if the driver exposes registered depth under another topic. Color and depth must share the color-camera pixel coordinates and image dimensions. Configure the camera for 640 x 480 color and aligned depth when using the bundled historical intrinsics.

## Intrinsics and units

`config/camera.json` retains the author's color camera matrix:

```text
fx = 577.54679    fy = 578.63325
cx = 310.24326    cy = 253.65539
```

These numbers are device/resolution-specific. Replace them with the calibrated color-camera intrinsics (the `K` matrix in `CameraInfo`) for a different camera or resolution. The application does not automatically rectify color images; use a calibrated, appropriately rectified/registered stream for accurate projection.

Supported depth encodings are `16UC1` in millimeters and `32FC1` in meters. Zero, negative, NaN, or infinite depth produces an invalid-depth message. The node rejects mismatched image dimensions, but it cannot verify physical registration from dimensions alone.

## Run

In a terminal with the driver workspace and inference environment sourced, change to this repository:

```bash
source /opt/ros/noetic/setup.bash
source ~/ros_ws/devel/setup.bash
source .venv-ros/bin/activate
python fire_cam.py --config config/camera.json
```

Press `q` in the display window, or use Ctrl+C. For operation without display windows:

```bash
python fire_cam.py --config config/camera.json --headless
```

The node uses approximate synchronization with a queue of 10 images and a 0.1-second tolerance, matching the original program. The coordinates are camera-relative and are logged/displayed; no world-frame transform or ROS coordinate publisher is implemented.

## Troubleshooting

- Missing ROS modules: source ROS and the camera workspace, and use the environment created with `--system-site-packages`.
- No callbacks: check both configured topics and image timestamps using ROS tools.
- Invalid depth: inspect the raw depth stream; objects outside the sensor range or poor depth returns can yield zero.
- Incorrect coordinates: verify depth registration, resolution, depth units, color intrinsics, and distortion handling.
- Missing CUDA: start with CPU inference; install a compatible GPU ONNX Runtime build only when the CUDA/cuDNN environment is available.
