# Publication verification

Verified on October 5, 2026, using Python 3.9.13 on Windows with the versions in `requirements-tested.txt`.

| Check | Result |
| --- | --- |
| Python syntax | All published Python files parse successfully |
| Automated checks | 9 tests passed, including actual CPU ONNX inference |
| Model loading | Input `[1,3,320,320]`; output `[1,6300,6]`; CPU provider |
| Complete local video processing | 2,975 input frames processed, 3,803 returned detection boxes |
| Annotated video writing | Output reopened and all 2,975 frames decoded at 426 x 240 |
| Camera CLI help | Accessible without a ROS installation |
| Publication text review | English source/documentation; no student identifier, personal machine path, or obvious credential assignment found |
| Git whitespace check | Passed |

The returned detection count is a processing check, not an accuracy measurement. The complete-video test used local footage excluded from publication because its redistribution terms were not recorded. The first output frame was visually inspected; no test output is presented as a fabricated experiment.

The ONNX file is unchanged from the original archive. Its SHA-256 is:

```text
c1b3738055caacfb2b96edcfda2b6c9bc978601e78dc4cf5bd835225cb825b94
```

Not verified during publication: physical RGB-D operation, camera driver installation, calibration/registration accuracy, CUDA execution, a fresh Linux/ROS dependency installation, or historical thesis performance. See [results](results.md) and [ROS setup](ros-setup.md) for the distinction between preserved experiments and current checks.
