# StereoVision

This repository contains a simple script for viewing two USB cameras side by side using OpenCV.

## Requirements

- Python 3
- `opencv-python` package (`pip install opencv-python`)
- `numpy` package (`pip install numpy`)

## Usage

1. Connect your dual USB cameras to the system. They will typically show up as camera indices 0 and 1. On Linux these may map to `/dev/video0` and `/dev/video1`, while on macOS you simply use the index numbers.
2. Run the viewer script:

```bash
python dual_camera_viewer.py --cam0 0 --cam1 1
```

Press `q` in the display window to exit.

### macOS notes
- macOS does not provide `/dev/video` device paths. Use integer camera indices (0, 1, etc.).
- If you have trouble opening cameras, try specifying the AVFoundation backend:
  ```bash
  python dual_camera_viewer.py --cam0 0 --cam1 1 --backend avfoundation
  ```
