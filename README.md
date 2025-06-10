# Stereo Vision System

A comprehensive Python-based stereo vision system that processes dual-camera stereo images to generate accurate depth maps. The system performs stereo calibration, rectification, and disparity computation using OpenCV.

## Features

- **Stereo Calibration**: Automatic camera calibration using chessboard patterns
- **Image Rectification**: Corrects lens distortion and aligns stereo pairs
- **Disparity Computation**: Uses OpenCV's StereoBM or StereoSGBM algorithms
- **Depth Map Generation**: Converts disparity to depth information
- **Flexible Input**: Processes test folders containing stereo image pairs
- **Multiple Algorithms**: Supports both Block Matching (BM) and Semi-Global Block Matching (SGBM)

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Or install manually:**
   ```bash
   pip install opencv-python>=4.5.0 numpy>=1.20.0
   ```

## Usage

### Basic Usage

Process a stereo image pair to generate a depth map:

```bash
python generate_depth_map.py --input_folder test/000040
```

### Advanced Options

```bash
# Use Block Matching algorithm instead of SGBM
python generate_depth_map.py --input_folder test/000040 --algorithm BM

# Save intermediate results (rectified images, disparity map)
python generate_depth_map.py --input_folder test/000040 --save_intermediate

# Combine options
python generate_depth_map.py --input_folder test/000040 --algorithm SGBM --save_intermediate
```

### Arguments

- `--input_folder`: Path to folder containing `left.png` and `right.png` (required)
- `--algorithm`: Stereo matching algorithm - `BM` or `SGBM` (default: SGBM)
- `--save_intermediate`: Save rectified images and disparity map

## Input Requirements

Each test folder should contain:
- `left.png` - Left camera image
- `right.png` - Right camera image

The system ignores other files like `left_disp.png`, `right_disp.png`, etc.

## Output

The system generates:
- `depth_map.png` - Main depth map output (0-255 normalized)
- `disparity_map.png` - Raw disparity map (if `--save_intermediate` is used)
- `left_rectified.png` / `right_rectified.png` - Rectified images (if `--save_intermediate` is used)

## Calibration

### Automatic Calibration (Recommended)

If you have chessboard calibration images:

1. Create a `calibration/` folder in the project root
2. Place chessboard calibration images named as `left_XXXX.png` and `right_XXXX.png`
3. Run the main script - it will automatically detect and use calibration images
4. Calibration parameters are saved to `stereo_calibration.json` for reuse

### Collect Calibration Images

Use the provided calibration script to collect chessboard images:

```bash
python calibrate_stereo.py --cam0 0 --cam1 1
```

Instructions:
- Hold a chessboard pattern (9x6 inner corners by default) in front of both cameras
- Press SPACE when the chessboard is detected in both cameras
- Press 'q' to quit
- Collect at least 10-20 good calibration pairs

### Default Parameters

If no calibration is available, the system uses estimated default parameters:
- Focal length: 80% of image width
- Principal point: Image center
- No lens distortion
- 10cm baseline between cameras

## Algorithms

### StereoSGBM (Default)
- **Pros**: Higher accuracy, better handling of textureless regions
- **Cons**: Slower computation
- **Best for**: General purpose, high-quality depth maps

### StereoBM
- **Pros**: Faster computation
- **Cons**: Requires good texture, less accurate
- **Best for**: Real-time applications, textured scenes

## File Structure

```
StereoVision/
├── generate_depth_map.py      # Main stereo processing script
├── calibrate_stereo.py        # Calibration image collector
├── dual_camera_viewer.py      # Live dual camera viewer
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── test/                      # Test image folders
│   ├── 000040/
│   │   ├── left.png
│   │   ├── right.png
│   │   └── depth_map.png      # Generated
│   └── 000080/
│       ├── left.png
│       ├── right.png
│       └── depth_map.png      # Generated
├── calibration/               # Calibration images (optional)
│   ├── left_001.png
│   ├── right_001.png
│   └── ...
└── stereo_calibration.json    # Saved calibration (auto-generated)
```

## Technical Details

### Processing Pipeline

1. **Image Loading**: Load left and right stereo images
2. **Calibration**: 
   - Load existing calibration or
   - Perform new calibration with chessboard images or
   - Use default parameters
3. **Rectification**: Apply lens correction and stereo rectification
4. **Disparity Computation**: Calculate pixel disparities using stereo matching
5. **Depth Conversion**: Convert disparity to depth using camera geometry
6. **Output**: Save normalized depth map

### Key Parameters

**StereoSGBM Settings:**
- Window size: 3x3
- Disparity range: 80 pixels (16*5)
- P1/P2: Smoothness penalties
- Uniqueness ratio: 10%

**StereoBM Settings:**
- Block size: 15x15
- Disparity range: 80 pixels
- Pre-filter: X-Sobel
- Texture threshold: 10

## Troubleshooting

### Common Issues

1. **"Could not find left.png or right.png"**
   - Ensure your test folder contains exactly these filenames
   - Check file extensions are lowercase `.png`

2. **Poor depth map quality**
   - Try different algorithms (BM vs SGBM)
   - Ensure good lighting and texture in scenes
   - Consider proper stereo calibration with chessboard

3. **Calibration fails**
   - Ensure chessboard has 9x6 inner corners
   - Collect more calibration images (15-20 recommended)
   - Ensure chessboard is clearly visible in both cameras

### Parameter Tuning

For better results, you can modify the stereo matching parameters in `generate_depth_map.py`:

- Increase `numDisparities` for scenes with larger depth range
- Adjust `blockSize` for different texture levels
- Modify `P1`/`P2` for SGBM smoothness control

## Android Export

Export YOLOv8 models for Android NCNN deployment:

### Basic Export
```bash
python tools/export_yolo_ncnn.py --weights yolov8s.pt --imgsz 960
```

### Deploy to Android Device
```bash
# Install dependencies first
pip install onnxsim

# Export model
python tools/export_yolo_ncnn.py --weights yolov8s.pt --imgsz 960

# Push to Android device
adb push build/ncnn/*.param /sdcard/models/
adb push build/ncnn/*.bin /sdcard/models/
```

### Export Options
- `--weights`: YOLOv8 weights file (default: yolov8s.pt)
- `--imgsz`: Input image size (default: 960)
- `--output`: Output directory (default: build/ncnn)
- `--skip-onnxsim`: Skip ONNX simplification
- `--keep-onnx`: Keep intermediate ONNX files

### Dependencies for Export
```bash
# Install Python dependencies
pip install -r dev-requirements.txt

# Install NCNN tools (automatic installer)
chmod +x tools/install_ncnn.sh
./tools/install_ncnn.sh

# Add to PATH if needed
export PATH="$PATH:$HOME/.local/bin"
```

**Note**: If automatic installation fails, download `onnx2ncnn` manually:
- Download from: https://github.com/Tencent/ncnn/releases
- Or build from source: https://github.com/Tencent/ncnn

### Complete Android Deployment Pipeline
```bash
# 1. Export model for Android
python tools/export_yolo_ncnn.py --weights yolov8s.pt --imgsz 960

# 2. Verify output files
ls build/ncnn/
# Should show: yolov8s.param and yolov8s.bin

# 3. Deploy to Android device
adb push build/ncnn/*.param /sdcard/models/
adb push build/ncnn/*.bin /sdcard/models/
```

## Examples

Process multiple test folders:
```bash
for folder in test/*/; do
    python generate_depth_map.py --input_folder "$folder"
done
```

Batch processing with different algorithms:
```bash
python generate_depth_map.py --input_folder test/000040 --algorithm BM
python generate_depth_map.py --input_folder test/000040 --algorithm SGBM
```

## Requirements

- Python 3.6+
- OpenCV 4.5+
- NumPy 1.20+
- macOS, Linux, or Windows

## License

This project is open source. Feel free to modify and distribute.
