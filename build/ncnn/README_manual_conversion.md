# Manual ONNX to NCNN Conversion Guide

## Current Status
✅ **ONNX Export Complete**: `yolov8s_960.onnx` (43MB)  
⚠️ **NCNN Conversion Pending**: Need `onnx2ncnn` tool

## Option 1: Install NCNN Tools (Recommended)

### Download Pre-built Tools
```bash
# Download for macOS
curl -L -O https://github.com/Tencent/ncnn/releases/download/20240930/ncnn-20240930-macos.zip
unzip ncnn-20240930-macos.zip
export PATH=$PATH:$(pwd)/ncnn-20240930-macos/bin

# Or install via our script
./tools/install_ncnn.sh
```

### Manual Conversion
```bash
# Convert ONNX to NCNN format
onnx2ncnn yolov8s_960.onnx yolov8s_960.param yolov8s_960.bin

# Verify output files
ls -la *.param *.bin
```

## Option 2: Use Online Converter

Visit: https://convertmodel.com/
- Upload: `yolov8s_960.onnx`
- Convert: ONNX → NCNN
- Download: `.param` and `.bin` files

## Option 3: Re-run Export Script

Once you have `onnx2ncnn` installed:
```bash
python tools/export_yolo_ncnn.py --weights yolov8s.pt --imgsz 960
```

## Android Deployment

Once you have both files:
```bash
# Push to Android device
adb push yolov8s_960.param /sdcard/models/
adb push yolov8s_960.bin /sdcard/models/

# In your Android app
ncnn::Net net;
net.load_param("/sdcard/models/yolov8s_960.param");
net.load_model("/sdcard/models/yolov8s_960.bin");
```

## Model Specifications

- **Input**: 1×3×960×960 (RGB, normalized 0-1)
- **Output**: N×84 detections (4 bbox + 1 conf + 80 classes)  
- **Classes**: 80 COCO object classes
- **Format**: NCNN (optimized for mobile inference)

## Expected Performance

- **Model Size**: ~20MB (.param + .bin combined)
- **Inference Speed**: ~50-200ms on modern Android devices
- **Memory Usage**: ~100-200MB during inference 