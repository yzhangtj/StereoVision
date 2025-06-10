#!/usr/bin/env python3
"""
Export YOLOv8 Models for Android NCNN Deployment

This script exports YOLOv8 models to NCNN format for on-device Android inference:
1. Exports model to ONNX format with optimized settings
2. Simplifies ONNX graph using onnxsim
3. Converts to NCNN param/bin format using onnx2ncnn
4. Outputs all files to build/ncnn/ directory

Usage:
    python tools/export_yolo_ncnn.py --weights yolov8s.pt --imgsz 960
    adb push build/ncnn/*.bin /sdcard/models/
    adb push build/ncnn/*.param /sdcard/models/
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
import shutil

def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    warnings = []
    
    # Check for onnxsim (optional)
    try:
        subprocess.run(['onnxsim', '--help'], capture_output=True, check=True)
        print("✅ onnxsim found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.append('onnxsim')
        print("⚠️  onnxsim not found (optional - use --skip-onnxsim)")
    
    # Check for onnx2ncnn (optional - will export ONNX only if missing)
    try:
        subprocess.run(['onnx2ncnn'], capture_output=True)
        print("✅ onnx2ncnn found")
    except FileNotFoundError:
        warnings.append('onnx2ncnn')
        print("⚠️  onnx2ncnn not found (will export ONNX only)")
    
    if warnings:
        print(f"\n⚠️  Missing optional tools: {', '.join(warnings)}")
        print("\nFor complete NCNN export, install:")
        if 'onnx2ncnn' in warnings:
            print("  Download from: https://github.com/Tencent/ncnn/releases")
            print("  Or run: ./tools/install_ncnn.sh")
        if 'onnxsim' in warnings:
            print("  conda install -c conda-forge onnx")
            print("  pip install onnxsim (requires cmake)")
        print("\n🔄 Proceeding with available tools...\n")
    
    return True

def export_to_onnx(weights_path, imgsz, output_dir):
    """Export YOLOv8 model to ONNX format."""
    print(f"\n🔄 Exporting {weights_path} to ONNX...")
    start_time = time.time()
    
    try:
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(weights_path)
        print(f"📦 Loaded YOLOv8 model: {weights_path}")
        
        # Export to ONNX with Android-optimized settings
        onnx_path = model.export(
            format='onnx',
            imgsz=imgsz,
            opset=12,           # Compatible opset version
            dynamic=False,      # Fixed input size for mobile
            optimize=False,     # We'll use onnxsim instead
            simplify=False,     # We'll use onnxsim separately
            int8=False,         # Keep FP32 for now
            half=False          # Keep FP32 for compatibility
        )
        
        # Move ONNX file to output directory
        onnx_filename = f"yolov8s_{imgsz}.onnx"
        output_onnx = output_dir / onnx_filename
        shutil.move(onnx_path, output_onnx)
        
        elapsed = time.time() - start_time
        print(f"✅ ONNX export completed in {elapsed:.2f}s")
        print(f"📁 Saved to: {output_onnx}")
        
        return output_onnx
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

def simplify_onnx(onnx_path, output_dir):
    """Simplify ONNX model using onnxsim."""
    print(f"\n🔄 Simplifying ONNX model...")
    start_time = time.time()
    
    simplified_path = output_dir / f"{onnx_path.stem}_simplified.onnx"
    
    try:
        cmd = [
            'onnxsim',
            str(onnx_path),
            str(simplified_path),
            '--input-shape', f'1,3,{args.imgsz},{args.imgsz}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        elapsed = time.time() - start_time
        print(f"✅ ONNX simplification completed in {elapsed:.2f}s")
        print(f"📁 Simplified model: {simplified_path}")
        
        return simplified_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ONNX simplification failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return onnx_path  # Return original if simplification fails

def convert_to_ncnn(onnx_path, output_dir):
    """Convert ONNX model to NCNN format."""
    print(f"\n🔄 Converting to NCNN format...")
    start_time = time.time()
    
    base_name = onnx_path.stem.replace('_simplified', '')
    param_path = output_dir / f"{base_name}.param"
    bin_path = output_dir / f"{base_name}.bin"
    
    try:
        cmd = [
            'onnx2ncnn',
            str(onnx_path),
            str(param_path),
            str(bin_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        elapsed = time.time() - start_time
        print(f"✅ NCNN conversion completed in {elapsed:.2f}s")
        print(f"📁 NCNN param: {param_path}")
        print(f"📁 NCNN bin: {bin_path}")
        
        # Verify output files exist
        if not param_path.exists() or not bin_path.exists():
            raise FileNotFoundError("NCNN output files not created")
        
        # Print file sizes
        param_size = param_path.stat().st_size / 1024  # KB
        bin_size = bin_path.stat().st_size / (1024 * 1024)  # MB
        print(f"📊 File sizes: {param_path.name} ({param_size:.1f}KB), {bin_path.name} ({bin_size:.1f}MB)")
        
        return param_path, bin_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ NCNN conversion failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None, None

def create_deployment_info(output_dir, weights_path, imgsz, param_path, bin_path):
    """Create deployment information file."""
    info_file = output_dir / "deployment_info.txt"
    
    info_content = f"""# YOLOv8 NCNN Deployment Information
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Model Details
Original weights: {weights_path}
Input size: {imgsz}x{imgsz}
Format: NCNN

## Files
Param file: {param_path.name}
Bin file: {bin_path.name}

## Android Deployment
1. Push files to device:
   adb push {param_path.name} /sdcard/models/
   adb push {bin_path.name} /sdcard/models/

2. Load in Android app:
   ncnn::Net net;
   net.load_param("/sdcard/models/{param_path.name}");
   net.load_model("/sdcard/models/{bin_path.name}");

## Input/Output
Input: 1x3x{imgsz}x{imgsz} (RGB, normalized 0-1)
Output: Nx85 (N detections, 85 = 4 bbox + 1 conf + 80 classes)

## Classes
{len(get_yolo_classes())} COCO classes supported
"""
    
    with open(info_file, 'w') as f:
        f.write(info_content)
    
    print(f"📝 Deployment info saved: {info_file}")

def get_yolo_classes():
    """Get YOLO class names."""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Use nano for class names
        return model.names
    except:
        return {}

def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 models for Android NCNN deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/export_yolo_ncnn.py --weights yolov8s.pt --imgsz 960
  python tools/export_yolo_ncnn.py --weights yolov8n.pt --imgsz 640
  python tools/export_yolo_ncnn.py --weights custom_model.pt --imgsz 512

Android deployment:
  adb push build/ncnn/*.param /sdcard/models/
  adb push build/ncnn/*.bin /sdcard/models/
        """
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default='yolov8s.pt',
        help='Path to YOLOv8 weights file (default: yolov8s.pt)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=960,
        help='Input image size for export (default: 960)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='build/ncnn',
        help='Output directory (default: build/ncnn)'
    )
    
    parser.add_argument(
        '--skip-onnxsim',
        action='store_true',
        help='Skip ONNX simplification step'
    )
    
    parser.add_argument(
        '--keep-onnx',
        action='store_true',
        help='Keep intermediate ONNX files'
    )
    
    global args
    args = parser.parse_args()
    
    print("🚀 YOLOv8 NCNN Export Tool")
    print("=" * 50)
    
    # Validate inputs
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ Weights file not found: {weights_path}")
        return 1
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Weights: {weights_path}")
    print(f"📐 Image size: {args.imgsz}x{args.imgsz}")
    print(f"📂 Output: {output_dir}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing required dependencies. Please install them before continuing.")
        return 1
    
    # Auto-enable skip-onnxsim if not available
    try:
        subprocess.run(['onnxsim', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if not args.skip_onnxsim:
            print("\n⚠️  Auto-enabling --skip-onnxsim (onnxsim not available)")
            args.skip_onnxsim = True
    
    total_start = time.time()
    
    try:
        # Step 1: Export to ONNX
        onnx_path = export_to_onnx(weights_path, args.imgsz, output_dir)
        if not onnx_path:
            return 1
        
        # Step 2: Simplify ONNX (optional)
        if not args.skip_onnxsim:
            simplified_path = simplify_onnx(onnx_path, output_dir)
            final_onnx = simplified_path
        else:
            print("\n⏭️  Skipping ONNX simplification")
            final_onnx = onnx_path
        
        # Step 3: Convert to NCNN (if onnx2ncnn available)
        param_path, bin_path = None, None
        try:
            subprocess.run(['onnx2ncnn'], capture_output=True)
            param_path, bin_path = convert_to_ncnn(final_onnx, output_dir)
            if param_path and bin_path:
                # Step 4: Create deployment info
                create_deployment_info(output_dir, weights_path, args.imgsz, param_path, bin_path)
        except FileNotFoundError:
            print("\n⚠️  onnx2ncnn not available - ONNX export completed")
            print("📋 To complete NCNN conversion:")
            print("   1. Install onnx2ncnn: ./tools/install_ncnn.sh")
            print(f"   2. Run: onnx2ncnn build/ncnn/{final_onnx.name} build/ncnn/yolov8s_960.param build/ncnn/yolov8s_960.bin")
        
        # Cleanup intermediate files (only if NCNN conversion succeeded)
        if not args.keep_onnx and param_path and bin_path:
            print(f"\n🧹 Cleaning up intermediate ONNX files...")
            for onnx_file in output_dir.glob("*.onnx"):
                onnx_file.unlink()
                print(f"   Removed: {onnx_file.name}")
        elif not param_path or not bin_path:
            print(f"\n📁 Keeping ONNX file for manual conversion: {final_onnx.name}")
        
        total_elapsed = time.time() - total_start
        print(f"\n🎉 Export completed successfully in {total_elapsed:.2f}s")
        print(f"📦 Output files in: {output_dir}")
        
        if param_path and bin_path:
            print(f"   • {param_path.name}")
            print(f"   • {bin_path.name}")
            print(f"   • deployment_info.txt")
            
            print(f"\n📱 Android deployment commands:")
            print(f"   adb push {output_dir}/{param_path.name} /sdcard/models/")
            print(f"   adb push {output_dir}/{bin_path.name} /sdcard/models/")
        else:
            print(f"   • {final_onnx.name} (ONNX format)")
            print(f"\n📱 Next steps for Android deployment:")
            print(f"   1. Install onnx2ncnn tools")
            print(f"   2. Convert to NCNN format")
            print(f"   3. Push .param and .bin files to Android")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n❌ Export interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 