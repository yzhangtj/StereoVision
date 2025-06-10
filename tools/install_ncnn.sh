#!/bin/bash

# Install NCNN Tools for YOLOv8 Export
# This script downloads and installs ncnn tools including onnx2ncnn

set -e

NCNN_VERSION="20250503"
INSTALL_DIR="$HOME/.local/bin"
TEMP_DIR="/tmp/ncnn_install"

echo "🚀 Installing NCNN Tools for YOLOv8 Export"
echo "========================================="

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case $OS in
    Darwin)
        PLATFORM="macos"
        ;;
    Linux)
        if [[ $ARCH == "x86_64" ]]; then
            PLATFORM="ubuntu-2204"
        else
            echo "❌ Unsupported Linux architecture: $ARCH"
            echo "ℹ️  Please download manually from: https://github.com/Tencent/ncnn/releases"
            exit 1
        fi
        ;;
    *)
        echo "❌ Unsupported OS: $OS"
        echo "ℹ️  Please download manually from: https://github.com/Tencent/ncnn/releases"
        exit 1
        ;;
esac

echo "📍 Detected platform: $PLATFORM"
echo "📁 Install directory: $INSTALL_DIR"

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Download URL
DOWNLOAD_URL="https://github.com/Tencent/ncnn/releases/download/$NCNN_VERSION/ncnn-$NCNN_VERSION-$PLATFORM.zip"

echo "⬇️  Downloading NCNN tools..."
echo "   URL: $DOWNLOAD_URL"

# Download
if command -v wget >/dev/null 2>&1; then
    wget -O ncnn.zip "$DOWNLOAD_URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L -o ncnn.zip "$DOWNLOAD_URL"
else
    echo "❌ Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Extract
echo "📦 Extracting archive..."
if command -v unzip >/dev/null 2>&1; then
    unzip -q ncnn.zip
else
    echo "❌ unzip not found. Please install unzip."
    exit 1
fi

# Find extracted directory
EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "ncnn-*" | head -1)
if [[ -z "$EXTRACTED_DIR" ]]; then
    echo "❌ Could not find extracted NCNN directory"
    exit 1
fi

echo "📁 Found extracted directory: $EXTRACTED_DIR"

# Copy tools
echo "📋 Installing tools to $INSTALL_DIR..."

# Find and copy onnx2ncnn
ONNX2NCNN_PATH=$(find "$EXTRACTED_DIR" -name "onnx2ncnn" -type f | head -1)
if [[ -n "$ONNX2NCNN_PATH" ]]; then
    cp "$ONNX2NCNN_PATH" "$INSTALL_DIR/"
    chmod +x "$INSTALL_DIR/onnx2ncnn"
    echo "✅ Installed: onnx2ncnn"
else
    echo "❌ Could not find onnx2ncnn in the archive"
    exit 1
fi

# Copy other useful tools if available
for tool in ncnn2mem ncnn2table ncnnoptimize; do
    TOOL_PATH=$(find "$EXTRACTED_DIR" -name "$tool" -type f | head -1)
    if [[ -n "$TOOL_PATH" ]]; then
        cp "$TOOL_PATH" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/$tool"
        echo "✅ Installed: $tool"
    fi
done

# Cleanup
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "🎉 NCNN tools installed successfully!"
echo "📍 Location: $INSTALL_DIR"
echo ""
echo "🔧 Add to PATH (add to ~/.bashrc or ~/.zshrc):"
echo "   export PATH=\"\$PATH:$INSTALL_DIR\""
echo ""
echo "🧪 Test installation:"
echo "   source ~/.bashrc  # or ~/.zshrc"
echo "   onnx2ncnn"
echo ""
echo "🚀 Ready to export YOLOv8 models:"
echo "   python tools/export_yolo_ncnn.py --weights yolov8s.pt --imgsz 960" 