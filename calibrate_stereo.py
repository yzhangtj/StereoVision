#!/usr/bin/env python3
"""
Stereo Camera Calibration Image Collector

This script helps collect chessboard calibration images from dual cameras
for accurate stereo calibration.
"""

import cv2
import numpy as np
import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Collect stereo calibration images")
    parser.add_argument('--cam0', type=int, default=0, help='Index of the left camera (default: 0)')
    parser.add_argument('--cam1', type=int, default=1, help='Index of the right camera (default: 1)')
    parser.add_argument('--width', type=int, default=640, help='Width of each camera frame (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Height of each camera frame (default: 480)')
    parser.add_argument('--output_dir', type=str, default='calibration', help='Output directory for calibration images')
    parser.add_argument('--chessboard_size', type=str, default='9x6', help='Chessboard size as WxH (default: 9x6)')
    
    args = parser.parse_args()
    
    # Parse chessboard size
    cb_width, cb_height = map(int, args.chessboard_size.split('x'))
    chessboard_size = (cb_width, cb_height)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize cameras
    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open cameras")
        return 1
    
    # Set camera properties
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    print(f"Collecting calibration images...")
    print(f"Chessboard size: {chessboard_size}")
    print(f"Output directory: {args.output_dir}")
    print("\nInstructions:")
    print("- Hold a chessboard pattern in front of both cameras")
    print("- Press SPACE to capture when chessboard is detected in both cameras")
    print("- Press 'q' to quit")
    print("- Collect at least 10-20 good calibration pairs")
    
    image_count = 0
    
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("Error reading from cameras")
            break
        
        # Convert to grayscale for chessboard detection
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret0, corners0 = cv2.findChessboardCorners(gray0, chessboard_size, None)
        ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
        
        # Draw chessboard corners if found
        display0 = frame0.copy()
        display1 = frame1.copy()
        
        if ret0:
            cv2.drawChessboardCorners(display0, chessboard_size, corners0, ret0)
        if ret1:
            cv2.drawChessboardCorners(display1, chessboard_size, corners1, ret1)
        
        # Status indicators
        status0 = "FOUND" if ret0 else "NOT FOUND"
        status1 = "FOUND" if ret1 else "NOT FOUND"
        color0 = (0, 255, 0) if ret0 else (0, 0, 255)
        color1 = (0, 255, 0) if ret1 else (0, 0, 255)
        
        cv2.putText(display0, f"Left: {status0}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color0, 2)
        cv2.putText(display1, f"Right: {status1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2)
        cv2.putText(display0, f"Images: {image_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display1, f"Images: {image_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Combine frames for display
        combined = np.hstack((display0, display1))
        
        cv2.imshow('Stereo Calibration - Press SPACE to capture, Q to quit', combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            if ret0 and ret1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                left_filename = os.path.join(args.output_dir, f"left_{timestamp}.png")
                right_filename = os.path.join(args.output_dir, f"right_{timestamp}.png")
                
                cv2.imwrite(left_filename, frame0)
                cv2.imwrite(right_filename, frame1)
                
                image_count += 1
                print(f"Captured pair {image_count}: {timestamp}")
            else:
                print("Cannot capture: Chessboard not found in both cameras")
        
        elif key == ord('q'):  # Quit
            break
    
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    
    print(f"\nCalibration complete! Captured {image_count} image pairs.")
    if image_count >= 10:
        print("You have enough images for calibration. Run the main script to process stereo pairs.")
    else:
        print("Warning: You should collect at least 10 good calibration pairs for accurate results.")
    
    return 0


if __name__ == "__main__":
    exit(main()) 