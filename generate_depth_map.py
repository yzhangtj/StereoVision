#!/usr/bin/env python3
"""
Stereo Vision System for Depth Map Generation

This system processes stereo image pairs (left.png and right.png) from test folders,
performs stereo calibration and rectification, computes disparity maps, and saves
the resulting depth map.
"""

import cv2
import numpy as np
import argparse
import os
import glob
import json
from pathlib import Path

# HITNet imports
try:
    import onnxruntime
    HITNET_AVAILABLE = True
except ImportError:
    HITNET_AVAILABLE = False
    print("Warning: onnxruntime not available. HITNet algorithm will be disabled.")


class StereoVisionSystem:
    def __init__(self):
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation matrix between cameras
        self.T = None  # Translation vector between cameras
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        self.R1 = None  # Rectification rotation matrix for left camera
        self.R2 = None  # Rectification rotation matrix for right camera
        self.P1 = None  # Projection matrix for left camera
        self.P2 = None  # Projection matrix for right camera
        self.Q = None   # Disparity-to-depth mapping matrix
        self.roi_left = None
        self.roi_right = None
        self.map1_left = None
        self.map2_left = None
        self.map1_right = None
        self.map2_right = None
        
        # Chessboard parameters
        self.chessboard_size = (9, 6)  # Inner corners (width, height)
        self.square_size = 1.0  # Size of chessboard squares (in arbitrary units)
        
        # HITNet model
        self.hitnet_session = None
        self.hitnet_input_height = 480
        self.hitnet_input_width = 640
        
    def find_chessboard_images(self, calibration_folder):
        """Find chessboard calibration images in the specified folder."""
        left_pattern = os.path.join(calibration_folder, "left_*.png")
        right_pattern = os.path.join(calibration_folder, "right_*.png")
        
        left_images = sorted(glob.glob(left_pattern))
        right_images = sorted(glob.glob(right_pattern))
        
        # Filter to match pairs
        left_base = [os.path.basename(img).replace("left_", "").replace(".png", "") for img in left_images]
        right_base = [os.path.basename(img).replace("right_", "").replace(".png", "") for img in right_images]
        
        common = set(left_base) & set(right_base)
        
        left_calib = [os.path.join(calibration_folder, f"left_{base}.png") for base in sorted(common)]
        right_calib = [os.path.join(calibration_folder, f"right_{base}.png") for base in sorted(common)]
        
        return left_calib, right_calib
    
    def prepare_object_points(self, num_images):
        """Prepare 3D object points for chessboard calibration."""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        return [objp for _ in range(num_images)]
    
    def calibrate_cameras(self, left_images, right_images):
        """Perform stereo calibration using chessboard images."""
        print("Starting stereo calibration...")
        
        # Prepare object points
        objpoints = []  # 3D points in real world space
        imgpoints_left = []  # 2D points in left image plane
        imgpoints_right = []  # 2D points in right image plane
        
        print(f"Processing {len(left_images)} calibration image pairs...")
        
        for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            print(f"Processing calibration pair {i+1}/{len(left_images)}")
            
            # Read images
            img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
            
            if img_left is None or img_right is None:
                print(f"Warning: Could not load images {left_path} or {right_path}")
                continue
            
            # Find chessboard corners
            ret_left, corners_left = cv2.findChessboardCorners(img_left, self.chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(img_right, self.chessboard_size, None)
            
            if ret_left and ret_right:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(img_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(img_right, corners_right, (11, 11), (-1, -1), criteria)
                
                # Prepare object points
                objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
                objp *= self.square_size
                
                objpoints.append(objp)
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
                print(f"  Successfully found corners in pair {i+1}")
            else:
                print(f"  Warning: Could not find chessboard corners in pair {i+1}")
        
        if len(objpoints) < 10:
            raise ValueError(f"Not enough valid calibration pairs found. Need at least 10, got {len(objpoints)}")
        
        print(f"Using {len(objpoints)} valid calibration pairs")
        
        # Get image size
        img_shape = img_left.shape[::-1]  # (width, height)
        
        print("Calibrating individual cameras...")
        
        # Calibrate left camera
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints, imgpoints_left, img_shape, None, None)
        
        # Calibrate right camera
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints, imgpoints_right, img_shape, None, None)
        
        print("Performing stereo calibration...")
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx_left, dist_left, mtx_right, dist_right,
            img_shape, flags=flags
        )
        
        # Store calibration results
        self.camera_matrix_left = mtx_left
        self.camera_matrix_right = mtx_right
        self.dist_coeffs_left = dist_left
        self.dist_coeffs_right = dist_right
        self.R = R
        self.T = T
        self.E = E
        self.F = F
        
        print(f"Stereo calibration completed with RMS error: {ret_stereo:.4f}")
        
        return img_shape
    
    def compute_rectification(self, img_shape):
        """Compute stereo rectification parameters."""
        print("Computing stereo rectification...")
        
        # Stereo rectification
        self.R1, self.R2, self.P1, self.P2, self.Q, roi_left, roi_right = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            img_shape, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.9  # Keep more of the original image
        )
        
        self.roi_left = roi_left
        self.roi_right = roi_right
        
        # Compute rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1, img_shape, cv2.CV_16SC2)
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2, img_shape, cv2.CV_16SC2)
        
        print("Rectification computation completed")
    
    def use_default_calibration(self, img_shape):
        """Use default camera parameters when calibration images are not available."""
        print("Using default camera calibration parameters...")
        
        width, height = img_shape
        
        # Default camera parameters (you may need to adjust these based on your cameras)
        focal_length = width * 0.8  # Rough estimation
        center_x = width / 2.0
        center_y = height / 2.0
        
        # Camera matrices
        self.camera_matrix_left = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.camera_matrix_right = self.camera_matrix_left.copy()
        
        # Distortion coefficients (assuming minimal distortion)
        self.dist_coeffs_left = np.zeros((4, 1))
        self.dist_coeffs_right = np.zeros((4, 1))
        
        # Stereo parameters (rough estimation)
        self.R = np.eye(3, dtype=np.float64)  # No rotation between cameras
        self.T = np.array([-0.1, 0.0, 0.0], dtype=np.float64)  # 10cm baseline (adjust as needed)
        
        # Compute rectification with default parameters
        self.R1, self.R2, self.P1, self.P2, self.Q, roi_left, roi_right = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            img_shape, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.9
        )
        
        self.roi_left = roi_left
        self.roi_right = roi_right
        
        # Compute rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1, img_shape, cv2.CV_16SC2)
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2, img_shape, cv2.CV_16SC2)
        
        print("Default calibration setup completed")
    
    def initialize_hitnet(self):
        """Initialize HITNet model for dense stereo matching."""
        if not HITNET_AVAILABLE:
            raise ImportError("onnxruntime is not available. Install with: pip install onnxruntime")
        
        model_path = "ONNX-HITNET-Stereo-Depth-estimation/models/eth3d/saved_model_480x640/model_float32.onnx"
        
        if not os.path.exists(model_path):
            print(f"Warning: HITNet model not found at {model_path}")
            print("Please download the model from: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/142_HITNET")
            print("Falling back to enhanced SGBM with dense post-processing...")
            return False
        
        try:
            print("Initializing HITNet model...")
            self.hitnet_session = onnxruntime.InferenceSession(model_path, providers=[
                'CPUExecutionProvider'])  # Remove CUDA for now
            
            # Get model input details
            model_inputs = self.hitnet_session.get_inputs()
            input_shape = model_inputs[0].shape
            self.hitnet_input_height = input_shape[2]
            self.hitnet_input_width = input_shape[3]
            
            print(f"HITNet model loaded successfully. Input size: {self.hitnet_input_width}x{self.hitnet_input_height}")
            return True
            
        except Exception as e:
            print(f"Failed to load HITNet model: {e}")
            print("Falling back to enhanced SGBM with dense post-processing...")
            return False
    
    def compute_disparity_hitnet(self, rect_left, rect_right):
        """Compute dense disparity using HITNet model or enhanced SGBM fallback."""
        
        # Try to initialize HITNet if not already done
        if self.hitnet_session is None:
            hitnet_available = self.initialize_hitnet()
            if not hitnet_available:
                # Fallback to enhanced SGBM with post-processing for density
                print("Using enhanced SGBM with density improvements...")
                return self.compute_dense_sgbm_fallback(rect_left, rect_right)
        
        print("Computing dense disparity using HITNet...")
        
        # Store original dimensions
        orig_height, orig_width = rect_left.shape[:2]
        
        # Resize images to HITNet input size
        left_resized = cv2.resize(rect_left, (self.hitnet_input_width, self.hitnet_input_height))
        right_resized = cv2.resize(rect_right, (self.hitnet_input_width, self.hitnet_input_height))
        
        # Convert to grayscale if needed (HITNet ETH3D model expects grayscale)
        if len(left_resized.shape) == 3:
            left_resized = cv2.cvtColor(left_resized, cv2.COLOR_BGR2GRAY)
        if len(right_resized.shape) == 3:
            right_resized = cv2.cvtColor(right_resized, cv2.COLOR_BGR2GRAY)
        
        # Prepare input tensor (1, 2, H, W) - concatenate left and right
        left_normalized = left_resized.astype(np.float32) / 255.0
        right_normalized = right_resized.astype(np.float32) / 255.0
        
        # Shape: (H, W, 2) -> (1, 2, H, W)
        combined = np.stack([left_normalized, right_normalized], axis=2)
        input_tensor = combined.transpose(2, 0, 1)  # (2, H, W)
        input_tensor = np.expand_dims(input_tensor, 0)  # (1, 2, H, W)
        
        # Run inference
        input_name = self.hitnet_session.get_inputs()[0].name
        output_name = self.hitnet_session.get_outputs()[0].name
        
        disparity_small = self.hitnet_session.run([output_name], {input_name: input_tensor})[0]
        disparity_small = np.squeeze(disparity_small)  # Remove batch dimension
        
        # Resize disparity back to original resolution
        scale_x = orig_width / self.hitnet_input_width
        scale_y = orig_height / self.hitnet_input_height
        
        disparity_full = cv2.resize(disparity_small, (orig_width, orig_height))
        
        # Scale disparity values to match the resolution change
        disparity_full = disparity_full * scale_x
        
        # Calculate statistics
        valid_disparities = disparity_full[disparity_full > 0]
        if len(valid_disparities) > 0:
            print(f"  Disparity range: {valid_disparities.min():.1f} - {valid_disparities.max():.1f}")
            print(f"  Valid pixels: {len(valid_disparities)}/{disparity_full.size} ({100*len(valid_disparities)/disparity_full.size:.1f}%)")
        else:
            print("  Warning: No valid disparities found!")
        
        print("HITNet disparity computation completed")
        
        return disparity_full
    
    def compute_dense_sgbm_fallback(self, rect_left, rect_right):
        """Enhanced SGBM with post-processing to create denser disparity maps."""
        print("Computing enhanced dense SGBM disparity...")
        
        # Use aggressive SGBM parameters for better coverage
        img_width = rect_left.shape[1]
        num_disparities = 128  # More disparities
        block_size = 5         # Smaller block size for more detail
        
        # Create SGBM matcher with aggressive settings
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 1 * block_size ** 2,      # Reduced penalty for smooth regions
            P2=32 * 1 * block_size ** 2,     # Reduced penalty for large changes
            disp12MaxDiff=2,                 # Allow more variation
            uniquenessRatio=5,               # Reduced uniqueness for denser results
            speckleWindowSize=50,            # Smaller speckle window
            speckleRange=16,                 # Reduced speckle range
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Compute initial disparity
        disparity = sgbm.compute(rect_left, rect_right).astype(np.float32) / 16.0
        
        # Post-processing for density improvement
        # 1. Median filter to reduce noise
        disparity_filtered = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)
        
        # 2. Fill holes using inpainting-like approach
        mask = (disparity_filtered <= 0).astype(np.uint8)
        if np.sum(mask) > 0:
            # Dilate valid regions to fill small holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            valid_mask = (disparity_filtered > 0).astype(np.uint8)
            dilated_valid = cv2.dilate(valid_mask, kernel, iterations=2)
            
            # Use the dilated valid areas to estimate missing values
            for i in range(3):  # Multiple passes for better filling
                disparity_filtered = cv2.filter2D(disparity_filtered, -1, 
                                                  np.ones((3, 3)) / 9)
                disparity_filtered[valid_mask > 0] = disparity[valid_mask > 0]
        
        # Calculate statistics
        valid_disparities = disparity_filtered[disparity_filtered > 0]
        if len(valid_disparities) > 0:
            print(f"  Disparity range: {valid_disparities.min():.1f} - {valid_disparities.max():.1f}")
            print(f"  Valid pixels: {len(valid_disparities)}/{disparity_filtered.size} ({100*len(valid_disparities)/disparity_filtered.size:.1f}%)")
        
        print("Enhanced SGBM disparity computation completed")
        
        return disparity_filtered
    
    def save_calibration(self, filepath):
        """Save calibration parameters to a file."""
        calibration_data = {
            'camera_matrix_left': self.camera_matrix_left.tolist(),
            'camera_matrix_right': self.camera_matrix_right.tolist(),
            'dist_coeffs_left': self.dist_coeffs_left.tolist(),
            'dist_coeffs_right': self.dist_coeffs_right.tolist(),
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'R1': self.R1.tolist(),
            'R2': self.R2.tolist(),
            'P1': self.P1.tolist(),
            'P2': self.P2.tolist(),
            'Q': self.Q.tolist(),
            'roi_left': self.roi_left,
            'roi_right': self.roi_right
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath):
        """Load calibration parameters from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.camera_matrix_left = np.array(data['camera_matrix_left'])
        self.camera_matrix_right = np.array(data['camera_matrix_right'])
        self.dist_coeffs_left = np.array(data['dist_coeffs_left'])
        self.dist_coeffs_right = np.array(data['dist_coeffs_right'])
        self.R = np.array(data['R'])
        self.T = np.array(data['T'])
        self.R1 = np.array(data['R1'])
        self.R2 = np.array(data['R2'])
        self.P1 = np.array(data['P1'])
        self.P2 = np.array(data['P2'])
        self.Q = np.array(data['Q'])
        self.roi_left = tuple(data['roi_left'])
        self.roi_right = tuple(data['roi_right'])
        
        print(f"Calibration loaded from {filepath}")
    
    def rectify_images(self, img_left, img_right):
        """Apply rectification to stereo image pair."""
        rect_left = cv2.remap(img_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        return rect_left, rect_right
    
    def create_epipolar_visualization(self, rect_left, rect_right, output_path):
        """Create visualization showing epipolar lines for calibration verification."""
        print("Creating epipolar line visualization...")
        
        # Convert to color for visualization
        if len(rect_left.shape) == 2:
            left_color = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
        else:
            left_color = rect_left.copy()
            
        if len(rect_right.shape) == 2:
            right_color = cv2.cvtColor(rect_right, cv2.COLOR_GRAY2BGR)
        else:
            right_color = rect_right.copy()
        
        # Draw horizontal epipolar lines every 50 pixels
        height = rect_left.shape[0]
        for y in range(0, height, 50):
            cv2.line(left_color, (0, y), (left_color.shape[1], y), (0, 255, 0), 1)
            cv2.line(right_color, (0, y), (right_color.shape[1], y), (0, 255, 0), 1)
        
        # Combine images side by side
        combined = np.hstack((left_color, right_color))
        
        # Add text labels
        cv2.putText(combined, "Left Rectified", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Right Rectified", (left_color.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Green lines should align horizontally", (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), combined)
        print(f"Epipolar visualization saved to {output_path}")

    def compute_disparity(self, rect_left, rect_right, algorithm='SGBM', 
                         block_size=None, num_disparities=None, min_disparity=None):
        """Compute disparity map from rectified stereo pair with enhanced parameters."""
        
        # HITNet uses its own dense disparity computation
        if algorithm == 'HITNET':
            return self.compute_disparity_hitnet(rect_left, rect_right)
        
        print(f"Computing disparity using {algorithm}...")
        
        # Auto-adjust parameters based on image size if not provided
        img_width = rect_left.shape[1]
        
        if block_size is None:
            # Larger block size for less noise, smaller for more detail
            block_size = 21 if img_width > 1000 else 15
            
        if num_disparities is None:
            # More disparities for wider scenes - ensure multiple of 16
            num_disparities = max(64, (img_width // 8) // 16 * 16)
            
        if min_disparity is None:
            min_disparity = 0
        
        print(f"  Block size: {block_size}")
        print(f"  Num disparities: {num_disparities}")
        print(f"  Min disparity: {min_disparity}")
        
        if algorithm == 'BM':
            # Enhanced StereoBM parameters
            stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
            stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
            stereo.setPreFilterSize(9)
            stereo.setPreFilterCap(31)
            stereo.setTextureThreshold(10)
            stereo.setUniquenessRatio(15)
            stereo.setSpeckleRange(32)
            stereo.setSpeckleWindowSize(100)
            stereo.setMinDisparity(min_disparity)
            
        else:  # SGBM
            # Enhanced StereoSGBM parameters
            stereo = cv2.StereoSGBM_create(
                minDisparity=min_disparity,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * 3 * block_size ** 2,     # Small penalty for disparity changes
                P2=32 * 3 * block_size ** 2,    # Large penalty for large changes
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        
        # Compute disparity
        disparity = stereo.compute(rect_left, rect_right)
        
        # Convert to float and normalize
        disparity_float = disparity.astype(np.float32) / 16.0
        
        # Calculate statistics
        valid_disparities = disparity_float[disparity_float > 0]
        if len(valid_disparities) > 0:
            print(f"  Disparity range: {valid_disparities.min():.1f} - {valid_disparities.max():.1f}")
            print(f"  Valid pixels: {len(valid_disparities)}/{disparity_float.size} ({100*len(valid_disparities)/disparity_float.size:.1f}%)")
        else:
            print("  Warning: No valid disparities found!")
        
        print("Disparity computation completed")
        
        return disparity_float
    
    def disparity_to_depth(self, disparity, algorithm='SGBM'):
        """Convert disparity map to depth map using Q matrix or HITNet formula."""
        
        if algorithm == 'HITNET':
            # For HITNet, use the direct formula: depth = (baseline * fx) / disparity
            # Extract baseline and focal length from calibration
            if self.T is not None and self.camera_matrix_left is not None:
                baseline = abs(self.T[0])  # Baseline is the X translation
                fx = self.camera_matrix_left[0, 0]  # Focal length
            else:
                # Use default values if calibration is not available
                baseline = 0.1  # 10cm baseline estimate
                fx = 800      # Default focal length estimate
            
            print(f"  Using baseline: {baseline:.3f}, fx: {fx:.1f}")
            
            # Avoid division by zero
            depth = np.zeros_like(disparity)
            valid_mask = disparity > 1e-6
            depth[valid_mask] = (baseline * fx) / disparity[valid_mask]
            
            # Debug: Check depth values
            if np.any(valid_mask):
                valid_depths = depth[valid_mask]
                print(f"  Debug - Depth range: {valid_depths.min():.2f} - {valid_depths.max():.2f}")
                print(f"  Debug - Disparity range for valid depths: {disparity[valid_mask].min():.2f} - {disparity[valid_mask].max():.2f}")
                
                # Check for extremely large values that might cause visualization issues
                reasonable_depths = valid_depths[valid_depths < 1000]  # Less than 1000 units
                if len(reasonable_depths) > 0:
                    print(f"  Debug - Reasonable depth range (<1000): {reasonable_depths.min():.2f} - {reasonable_depths.max():.2f}")
            
            return depth
        
        else:
            # Standard OpenCV method using Q matrix
            # Reproject to 3D
            points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
            
            # Extract depth (Z coordinate)
            depth = points_3d[:, :, 2]
            
            # Handle invalid disparities
            depth[disparity <= 0] = 0
            
            return depth
    
    def save_depth_visualizations(self, depth, output_path, algorithm):
        """Save multiple depth visualizations: grayscale, colormap, and enhanced analysis."""
        
        # Create mask for valid depth values
        valid_mask = depth > 0
        
        if not np.any(valid_mask):
            print("Warning: No valid depth values found!")
            # Save empty visualizations
            empty_img = np.zeros(depth.shape, dtype=np.uint8)
            cv2.imwrite(str(output_path / f"depth_map_{algorithm}.png"), empty_img)
            cv2.imwrite(str(output_path / f"depth_colormap_{algorithm}.png"), empty_img)
            return {
                'grayscale': str(output_path / f"depth_map_{algorithm}.png"),
                'colormap': str(output_path / f"depth_colormap_{algorithm}.png")
            }
        
        # Get depth statistics
        valid_depths = depth[valid_mask]
        depth_min = np.min(valid_depths)
        depth_max = np.max(valid_depths)
        depth_mean = np.mean(valid_depths)
        valid_percentage = 100 * np.sum(valid_mask) / depth.size
        
        print(f"Depth statistics:")
        print(f"  Range: {depth_min:.2f} - {depth_max:.2f} units")
        print(f"  Mean: {depth_mean:.2f} units") 
        print(f"  Valid pixels: {valid_percentage:.1f}%")
        
        # Handle extreme ranges by using percentile-based normalization
        if depth_max > 10 * depth_mean or depth_max > 100:  # Detect extreme ranges
            # Use 95th percentile to avoid outliers dominating the visualization
            depth_95th = np.percentile(valid_depths, 95)
            depth_5th = np.percentile(valid_depths, 5)
            
            print(f"  Extreme range detected. Using 5th-95th percentile: {depth_5th:.2f} - {depth_95th:.2f}")
            
            # Clip extreme values for better visualization
            depth_clipped = np.copy(depth)
            depth_clipped[valid_mask] = np.clip(valid_depths, depth_5th, depth_95th)
            
            # Use clipped values for normalization
            depth_for_norm = np.zeros_like(depth, dtype=np.float32)
            depth_for_norm[valid_mask] = depth_clipped[valid_mask]
            
            # Update range for display
            effective_min, effective_max = depth_5th, depth_95th
        else:
            # Normal range - use all values
            depth_for_norm = np.zeros_like(depth, dtype=np.float32)
            depth_for_norm[valid_mask] = valid_depths
            effective_min, effective_max = depth_min, depth_max
        
        # Method 1: Improved grayscale normalization using cv2.normalize
        depth_vis_gray = cv2.normalize(depth_for_norm, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis_gray = depth_vis_gray.astype(np.uint8)
        
        # Method 2: Colormap visualization (the recommended approach from your suggestion)
        depth_vis_color = cv2.normalize(depth_for_norm, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis_color = depth_vis_color.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_vis_color, cv2.COLORMAP_INFERNO)
        
        # Set invalid areas to black in colormap
        depth_colormap[~valid_mask] = [0, 0, 0]
        
        # Save both versions
        depth_gray_path = output_path / f"depth_map_{algorithm}.png"
        depth_color_path = output_path / f"depth_colormap_{algorithm}.png"
        
        cv2.imwrite(str(depth_gray_path), depth_vis_gray)
        cv2.imwrite(str(depth_color_path), depth_colormap)
        
        print(f"  Grayscale depth map: {depth_gray_path}")
        print(f"  Colormap depth map: {depth_color_path}")
        print(f"  Visualization range: {effective_min:.2f} - {effective_max:.2f} units")
        
        return {
            'grayscale': str(depth_gray_path),
            'colormap': str(depth_color_path)
        }
    
    def process_stereo_pair(self, input_folder, algorithm='SGBM', save_intermediate=False,
                           block_size=None, num_disparities=None, min_disparity=None,
                           check_calibration=False):
        """Process a stereo image pair and generate depth map with enhanced parameters."""
        input_path = Path(input_folder)
        
        # Load stereo images
        left_path = input_path / "left.png"
        right_path = input_path / "right.png"
        
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError(f"Could not find left.png or right.png in {input_folder}")
        
        print(f"Loading stereo pair from {input_folder}")
        img_left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        
        if img_left is None or img_right is None:
            raise ValueError("Could not load stereo images")
        
        img_shape = img_left.shape[::-1]  # (width, height)
        print(f"Image size: {img_shape[0]} x {img_shape[1]}")
        
        # Check if we have calibration data
        calibration_file = "stereo_calibration.json"
        if os.path.exists(calibration_file):
            print("Loading existing calibration...")
            self.load_calibration(calibration_file)
            # Recompute rectification maps for current image size
            self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
                self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1, img_shape, cv2.CV_16SC2)
            self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
                self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2, img_shape, cv2.CV_16SC2)
        else:
            # Look for calibration images
            calib_folder = "calibration"
            if os.path.exists(calib_folder):
                left_calib, right_calib = self.find_chessboard_images(calib_folder)
                if len(left_calib) > 0 and len(right_calib) > 0:
                    img_shape = self.calibrate_cameras(left_calib, right_calib)
                    self.compute_rectification(img_shape)
                    self.save_calibration(calibration_file)
                else:
                    print("No chessboard calibration images found, using default parameters")
                    self.use_default_calibration(img_shape)
            else:
                print("No calibration folder found, using default parameters")
                self.use_default_calibration(img_shape)
        
        # Rectify images
        print("Rectifying stereo images...")
        rect_left, rect_right = self.rectify_images(img_left, img_right)
        
        if save_intermediate:
            cv2.imwrite(str(input_path / "left_rectified.png"), rect_left)
            cv2.imwrite(str(input_path / "right_rectified.png"), rect_right)
            print("Saved rectified images")
        
        # Create calibration check visualization
        if check_calibration or save_intermediate:
            epipolar_path = input_path / "epipolar_check.png"
            self.create_epipolar_visualization(rect_left, rect_right, epipolar_path)
        
        # Compute disparity with enhanced parameters
        disparity = self.compute_disparity(rect_left, rect_right, algorithm, 
                                         block_size, num_disparities, min_disparity)
        
        # Convert disparity to depth
        depth = self.disparity_to_depth(disparity, algorithm)
        
        # Save raw data for precise distance measurements
        raw_data_folder = input_path / "raw_data"
        raw_data_folder.mkdir(exist_ok=True)
        
        # Save raw disparity and depth arrays with full precision
        np.save(str(raw_data_folder / f"disparity_raw_{algorithm}.npy"), disparity.astype(np.float32))
        np.save(str(raw_data_folder / f"depth_raw_{algorithm}.npy"), depth.astype(np.float32))
        
        # Save calibration parameters for distance calculation
        calibration_params = {
            'algorithm': algorithm,
            'baseline': abs(self.T[0]) if self.T is not None else 0.1,
            'fx': self.camera_matrix_left[0, 0] if self.camera_matrix_left is not None else 800,
            'fy': self.camera_matrix_left[1, 1] if self.camera_matrix_left is not None else 800,
            'cx': self.camera_matrix_left[0, 2] if self.camera_matrix_left is not None else disparity.shape[1]/2,
            'cy': self.camera_matrix_left[1, 2] if self.camera_matrix_left is not None else disparity.shape[0]/2,
            'image_width': disparity.shape[1],
            'image_height': disparity.shape[0]
        }
        
        import json
        with open(str(raw_data_folder / f"calibration_params_{algorithm}.json"), 'w') as f:
            json.dump(calibration_params, f, indent=2)
        
        print(f"Raw data saved to: {raw_data_folder}")
        
        # Save enhanced depth visualizations
        depth_paths = self.save_depth_visualizations(depth, input_path, algorithm)
        
        # Also save raw disparity if requested
        if save_intermediate:
            disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(str(input_path / f"disparity_map_{algorithm}.png"), disp_normalized)
            print(f"Saved disparity map ({algorithm})")
        
        return depth_paths['colormap']  # Return the colormap version as primary


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth maps from stereo image pairs with enhanced parameter control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_depth_map.py --input_folder test/000040
  python generate_depth_map.py --input_folder test/000040 --algorithm BM --block_size 25
  python generate_depth_map.py --input_folder test/000040 --num_disparities 96 --min_disparity 16
  python generate_depth_map.py --input_folder test/000040 --save_intermediate --check_calibration
        """
    )
    
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='Path to folder containing left.png and right.png'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['BM', 'SGBM', 'HITNET'],
        default='HITNET',
        help='Stereo matching algorithm to use (default: HITNET). HITNET provides dense disparity maps.'
    )
    
    parser.add_argument(
        '--block_size',
        type=int,
        help='Block size for stereo matching (must be odd, larger = smoother but less detail)'
    )
    
    parser.add_argument(
        '--num_disparities',
        type=int,
        help='Number of disparities to search (must be multiple of 16, larger = wider depth range)'
    )
    
    parser.add_argument(
        '--min_disparity',
        type=int,
        help='Minimum disparity (adjust for near/far object focus)'
    )
    
    parser.add_argument(
        '--save_intermediate',
        action='store_true',
        help='Save intermediate results (rectified images, disparity map, epipolar check)'
    )
    
    parser.add_argument(
        '--check_calibration',
        action='store_true',
        help='Generate epipolar line visualization to check calibration quality'
    )
    
    args = parser.parse_args()
    
    # Validate block_size (must be odd)
    if args.block_size is not None and args.block_size % 2 == 0:
        print("Error: block_size must be odd")
        return 1
    
    # Validate num_disparities (must be multiple of 16)
    if args.num_disparities is not None and args.num_disparities % 16 != 0:
        print("Error: num_disparities must be a multiple of 16")
        return 1
    
    try:
        stereo_system = StereoVisionSystem()
        depth_path = stereo_system.process_stereo_pair(
            args.input_folder,
            algorithm=args.algorithm,
            save_intermediate=args.save_intermediate,
            block_size=args.block_size,
            num_disparities=args.num_disparities,
            min_disparity=args.min_disparity,
            check_calibration=args.check_calibration
        )
        
        print(f"\nProcessing completed successfully!")
        print(f"Depth map saved to: {depth_path}")
        
        # Provide optimization suggestions
        print("\n🔧 Parameter Tuning Suggestions:")
        print("- Increase block_size to reduce noise (but lose detail)")
        print("- Increase num_disparities for wider depth range")
        print("- Adjust min_disparity to focus on near/far objects")
        print("- Use --check_calibration to verify epipolar alignment")
        print("- Try both BM and SGBM algorithms for comparison")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 