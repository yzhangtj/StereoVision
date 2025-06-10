#!/usr/bin/env python3
"""
3D Object Detection with YOLOv8 and Stereo Vision

This script combines YOLOv8 object detection with stereo vision depth data to:
1. Detect objects in the left rectified RGB frame using YOLOv8
2. Compute 3D positions for each detected object
3. Save results for path planning applications

Uses the left rectified frame for detection and corresponding depth data
to convert 2D bounding boxes to 3D world coordinates.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime


class ObjectDetection3D:
    def __init__(self, folder_path, algorithm='HITNET', model_size='s'):
        self.folder_path = Path(folder_path)
        self.algorithm = algorithm
        
        # Initialize YOLO model
        self.init_yolo_model(model_size)
        
        # Load depth data and calibration
        self.load_stereo_data()
        
        # Load rectified left image for detection
        self.load_rectified_image()
    
    def init_yolo_model(self, model_size='s'):
        """Initialize YOLOv8 model."""
        model_names = {
            'n': 'yolov8n.pt',    # Nano - fastest
            's': 'yolov8s.pt',    # Small - better accuracy
            'm': 'yolov8m.pt',    # Medium
            'l': 'yolov8l.pt',    # Large
            'x': 'yolov8x.pt'     # Extra Large - most accurate
        }
        
        model_name = model_names.get(model_size, 'yolov8s.pt')
        print(f"Loading YOLOv8 model: {model_name}")
        
        try:
            self.yolo_model = YOLO(model_name)
            print(f"✅ YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise
    
    def load_stereo_data(self):
        """Load depth data and calibration parameters."""
        raw_folder = self.folder_path / "raw_data"
        
        # Load raw depth array
        depth_path = raw_folder / f"depth_raw_{self.algorithm}.npy"
        disparity_path = raw_folder / f"disparity_raw_{self.algorithm}.npy"
        
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth data not found: {depth_path}")
        
        self.depth_array = np.load(str(depth_path))
        self.disparity_array = np.load(str(disparity_path))
        
        print(f"Loaded depth data:")
        print(f"  Shape: {self.depth_array.shape}")
        print(f"  Depth range: {self.depth_array.min():.2f} - {self.depth_array.max():.2f}")
        
        # Load calibration parameters
        calib_path = raw_folder / f"calibration_params_{self.algorithm}.json"
        
        if calib_path.exists():
            with open(str(calib_path), 'r') as f:
                self.calib_params = json.load(f)
            
            self.baseline = self.calib_params['baseline']
            self.fx = self.calib_params['fx']
            self.fy = self.calib_params['fy']
            self.cx = self.calib_params['cx']
            self.cy = self.calib_params['cy']
        else:
            print("Warning: Using default calibration parameters")
            self.baseline = 0.1
            self.fx = 800
            self.fy = 800
            self.cx = self.depth_array.shape[1] / 2
            self.cy = self.depth_array.shape[0] / 2
        
        print(f"Camera parameters:")
        print(f"  Focal length: fx={self.fx:.1f}, fy={self.fy:.1f}")
        print(f"  Principal point: ({self.cx:.1f}, {self.cy:.1f})")
        print(f"  Baseline: {self.baseline:.3f}m")
    
    def load_rectified_image(self):
        """Load the left rectified image for object detection."""
        # Try to load processed rectified image first
        rect_left_path = self.folder_path / f"rect_left_{self.algorithm}.png"
        
        if rect_left_path.exists():
            self.left_image = cv2.imread(str(rect_left_path))
            print(f"Loaded rectified left image: {rect_left_path}")
        else:
            # Fallback to original left image
            left_path = self.folder_path / "left.png"
            if left_path.exists():
                self.left_image = cv2.imread(str(left_path))
                print(f"Warning: Using original left image (not rectified): {left_path}")
            else:
                raise FileNotFoundError(f"No left image found in {self.folder_path}")
        
        print(f"Image shape: {self.left_image.shape}")
        
        # Ensure image and depth have compatible dimensions
        img_h, img_w = self.left_image.shape[:2]
        depth_h, depth_w = self.depth_array.shape
        
        if img_h != depth_h or img_w != depth_w:
            print(f"Warning: Image ({img_w}x{img_h}) and depth ({depth_w}x{depth_h}) dimensions mismatch")
            print("Resizing depth array to match image...")
            self.depth_array = cv2.resize(self.depth_array, (img_w, img_h))
            self.disparity_array = cv2.resize(self.disparity_array, (img_w, img_h))
    
    def pixel_to_3d(self, u, v, depth):
        """Convert pixel coordinates (u,v) and depth to 3D world coordinates (X,Y,Z)."""
        if depth <= 0 or not np.isfinite(depth):
            return None, None, None
        
        # Convert to 3D coordinates using camera intrinsics
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        
        return X, Y, Z
    
    def calculate_3d_bounding_box(self, bbox_2d, center_depth):
        """Calculate 3D bounding box dimensions and corners from 2D bbox and depth."""
        x1, y1, x2, y2 = bbox_2d
        
        # Get depth at multiple points for better estimation
        depths = []
        sample_points = [
            (x1, y1), (x2, y1), (x1, y2), (x2, y2),  # corners
            ((x1+x2)//2, y1), ((x1+x2)//2, y2),      # top/bottom center
            (x1, (y1+y2)//2), (x2, (y1+y2)//2)       # left/right center
        ]
        
        for px, py in sample_points:
            if 0 <= px < self.depth_array.shape[1] and 0 <= py < self.depth_array.shape[0]:
                d = self.depth_array[py, px]
                if d > 0:
                    depths.append(d)
        
        if not depths:
            depths = [center_depth]
        
        # Use median depth for robustness
        median_depth = np.median(depths)
        min_depth = np.min(depths)
        max_depth = np.max(depths)
        
        # Calculate 3D coordinates of bounding box corners
        corners_3d = []
        corner_pixels = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # top-left, top-right, bottom-right, bottom-left
        
        for px, py in corner_pixels:
            # Use depth at this corner if available, otherwise use median
            if 0 <= px < self.depth_array.shape[1] and 0 <= py < self.depth_array.shape[0]:
                corner_depth = self.depth_array[py, px]
                if corner_depth <= 0:
                    corner_depth = median_depth
            else:
                corner_depth = median_depth
            
            X, Y, Z = self.pixel_to_3d(px, py, corner_depth)
            if X is not None:
                corners_3d.append([float(X), float(Y), float(Z)])
            else:
                corners_3d.append([None, None, None])
        
        # Calculate object dimensions in meters
        if len([c for c in corners_3d if c[0] is not None]) >= 2:
            valid_corners = [c for c in corners_3d if c[0] is not None]
            
            # Width (X direction)
            x_coords = [c[0] for c in valid_corners]
            width = max(x_coords) - min(x_coords)
            
            # Height (Y direction) 
            y_coords = [c[1] for c in valid_corners]
            height = max(y_coords) - min(y_coords)
            
            # Depth extent (Z direction) - estimate from depth variation
            depth_extent = max_depth - min_depth if len(depths) > 1 else 0.1
            
            # Ensure minimum realistic dimensions
            width = max(width, 0.05)   # minimum 5cm
            height = max(height, 0.05) # minimum 5cm
            depth_extent = max(depth_extent, 0.05)  # minimum 5cm
            
        else:
            # Fallback calculation using pixel size and average depth
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            
            # Convert pixel dimensions to meters using depth
            width = (pixel_width * median_depth) / self.fx
            height = (pixel_height * median_depth) / self.fy
            depth_extent = 0.1  # default 10cm depth
            
            corners_3d = [[None, None, None]] * 4
        
        return {
            'corners_3d': corners_3d,
            'dimensions': {
                'width': float(width),
                'height': float(height), 
                'depth': float(depth_extent)
            },
            'depth_info': {
                'center_depth': float(center_depth),
                'median_depth': float(median_depth),
                'min_depth': float(min_depth),
                'max_depth': float(max_depth),
                'depth_variation': float(max_depth - min_depth)
            }
        }
    
    def get_object_volume(self, dimensions):
        """Calculate object volume in cubic meters."""
        return dimensions['width'] * dimensions['height'] * dimensions['depth']
    
    def get_object_depth(self, bbox, method='center'):
        """Get depth for an object given its bounding box."""
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        h, w = self.depth_array.shape
        x1, x2 = max(0, min(w-1, x1)), max(0, min(w-1, x2))
        y1, y2 = max(0, min(h-1, y1)), max(0, min(h-1, y2))
        
        if method == 'center':
            # Use center point of bounding box
            center_u = int((x1 + x2) / 2)
            center_v = int((y1 + y2) / 2)
            depth = self.depth_array[center_v, center_u]
            return depth, center_u, center_v
        
        elif method == 'median':
            # Use median depth within bounding box
            bbox_depths = self.depth_array[y1:y2+1, x1:x2+1]
            valid_depths = bbox_depths[bbox_depths > 0]
            
            if len(valid_depths) > 0:
                depth = np.median(valid_depths)
                # Return center coordinates for consistency
                center_u = int((x1 + x2) / 2)
                center_v = int((y1 + y2) / 2)
                return depth, center_u, center_v
            else:
                return 0, int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        elif method == 'closest':
            # Use closest (minimum) valid depth within bounding box
            bbox_depths = self.depth_array[y1:y2+1, x1:x2+1]
            valid_depths = bbox_depths[bbox_depths > 0]
            
            if len(valid_depths) > 0:
                depth = np.min(valid_depths)
                # Find coordinates of closest point
                closest_idx = np.unravel_index(np.argmin(bbox_depths + (bbox_depths <= 0) * 1e6), bbox_depths.shape)
                center_u = x1 + closest_idx[1]
                center_v = y1 + closest_idx[0]
                return depth, center_u, center_v
            else:
                return 0, int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def run_detection(self, confidence_threshold=0.15, depth_method='center', save_results=True):
        """Run YOLOv8 detection and compute 3D positions."""
        print(f"\n🔍 Running YOLOv8 object detection...")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Depth method: {depth_method}")
        print(f"Using improved detection parameters: imgsz=960, iou=0.5, max_det=20")
        
        # Run YOLO detection with improved parameters
        results = self.yolo_model(
            self.left_image, 
            conf=confidence_threshold,
            imgsz=960,           # Higher resolution for better detection
            iou=0.5,            # IoU threshold for non-maximum suppression
            max_det=20,         # Maximum detections per image
            verbose=False
        )
        
        # Process detections
        detections_3d = []
        annotated_image = self.left_image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Get depth information
                    depth, center_u, center_v = self.get_object_depth([x1, y1, x2, y2], method=depth_method)
                    
                    # Convert to 3D coordinates
                    X, Y, Z = self.pixel_to_3d(center_u, center_v, depth)
                    
                    # Calculate distance from camera
                    distance = np.sqrt(X**2 + Y**2 + Z**2) if X is not None else None
                    
                    # Calculate 3D bounding box information
                    bbox_3d_info = self.calculate_3d_bounding_box([x1, y1, x2, y2], depth)
                    
                    # Store detection with enhanced 3D information
                    detection = {
                        'id': i,
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': float(confidence),
                        'bbox_2d': [int(x1), int(y1), int(x2), int(y2)],
                        'bbox_2d_size': [int(x2-x1), int(y2-y1)],  # width, height in pixels
                        'center_2d': [center_u, center_v],
                        'depth': float(depth) if np.isfinite(depth) else None,
                        'position_3d': [float(X), float(Y), float(Z)] if X is not None else None,
                        'distance': float(distance) if distance is not None else None,
                        'bbox_3d': {
                            'corners': bbox_3d_info['corners_3d'],
                            'dimensions_m': bbox_3d_info['dimensions'],
                            'depth_analysis': bbox_3d_info['depth_info']
                        }
                    }
                    
                    detections_3d.append(detection)
                    
                    # Draw bounding box
                    color = (0, 255, 0) if depth > 0 else (0, 0, 255)  # Green if valid depth, red if invalid
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label with 3D information including dimensions
                    if X is not None:
                        dims = bbox_3d_info['dimensions']
                        label = f"{class_name} {confidence:.2f}\n({X:.2f}, {Y:.2f}, {Z:.2f})m\nDist: {distance:.2f}m\nSize: {dims['width']:.2f}x{dims['height']:.2f}x{dims['depth']:.2f}m"
                    else:
                        label = f"{class_name} {confidence:.2f}\nNo depth"
                    
                    # Draw label background
                    label_lines = label.split('\n')
                    line_height = 20
                    label_height = len(label_lines) * line_height
                    cv2.rectangle(annotated_image, (x1, y1-label_height-5), (x1+200, y1), color, -1)
                    
                    # Draw label text
                    for j, line in enumerate(label_lines):
                        cv2.putText(annotated_image, line, (x1+5, y1-label_height+j*line_height+15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw center point
                    cv2.circle(annotated_image, (center_u, center_v), 3, (0, 255, 255), -1)
        
        print(f"✅ Detected {len(detections_3d)} objects")
        
        # Save results
        if save_results:
            self.save_detection_results(detections_3d, annotated_image, depth_method)
        
        return detections_3d, annotated_image
    
    def save_detection_results(self, detections, annotated_image, depth_method):
        """Save detection results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save annotated image
        output_image_path = self.folder_path / f"object_detection_3d_{self.algorithm}_{depth_method}.png"
        cv2.imwrite(str(output_image_path), annotated_image)
        print(f"📸 Annotated image saved: {output_image_path}")
        
        # Save detection data as JSON
        output_json_path = self.folder_path / f"detections_3d_{self.algorithm}_{depth_method}.json"
        
        detection_data = {
            'timestamp': timestamp,
            'folder': str(self.folder_path),
            'algorithm': self.algorithm,
            'depth_method': depth_method,
            'camera_parameters': {
                'fx': self.fx, 'fy': self.fy,
                'cx': self.cx, 'cy': self.cy,
                'baseline': self.baseline
            },
            'detections': detections,
            'summary': {
                'total_objects': len(detections),
                'objects_with_depth': len([d for d in detections if d['depth'] is not None]),
                'class_counts': {}
            }
        }
        
        # Count objects by class
        for detection in detections:
            class_name = detection['class_name']
            detection_data['summary']['class_counts'][class_name] = \
                detection_data['summary']['class_counts'].get(class_name, 0) + 1
        
        with open(str(output_json_path), 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        print(f"💾 Detection data saved: {output_json_path}")
        
        # Print summary
        self.print_detection_summary(detections)
    
    def print_detection_summary(self, detections):
        """Print a summary of detections."""
        print(f"\n📊 Detection Summary:")
        print(f"Total objects detected: {len(detections)}")
        
        # Count by class
        class_counts = {}
        valid_depth_count = 0
        
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if detection['depth'] is not None and detection['depth'] > 0:
                valid_depth_count += 1
        
        print(f"Objects with valid depth: {valid_depth_count}/{len(detections)} ({100*valid_depth_count/len(detections):.1f}%)")
        
        print("\nDetected classes:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
        
        print("\nDetailed detections:")
        for i, detection in enumerate(detections):
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            if detection['position_3d'] is not None:
                X, Y, Z = detection['position_3d']
                distance = detection['distance']
                dims = detection['bbox_3d']['dimensions_m']
                print(f"  {i+1}. {class_name} (conf: {confidence:.2f})")
                print(f"      Position: ({X:.2f}, {Y:.2f}, {Z:.2f})m, Distance: {distance:.2f}m")
                print(f"      Size: {dims['width']:.2f}×{dims['height']:.2f}×{dims['depth']:.2f}m")
            else:
                print(f"  {i+1}. {class_name} (conf: {confidence:.2f}) -> No valid depth")


def main():
    parser = argparse.ArgumentParser(
        description="3D Object Detection using YOLOv8 and Stereo Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python object_detection_3d.py --folder test/000040
  python object_detection_3d.py --folder test/000040 --algorithm HITNET --model_size s
  python object_detection_3d.py --folder test/000040 --confidence 0.3 --depth_method median
  python object_detection_3d.py --folder test/010200 --model_size m --depth_method closest
        """
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to folder containing processed stereo data'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['SGBM', 'HITNET'],
        default='HITNET',
        help='Stereo algorithm used for depth data (default: HITNET)'
    )
    
    parser.add_argument(
        '--model_size',
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        default='s',
        help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(tra large) (default: s)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.15,
        help='Confidence threshold for object detection (default: 0.15)'
    )
    
    parser.add_argument(
        '--depth_method',
        type=str,
        choices=['center', 'median', 'closest'],
        default='center',
        help='Method for extracting depth from bounding box (default: center)'
    )
    
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Skip saving results to files'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize 3D object detector
        detector = ObjectDetection3D(
            folder_path=args.folder,
            algorithm=args.algorithm,
            model_size=args.model_size
        )
        
        # Run detection
        detections, annotated_image = detector.run_detection(
            confidence_threshold=args.confidence,
            depth_method=args.depth_method,
            save_results=not args.no_save
        )
        
        # Show annotated image
        cv2.imshow(f"3D Object Detection - {args.folder}", annotated_image)
        print(f"\n👁️  Press any key to close the image window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"\n✅ 3D Object Detection completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 