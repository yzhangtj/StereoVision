#!/usr/bin/env python3
"""
Interactive Depth Probe Tool

This tool allows you to click on depth/disparity images to get real-world distance measurements.
Load raw disparity data and click anywhere to get precise distance from camera.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


class DepthProbe:
    def __init__(self, folder_path, algorithm='SGBM'):
        self.folder_path = Path(folder_path)
        self.algorithm = algorithm
        
        # Mouse tracking variables for real-time display
        self.current_mouse_x = -1
        self.current_mouse_y = -1
        self.show_crosshair = False
        
        # Load raw data
        self.load_raw_data()
        
        # Load calibration parameters
        self.load_calibration()
        
        # Load visualization image
        self.load_visualization()
        
        # Setup mouse callback
        self.setup_mouse_callback()
    
    def load_raw_data(self):
        """Load raw disparity and depth arrays."""
        raw_folder = self.folder_path / "raw_data"
        
        disp_path = raw_folder / f"disparity_raw_{self.algorithm}.npy"
        depth_path = raw_folder / f"depth_raw_{self.algorithm}.npy"
        
        if not disp_path.exists():
            raise FileNotFoundError(f"Raw disparity data not found: {disp_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Raw depth data not found: {depth_path}")
        
        self.disparity = np.load(str(disp_path))
        self.depth = np.load(str(depth_path))
        
        print(f"Loaded raw data:")
        print(f"  Disparity shape: {self.disparity.shape}")
        print(f"  Depth shape: {self.depth.shape}")
        print(f"  Disparity range: {self.disparity.min():.2f} - {self.disparity.max():.2f}")
        print(f"  Depth range: {self.depth.min():.2f} - {self.depth.max():.2f}")
    
    def load_calibration(self):
        """Load calibration parameters."""
        raw_folder = self.folder_path / "raw_data"
        calib_path = raw_folder / f"calibration_params_{self.algorithm}.json"
        
        if not calib_path.exists():
            print(f"Warning: Calibration file not found: {calib_path}")
            # Use default parameters
            self.baseline = 0.1
            self.fx = 800
            self.fy = 800
            self.cx = self.disparity.shape[1] / 2
            self.cy = self.disparity.shape[0] / 2
        else:
            with open(str(calib_path), 'r') as f:
                params = json.load(f)
            
            self.baseline = params['baseline']
            self.fx = params['fx']
            self.fy = params['fy']
            self.cx = params['cx']
            self.cy = params['cy']
        
        print(f"Calibration parameters:")
        print(f"  Baseline: {self.baseline:.3f} m")
        print(f"  Focal length (fx): {self.fx:.1f} pixels")
        print(f"  Principal point: ({self.cx:.1f}, {self.cy:.1f})")
    
    def load_visualization(self):
        """Load depth colormap for visualization."""
        colormap_path = self.folder_path / f"depth_colormap_{self.algorithm}.png"
        
        if colormap_path.exists():
            self.vis_image = cv2.imread(str(colormap_path))
            print(f"Loaded visualization: {colormap_path}")
        else:
            # Create a simple visualization from raw data
            self.create_fallback_visualization()
    
    def create_fallback_visualization(self):
        """Create a basic visualization if colormap doesn't exist."""
        # Normalize depth for visualization
        valid_mask = self.depth > 0
        if np.any(valid_mask):
            depth_norm = np.zeros_like(self.depth)
            depth_norm[valid_mask] = self.depth[valid_mask]
            
            # Use percentile-based normalization
            depth_95th = np.percentile(self.depth[valid_mask], 95)
            depth_5th = np.percentile(self.depth[valid_mask], 5)
            
            depth_clipped = np.clip(depth_norm, depth_5th, depth_95th)
            depth_vis = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.vis_image = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            
            # Set invalid areas to black
            self.vis_image[~valid_mask] = [0, 0, 0]
        else:
            self.vis_image = np.zeros((self.depth.shape[0], self.depth.shape[1], 3), dtype=np.uint8)
    
    def get_depth_meters(self, x, y):
        """Get depth in meters for pixel coordinates (x, y)."""
        if x < 0 or x >= self.disparity.shape[1] or y < 0 or y >= self.disparity.shape[0]:
            return np.inf, "Out of bounds"
        
        # Get disparity value
        d = self.disparity[y, x]
        
        if d <= 0:
            return np.inf, "Invalid/occluded"
        
        # Convert to depth using calibration
        if self.algorithm == 'HITNET':
            # Direct formula for HITNET
            depth_m = (self.baseline * self.fx) / d
        else:
            # For SGBM, depth is already calculated by OpenCV's reprojectImageTo3D
            depth_m = self.depth[y, x]
        
        return depth_m, "Valid"
    
    def get_3d_coordinates(self, x, y):
        """Get 3D coordinates in camera coordinate system."""
        depth_m, status = self.get_depth_meters(x, y)
        
        if not np.isfinite(depth_m):
            return None, None, None, status
        
        # Convert to 3D coordinates
        X = (x - self.cx) * depth_m / self.fx
        Y = (y - self.cy) * depth_m / self.fy
        Z = depth_m
        
        return X, Y, Z, status
    
    def get_display_image(self):
        """Get the current display image with real-time overlay."""
        display_image = self.vis_image.copy()
        
        # Add real-time crosshair and distance if mouse is over image
        if self.show_crosshair and 0 <= self.current_mouse_x < display_image.shape[1] and 0 <= self.current_mouse_y < display_image.shape[0]:
            x, y = self.current_mouse_x, self.current_mouse_y
            
            # Get depth measurement for current position
            depth_m, status = self.get_depth_meters(x, y)
            
            # Draw crosshair
            cv2.line(display_image, (x-10, y), (x+10, y), (255, 255, 255), 1)
            cv2.line(display_image, (x, y-10), (x, y+10), (255, 255, 255), 1)
            
            # Prepare distance text
            if np.isfinite(depth_m):
                distance_text = f"{depth_m:.2f}m"
                color = (0, 255, 255)  # Yellow for valid measurements
            else:
                distance_text = status
                color = (0, 0, 255)  # Red for invalid measurements
            
            # Position text to avoid going off screen
            text_x = x + 15
            text_y = y - 10
            if text_x > display_image.shape[1] - 100:
                text_x = x - 100
            if text_y < 20:
                text_y = y + 25
            
            # Add semi-transparent background for text
            text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_image, 
                        (text_x-5, text_y-text_size[1]-5), 
                        (text_x+text_size[0]+5, text_y+5), 
                        (0, 0, 0), -1)
            
            # Draw distance text
            cv2.putText(display_image, distance_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show coordinates in corner
            coord_text = f"({x}, {y})"
            cv2.putText(display_image, coord_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_image
    
    def on_mouse(self, event, x, y, flags, param):
        """Mouse callback for interactive probing."""
        # Handle mouse movement for real-time distance display
        if event == cv2.EVENT_MOUSEMOVE:
            # Update current mouse position
            self.current_mouse_x = x
            self.current_mouse_y = y
            self.show_crosshair = True
        
        # Handle clicks for permanent markers
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get depth measurement
            depth_m, status = self.get_depth_meters(x, y)
            
            # Get 3D coordinates
            X, Y, Z, _ = self.get_3d_coordinates(x, y)
            
            print(f"\n📍 Clicked at pixel ({x}, {y}):")
            print(f"  Disparity: {self.disparity[y, x]:.2f}")
            
            if np.isfinite(depth_m):
                print(f"  Distance: {depth_m:.3f} m ({depth_m*100:.1f} cm)")
                if X is not None:
                    print(f"  3D position: ({X:.3f}, {Y:.3f}, {Z:.3f}) m")
                    # Calculate total distance from camera center
                    total_distance = np.sqrt(X**2 + Y**2 + Z**2)
                    print(f"  Total distance from camera: {total_distance:.3f} m ({total_distance*100:.1f} cm)")
            else:
                print(f"  Status: {status}")
            
            # Draw a permanent circle on the base image
            cv2.circle(self.vis_image, (x, y), 5, (0, 255, 0), 2)
            cv2.putText(self.vis_image, f"{depth_m:.2f}m", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def setup_mouse_callback(self):
        """Setup the mouse callback for the display window."""
        self.window_name = f"Depth Probe - {self.algorithm} - Real-time Distance"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
    
    def run(self):
        """Run the interactive depth probe."""
        print(f"\n🔍 Interactive Depth Probe Active")
        print(f"Algorithm: {self.algorithm}")
        print(f"Folder: {self.folder_path}")
        print("\nInstructions:")
        print("  • Move mouse over image to see real-time distance measurements")
        print("  • Click anywhere to place permanent markers with distance labels")
        print("  • Press 'r' to reset annotations")
        print("  • Press 'q' or ESC to quit")
        print("  • Press 's' to save annotated image")
        
        # Make a copy for reset functionality
        self.original_image = self.vis_image.copy()
        
        while True:
            # Display image with real-time overlay
            display_image = self.get_display_image()
            cv2.imshow(self.window_name, display_image)
            key = cv2.waitKey(30) & 0xFF  # Increased wait time for smoother display
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('r'):  # Reset
                self.vis_image = self.original_image.copy()
                print("🔄 Annotations reset")
            elif key == ord('s'):  # Save
                save_path = self.folder_path / f"depth_probe_annotated_{self.algorithm}.png"
                cv2.imwrite(str(save_path), self.vis_image)
                print(f"💾 Annotated image saved to: {save_path}")
        
        cv2.destroyAllWindows()
    
    def probe_specific_points(self, points):
        """Probe specific points and return measurements."""
        results = []
        
        print(f"\n📊 Probing {len(points)} specific points:")
        for i, (x, y) in enumerate(points):
            depth_m, status = self.get_depth_meters(x, y)
            X, Y, Z, _ = self.get_3d_coordinates(x, y)
            
            result = {
                'point': i+1,
                'pixel': (x, y),
                'disparity': self.disparity[y, x],
                'depth_m': depth_m,
                'status': status,
                '3d_coords': (X, Y, Z) if X is not None else None
            }
            
            results.append(result)
            
            print(f"  Point {i+1} ({x}, {y}): ", end="")
            if np.isfinite(depth_m):
                print(f"{depth_m:.3f} m ({depth_m*100:.1f} cm)")
                if X is not None:
                    total_dist = np.sqrt(X**2 + Y**2 + Z**2)
                    print(f"    3D: ({X:.3f}, {Y:.3f}, {Z:.3f}) m, Total: {total_dist:.3f} m")
            else:
                print(f"Invalid ({status})")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Interactive depth probe tool for measuring distances in stereo images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python depth_probe.py --folder test/000040 --algorithm SGBM
  python depth_probe.py --folder test/000040 --algorithm HITNET
  python depth_probe.py --folder test/000040 --algorithm HITNET --points 140,45 260,210
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
        help='Algorithm to probe (default: HITNET)'
    )
    
    parser.add_argument(
        '--points',
        type=str,
        nargs='*',
        help='Specific points to probe in format "x,y" (e.g., 140,45 260,210)'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip interactive mode, only probe specific points'
    )
    
    args = parser.parse_args()
    
    try:
        probe = DepthProbe(args.folder, args.algorithm)
        
        # Probe specific points if provided
        if args.points:
            points = []
            for point_str in args.points:
                try:
                    x, y = map(int, point_str.split(','))
                    points.append((x, y))
                except ValueError:
                    print(f"Warning: Invalid point format '{point_str}'. Use 'x,y' format.")
            
            if points:
                probe.probe_specific_points(points)
        
        # Run interactive mode
        if not args.no_interactive:
            probe.run()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 