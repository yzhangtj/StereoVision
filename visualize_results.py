#!/usr/bin/env python3
"""
Results Visualization Script for Stereo Vision System

This script displays stereo image pairs alongside their generated depth maps
for visual quality assessment.
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path


def load_images(folder_path):
    """Load stereo pair and depth map from a folder."""
    folder = Path(folder_path)
    
    # Load original images
    left_path = folder / "left.png"
    right_path = folder / "right.png"
    depth_path = folder / "depth_map.png"
    
    if not left_path.exists() or not right_path.exists():
        print(f"Error: Missing left.png or right.png in {folder_path}")
        return None, None, None
    
    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))
    
    # Load depth map if it exists
    depth_img = None
    if depth_path.exists():
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        # Convert to color for better visualization
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    else:
        print(f"Warning: No depth_map.png found in {folder_path}")
    
    return left_img, right_img, depth_img


def resize_to_fit(img, max_width, max_height):
    """Resize image to fit within max dimensions while maintaining aspect ratio."""
    if img is None:
        return img
    
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img


def create_visualization(left_img, right_img, depth_img, folder_name):
    """Create a combined visualization of stereo pair and depth map."""
    if left_img is None or right_img is None:
        return None
    
    # Resize images to fit display
    max_width = 400
    max_height = 300
    
    left_resized = resize_to_fit(left_img, max_width, max_height)
    right_resized = resize_to_fit(right_img, max_width, max_height)
    
    # Create top row (stereo pair)
    stereo_row = np.hstack((left_resized, right_resized))
    
    if depth_img is not None:
        depth_resized = resize_to_fit(depth_img, max_width, max_height)
        
        # Create placeholder for missing fourth quadrant
        h, w = depth_resized.shape[:2]
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add text to placeholder
        text = "Depth Map"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        cv2.putText(placeholder, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Create bottom row (depth map + placeholder)
        depth_row = np.hstack((depth_resized, placeholder))
        
        # Ensure both rows have the same width
        row_width = max(stereo_row.shape[1], depth_row.shape[1])
        
        if stereo_row.shape[1] < row_width:
            padding = np.zeros((stereo_row.shape[0], row_width - stereo_row.shape[1], 3), dtype=np.uint8)
            stereo_row = np.hstack((stereo_row, padding))
        
        if depth_row.shape[1] < row_width:
            padding = np.zeros((depth_row.shape[0], row_width - depth_row.shape[1], 3), dtype=np.uint8)
            depth_row = np.hstack((depth_row, padding))
        
        # Combine rows
        combined = np.vstack((stereo_row, depth_row))
    else:
        combined = stereo_row
    
    # Add title
    title_height = 50
    title_img = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)
    thickness = 2
    
    text_size = cv2.getTextSize(folder_name, font, font_scale, thickness)[0]
    text_x = (combined.shape[1] - text_size[0]) // 2
    text_y = (title_height + text_size[1]) // 2
    
    cv2.putText(title_img, folder_name, (text_x, text_y), font, font_scale, color, thickness)
    
    # Add labels
    label_height = 25
    label_img = np.zeros((label_height, combined.shape[1], 3), dtype=np.uint8)
    
    font_scale = 0.5
    thickness = 1
    
    # Left camera label
    cv2.putText(label_img, "Left Camera", (10, 20), font, font_scale, color, thickness)
    
    # Right camera label
    right_x = left_resized.shape[1] + 10
    cv2.putText(label_img, "Right Camera", (right_x, 20), font, font_scale, color, thickness)
    
    # Combine title, labels, and main image
    final_img = np.vstack((title_img, label_img, combined))
    
    return final_img


def main():
    parser = argparse.ArgumentParser(
        description="Visualize stereo vision results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_results.py test/000040                    # View single folder
  python visualize_results.py test/000040 test/000080        # View multiple folders
  python visualize_results.py --test_dir test --limit 5      # View first 5 folders from test directory
        """
    )
    
    parser.add_argument(
        'folders',
        nargs='*',
        help='Specific folders to visualize'
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        default='test',
        help='Directory containing test folders (default: test)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of folders to display'
    )
    
    parser.add_argument(
        '--depth_only',
        action='store_true',
        help='Only show folders that have depth maps'
    )
    
    args = parser.parse_args()
    
    # Determine folders to visualize
    if args.folders:
        folders_to_show = args.folders
    else:
        # Find folders in test directory
        test_path = Path(args.test_dir)
        folders_to_show = []
        
        if test_path.exists():
            for folder in sorted(test_path.iterdir()):
                if folder.is_dir():
                    left_img = folder / "left.png"
                    right_img = folder / "right.png"
                    depth_img = folder / "depth_map.png"
                    
                    if left_img.exists() and right_img.exists():
                        if not args.depth_only or depth_img.exists():
                            folders_to_show.append(str(folder))
        else:
            print(f"Error: Test directory '{args.test_dir}' does not exist")
            return 1
    
    if not folders_to_show:
        print("No folders to visualize!")
        return 1
    
    # Apply limit
    if args.limit:
        folders_to_show = folders_to_show[:args.limit]
    
    print(f"Visualizing {len(folders_to_show)} folders...")
    print("Press any key to move to the next folder, 'q' to quit")
    
    for i, folder_path in enumerate(folders_to_show):
        folder_name = Path(folder_path).name
        print(f"[{i+1}/{len(folders_to_show)}] Showing {folder_name}")
        
        # Load images
        left_img, right_img, depth_img = load_images(folder_path)
        
        if left_img is None or right_img is None:
            print(f"Skipping {folder_name}: Could not load images")
            continue
        
        # Create visualization
        vis_img = create_visualization(left_img, right_img, depth_img, folder_name)
        
        if vis_img is None:
            print(f"Skipping {folder_name}: Could not create visualization")
            continue
        
        # Display
        window_name = f"Stereo Vision Results - {folder_name}"
        cv2.imshow(window_name, vis_img)
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            print("Visualization stopped by user")
            break
    
    print("Visualization complete!")
    return 0


if __name__ == "__main__":
    exit(main()) 