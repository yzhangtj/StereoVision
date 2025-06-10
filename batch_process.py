#!/usr/bin/env python3
"""
Batch Processing Script for Stereo Vision System

This script processes multiple test folders containing stereo image pairs
and generates depth maps for each folder.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path


def find_test_folders(test_dir):
    """Find all valid test folders containing left.png and right.png."""
    test_path = Path(test_dir)
    valid_folders = []
    
    if not test_path.exists():
        print(f"Error: Test directory '{test_dir}' does not exist")
        return valid_folders
    
    for folder in sorted(test_path.iterdir()):
        if folder.is_dir():
            left_image = folder / "left.png"
            right_image = folder / "right.png"
            
            if left_image.exists() and right_image.exists():
                valid_folders.append(str(folder))
            else:
                print(f"Skipping {folder.name}: missing left.png or right.png")
    
    return valid_folders


def process_folder(folder_path, algorithm, save_intermediate, verbose):
    """Process a single test folder."""
    cmd = [
        sys.executable, "generate_depth_map.py",
        "--input_folder", folder_path,
        "--algorithm", algorithm
    ]
    
    if save_intermediate:
        cmd.append("--save_intermediate")
    
    try:
        if verbose:
            print(f"Processing {folder_path}...")
            result = subprocess.run(cmd, check=True, capture_output=False)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
        folder_name = Path(folder_path).name
        print(f"✓ Completed: {folder_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        folder_name = Path(folder_path).name
        print(f"✗ Failed: {folder_name}")
        if verbose:
            print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process stereo image pairs to generate depth maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process.py                                    # Process all test folders
  python batch_process.py --test_dir test --algorithm BM     # Use Block Matching
  python batch_process.py --folders test/000040 test/000080  # Process specific folders
  python batch_process.py --save_intermediate --verbose      # Save intermediate files with verbose output
        """
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        default='test',
        help='Directory containing test folders (default: test)'
    )
    
    parser.add_argument(
        '--folders',
        nargs='+',
        help='Specific folders to process (overrides --test_dir)'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['BM', 'SGBM'],
        default='SGBM',
        help='Stereo matching algorithm to use (default: SGBM)'
    )
    
    parser.add_argument(
        '--save_intermediate',
        action='store_true',
        help='Save intermediate results (rectified images, disparity map)'
    )
    
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip folders that already have depth_map.png'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output for each folder'
    )
    
    args = parser.parse_args()
    
    # Determine folders to process
    if args.folders:
        # Process specific folders
        folders_to_process = []
        for folder in args.folders:
            folder_path = Path(folder)
            if folder_path.exists() and folder_path.is_dir():
                left_img = folder_path / "left.png"
                right_img = folder_path / "right.png"
                if left_img.exists() and right_img.exists():
                    folders_to_process.append(str(folder_path))
                else:
                    print(f"Warning: {folder} missing left.png or right.png")
            else:
                print(f"Warning: {folder} does not exist or is not a directory")
    else:
        # Find all test folders
        folders_to_process = find_test_folders(args.test_dir)
    
    if not folders_to_process:
        print("No valid test folders found!")
        return 1
    
    # Filter out folders with existing depth maps if requested
    if args.skip_existing:
        original_count = len(folders_to_process)
        folders_to_process = [
            folder for folder in folders_to_process 
            if not (Path(folder) / "depth_map.png").exists()
        ]
        skipped = original_count - len(folders_to_process)
        if skipped > 0:
            print(f"Skipping {skipped} folders with existing depth maps")
    
    if not folders_to_process:
        print("All folders already processed!")
        return 0
    
    print(f"Found {len(folders_to_process)} folders to process")
    print(f"Algorithm: {args.algorithm}")
    print(f"Save intermediate: {args.save_intermediate}")
    print()
    
    # Process each folder
    successful = 0
    failed = 0
    
    for i, folder in enumerate(folders_to_process, 1):
        folder_name = Path(folder).name
        print(f"[{i}/{len(folders_to_process)}] {folder_name}", end=" ... ")
        
        if process_folder(folder, args.algorithm, args.save_intermediate, args.verbose):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\nBatch processing completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(folders_to_process)}")
    
    if failed > 0:
        print(f"\nRerun with --verbose to see detailed error messages")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 