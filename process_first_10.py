#!/usr/bin/env python3
"""
Process First 10 Stereo Pairs with SGBM vs HITNET Comparison

This script processes the first 10 test folders using both SGBM and HITNET algorithms
for direct comparison of disparity quality and density.
"""

import os
import subprocess
import sys
from pathlib import Path
import time


def get_first_n_test_folders(test_dir, n=10):
    """Get the first n test folders that contain stereo pairs."""
    test_path = Path(test_dir)
    valid_folders = []
    
    if not test_path.exists():
        print(f"Error: Test directory '{test_dir}' does not exist")
        return []
    
    # Get all directories and sort them
    all_folders = [f for f in test_path.iterdir() if f.is_dir()]
    all_folders.sort()
    
    for folder in all_folders:
        if len(valid_folders) >= n:
            break
            
        left_img = folder / "left.png"
        right_img = folder / "right.png"
        
        if left_img.exists() and right_img.exists():
            valid_folders.append(str(folder))
        else:
            print(f"Skipping {folder.name}: missing stereo pair")
    
    return valid_folders


def process_folder_with_algorithm(folder_path, algorithm):
    """Process a folder using specified algorithm."""
    if algorithm == "SGBM":
        cmd = [
            sys.executable, "generate_depth_map.py",
            "--input_folder", folder_path,
            "--algorithm", "SGBM",
            "--block_size", "15",
            "--num_disparities", "96", 
            "--min_disparity", "0",
            "--save_intermediate"
        ]
    elif algorithm == "HITNET":
        cmd = [
            sys.executable, "generate_depth_map.py",
            "--input_folder", folder_path,
            "--algorithm", "HITNET",
            "--save_intermediate"
        ]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        processing_time = time.time() - start_time
        
        # Parse output for statistics
        output = result.stdout
        valid_pixels_pct = 0
        
        for line in output.split('\n'):
            if "Valid pixels:" in line:
                parts = line.split('(')
                if len(parts) > 1:
                    pct_str = parts[1].split('%')[0]
                    try:
                        valid_pixels_pct = float(pct_str)
                    except:
                        pass
                break
        
        return {
            'success': True,
            'processing_time': processing_time,
            'valid_pixels_pct': valid_pixels_pct,
            'output': output
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'error': str(e),
            'output': e.stdout if e.stdout else e.stderr
        }


def main():
    print("🔧 SGBM vs HITNET Comparison on First 10 Test Folders")
    print("=" * 70)
    print("Algorithms: SGBM (optimized) vs HITNET (dense)")
    print()
    
    # Get first 10 test folders
    test_folders = get_first_n_test_folders("test", 10)
    
    if not test_folders:
        print("❌ No valid test folders found!")
        return 1
    
    print(f"Found {len(test_folders)} test folders to process:")
    for i, folder in enumerate(test_folders, 1):
        folder_name = Path(folder).name
        print(f"  {i:2d}. {folder_name}")
    print()
    
    # Process each folder with both algorithms
    algorithms = ["SGBM", "HITNET"]
    results = {alg: [] for alg in algorithms}
    
    for algorithm in algorithms:
        print(f"\n🚀 Processing with {algorithm} algorithm...")
        print("-" * 50)
        
        total_time = 0
        
        for i, folder in enumerate(test_folders, 1):
            folder_name = Path(folder).name
            print(f"[{i:2d}/{len(test_folders)}] {algorithm}: {folder_name}...", end=" ")
            
            result = process_folder_with_algorithm(folder, algorithm)
            
            if result['success']:
                total_time += result['processing_time']
                print(f"✓ {result['valid_pixels_pct']:.1f}% valid pixels ({result['processing_time']:.1f}s)")
            else:
                print(f"✗ Failed")
                print(f"    Error: {result['error']}")
            
            results[algorithm].append({
                'folder': folder_name,
                **result
            })
        
        # Algorithm summary
        successful = [r for r in results[algorithm] if r['success']]
        if successful:
            avg_valid_pixels = sum(r['valid_pixels_pct'] for r in successful) / len(successful)
            avg_time = total_time / len(successful)
            print(f"  {algorithm} Summary: {avg_valid_pixels:.1f}% avg valid pixels, {avg_time:.1f}s avg time")
    
    # Final comparison
    print("\n" + "=" * 70)
    print("📊 ALGORITHM COMPARISON")
    print("=" * 70)
    
    for algorithm in algorithms:
        successful = [r for r in results[algorithm] if r['success']]
        failed = [r for r in results[algorithm] if not r['success']]
        
        print(f"\n{algorithm} Results:")
        print(f"  ✅ Successful: {len(successful)}/{len(test_folders)}")
        print(f"  ❌ Failed: {len(failed)}/{len(test_folders)}")
        
        if successful:
            avg_valid_pixels = sum(r['valid_pixels_pct'] for r in successful) / len(successful)
            min_valid = min(r['valid_pixels_pct'] for r in successful)
            max_valid = max(r['valid_pixels_pct'] for r in successful)
            total_time = sum(r['processing_time'] for r in successful)
            avg_time = total_time / len(successful)
            
            print(f"  📊 Valid pixels: {avg_valid_pixels:.1f}% avg ({min_valid:.1f}%-{max_valid:.1f}% range)")
            print(f"  ⏱️  Processing: {avg_time:.1f}s avg, {total_time:.1f}s total")
    
    # File comparison
    print(f"\n📁 Generated Files Comparison:")
    for result_sgbm, result_hitnet in zip(results["SGBM"], results["HITNET"]):
        if result_sgbm['success'] and result_hitnet['success']:
            folder_path = f"test/{result_sgbm['folder']}"
            print(f"  📂 {result_sgbm['folder']}:")
            
            # Check file sizes for density comparison
            sgbm_disp = f"{folder_path}/disparity_map_SGBM.png"
            hitnet_disp = f"{folder_path}/disparity_map_HITNET.png"
            
            if os.path.exists(sgbm_disp) and os.path.exists(hitnet_disp):
                sgbm_size = os.path.getsize(sgbm_disp) // 1024
                hitnet_size = os.path.getsize(hitnet_disp) // 1024
                density_improvement = ((hitnet_size - sgbm_size) / sgbm_size) * 100
                print(f"    SGBM disparity: {sgbm_size}KB")
                print(f"    HITNET disparity: {hitnet_size}KB ({density_improvement:+.1f}% size change)")
    
    print(f"\n🎯 Next Steps:")
    print("  • Compare disparity map quality visually")
    print("  • HITNET should show denser, smoother disparity maps")
    print("  • Both algorithms now have colormap depth visualizations")
    print("  • Ready for integration of actual HITNet model when available")
    
    return 0


if __name__ == "__main__":
    exit(main()) 