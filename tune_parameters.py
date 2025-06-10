#!/usr/bin/env python3
"""
Parameter Tuning Script for Stereo Vision System

This script automatically tests different parameter combinations to help you
find the optimal settings for your stereo image pairs.
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np


def run_stereo_processing(folder, algorithm, block_size, num_disparities, min_disparity):
    """Run stereo processing with given parameters and return statistics."""
    cmd = [
        sys.executable, "generate_depth_map.py",
        "--input_folder", folder,
        "--algorithm", algorithm,
        "--block_size", str(block_size),
        "--num_disparities", str(num_disparities),
        "--min_disparity", str(min_disparity)
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        processing_time = time.time() - start_time
        
        # Parse output for statistics
        output = result.stdout
        valid_pixels_pct = 0
        disparity_range = "Unknown"
        depth_range = "Unknown"
        
        for line in output.split('\n'):
            if "Valid pixels:" in line:
                # Extract percentage
                parts = line.split('(')
                if len(parts) > 1:
                    pct_str = parts[1].split('%')[0]
                    try:
                        valid_pixels_pct = float(pct_str)
                    except:
                        pass
            elif "Disparity range:" in line:
                disparity_range = line.split("Disparity range: ")[1].strip()
            elif "Depth range:" in line:
                depth_range = line.split("Depth range: ")[1].strip()
        
        return {
            'success': True,
            'processing_time': processing_time,
            'valid_pixels_pct': valid_pixels_pct,
            'disparity_range': disparity_range,
            'depth_range': depth_range,
            'output': output
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'error': str(e),
            'output': e.stdout if e.stdout else e.stderr
        }


def analyze_depth_map(depth_map_path):
    """Analyze the quality of a generated depth map."""
    if not os.path.exists(depth_map_path):
        return None
    
    depth_img = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        return None
    
    # Calculate quality metrics
    non_zero_pixels = np.count_nonzero(depth_img)
    total_pixels = depth_img.size
    coverage = (non_zero_pixels / total_pixels) * 100
    
    # Calculate contrast/variance as a measure of detail
    if non_zero_pixels > 0:
        valid_pixels = depth_img[depth_img > 0]
        contrast = np.std(valid_pixels)
        mean_intensity = np.mean(valid_pixels)
    else:
        contrast = 0
        mean_intensity = 0
    
    return {
        'coverage': coverage,
        'contrast': contrast,
        'mean_intensity': mean_intensity,
        'non_zero_pixels': non_zero_pixels,
        'total_pixels': total_pixels
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune stereo vision parameters")
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Test folder to use for parameter tuning')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with fewer parameter combinations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Folder {args.input_folder} does not exist")
        return 1
    
    print(f"🔧 Parameter Tuning for: {args.input_folder}")
    print("=" * 60)
    
    # Define parameter ranges to test
    if args.quick:
        algorithms = ['SGBM']
        block_sizes = [15, 21]
        num_disparities_list = [96, 128]
        min_disparities = [-16, 0]
    else:
        algorithms = ['BM', 'SGBM']
        block_sizes = [11, 15, 19, 25]
        num_disparities_list = [64, 96, 128, 160]
        min_disparities = [-32, -16, 0, 16]
    
    results = []
    total_combinations = len(algorithms) * len(block_sizes) * len(num_disparities_list) * len(min_disparities)
    current = 0
    
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    for algorithm in algorithms:
        for block_size in block_sizes:
            for num_disparities in num_disparities_list:
                for min_disparity in min_disparities:
                    current += 1
                    
                    print(f"[{current}/{total_combinations}] Testing: {algorithm}, block={block_size}, disp={num_disparities}, min={min_disparity}")
                    
                    # Run processing
                    result = run_stereo_processing(
                        args.input_folder, algorithm, block_size, 
                        num_disparities, min_disparity
                    )
                    
                    if result['success']:
                        # Analyze generated depth map
                        depth_map_path = os.path.join(args.input_folder, "depth_map.png")
                        quality = analyze_depth_map(depth_map_path)
                        
                        result.update({
                            'algorithm': algorithm,
                            'block_size': block_size,
                            'num_disparities': num_disparities,
                            'min_disparity': min_disparity,
                            'quality': quality
                        })
                        
                        print(f"  ✓ Valid pixels: {result['valid_pixels_pct']:.1f}%, Time: {result['processing_time']:.1f}s")
                        if quality:
                            print(f"    Coverage: {quality['coverage']:.1f}%, Contrast: {quality['contrast']:.1f}")
                    else:
                        print(f"  ✗ Failed: {result['error']}")
                        result.update({
                            'algorithm': algorithm,
                            'block_size': block_size,
                            'num_disparities': num_disparities,
                            'min_disparity': min_disparity
                        })
                    
                    results.append(result)
                    print()
    
    # Analyze and rank results
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful parameter combinations found!")
        return 1
    
    # Sort by valid pixels percentage (descending)
    successful_results.sort(key=lambda x: x['valid_pixels_pct'], reverse=True)
    
    print(f"\n🏆 TOP 5 PARAMETER COMBINATIONS (by valid pixels):")
    print("-" * 60)
    
    for i, result in enumerate(successful_results[:5]):
        print(f"{i+1}. {result['algorithm']} | block={result['block_size']} | disp={result['num_disparities']} | min={result['min_disparity']}")
        print(f"   Valid pixels: {result['valid_pixels_pct']:.1f}% | Time: {result['processing_time']:.1f}s")
        if result['quality']:
            print(f"   Coverage: {result['quality']['coverage']:.1f}% | Contrast: {result['quality']['contrast']:.1f}")
        print(f"   Disparity range: {result['disparity_range']}")
        print()
    
    # Performance comparison
    sgbm_results = [r for r in successful_results if r['algorithm'] == 'SGBM']
    bm_results = [r for r in successful_results if r['algorithm'] == 'BM']
    
    if sgbm_results and bm_results:
        avg_sgbm_pixels = np.mean([r['valid_pixels_pct'] for r in sgbm_results])
        avg_bm_pixels = np.mean([r['valid_pixels_pct'] for r in bm_results])
        avg_sgbm_time = np.mean([r['processing_time'] for r in sgbm_results])
        avg_bm_time = np.mean([r['processing_time'] for r in bm_results])
        
        print("⚖️ ALGORITHM COMPARISON:")
        print("-" * 30)
        print(f"SGBM: {avg_sgbm_pixels:.1f}% valid pixels, {avg_sgbm_time:.1f}s avg time")
        print(f"BM:   {avg_bm_pixels:.1f}% valid pixels, {avg_bm_time:.1f}s avg time")
        print()
    
    # Recommendations
    best_result = successful_results[0]
    print("💡 RECOMMENDATIONS:")
    print("-" * 20)
    print(f"Best overall: python generate_depth_map.py --input_folder {args.input_folder} \\")
    print(f"              --algorithm {best_result['algorithm']} --block_size {best_result['block_size']} \\")
    print(f"              --num_disparities {best_result['num_disparities']} --min_disparity {best_result['min_disparity']}")
    print()
    
    # Find fastest with good quality
    good_quality_results = [r for r in successful_results if r['valid_pixels_pct'] > 50]
    if good_quality_results:
        fastest_good = min(good_quality_results, key=lambda x: x['processing_time'])
        if fastest_good != best_result:
            print(f"Fastest with good quality: python generate_depth_map.py --input_folder {args.input_folder} \\")
            print(f"                          --algorithm {fastest_good['algorithm']} --block_size {fastest_good['block_size']} \\")
            print(f"                          --num_disparities {fastest_good['num_disparities']} --min_disparity {fastest_good['min_disparity']}")
            print(f"                          ({fastest_good['valid_pixels_pct']:.1f}% valid, {fastest_good['processing_time']:.1f}s)")
    
    print(f"\n✅ Parameter tuning completed! Tested {len(successful_results)}/{total_combinations} combinations successfully.")
    
    return 0


if __name__ == "__main__":
    exit(main()) 