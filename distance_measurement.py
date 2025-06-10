#!/usr/bin/env python3
"""
Batch Distance Measurement Tool

This tool provides batch distance measurements across multiple stereo pairs.
Useful for quantitative analysis and validation of stereo vision results.
"""

import numpy as np
import json
import argparse
from pathlib import Path
import csv


class BatchDistanceMeasurement:
    def __init__(self):
        self.results = []
    
    def load_folder_data(self, folder_path, algorithm):
        """Load raw data from a specific folder."""
        folder_path = Path(folder_path)
        raw_folder = folder_path / "raw_data"
        
        # Load disparity and calibration
        disp_path = raw_folder / f"disparity_raw_{algorithm}.npy"
        calib_path = raw_folder / f"calibration_params_{algorithm}.json"
        
        if not disp_path.exists() or not calib_path.exists():
            return None
        
        disparity = np.load(str(disp_path))
        
        with open(str(calib_path), 'r') as f:
            calib = json.load(f)
        
        return {
            'folder': folder_path.name,
            'disparity': disparity,
            'baseline': calib['baseline'],
            'fx': calib['fx'],
            'fy': calib['fy'],
            'cx': calib['cx'],
            'cy': calib['cy']
        }
    
    def measure_distance(self, data, x, y, algorithm='HITNET'):
        """Measure distance at specific pixel coordinates."""
        if x < 0 or x >= data['disparity'].shape[1] or y < 0 or y >= data['disparity'].shape[0]:
            return None, "Out of bounds"
        
        d = data['disparity'][y, x]
        
        if d <= 0:
            return None, "Invalid/occluded"
        
        # Calculate depth
        if algorithm == 'HITNET':
            depth_m = (data['baseline'] * data['fx']) / d
        else:
            # For SGBM, load depth array instead
            folder_path = Path("test") / data['folder']
            depth_path = folder_path / "raw_data" / f"depth_raw_{algorithm}.npy"
            if depth_path.exists():
                depth = np.load(str(depth_path))
                depth_m = depth[y, x]
            else:
                return None, "Depth file not found"
        
        # Calculate 3D coordinates
        X = (x - data['cx']) * depth_m / data['fx']
        Y = (y - data['cy']) * depth_m / data['fy']
        Z = depth_m
        
        # Total distance from camera center
        total_distance = np.sqrt(X**2 + Y**2 + Z**2)
        
        return {
            'depth_m': depth_m,
            'distance_m': total_distance,
            '3d_coords': (X, Y, Z),
            'disparity': d
        }, "Valid"
    
    def measure_common_points(self, folders, algorithm, points):
        """Measure the same points across multiple folders."""
        results = []
        
        print(f"📏 Measuring {len(points)} points across {len(folders)} folders")
        print(f"Algorithm: {algorithm}")
        print(f"Points: {points}")
        print()
        
        for folder in folders:
            data = self.load_folder_data(folder, algorithm)
            if data is None:
                print(f"❌ {Path(folder).name}: No data available")
                continue
            
            folder_results = {
                'folder': data['folder'],
                'algorithm': algorithm,
                'points': []
            }
            
            print(f"📂 {data['folder']}:")
            for i, (x, y) in enumerate(points):
                measurement, status = self.measure_distance(data, x, y, algorithm)
                
                point_result = {
                    'point_id': i + 1,
                    'pixel': (x, y),
                    'status': status
                }
                
                if measurement is not None:
                    point_result.update(measurement)
                    print(f"  Point {i+1} ({x},{y}): {measurement['distance_m']:.3f}m ({measurement['distance_m']*100:.1f}cm)")
                else:
                    print(f"  Point {i+1} ({x},{y}): {status}")
                
                folder_results['points'].append(point_result)
            
            results.append(folder_results)
            print()
        
        return results
    
    def analyze_consistency(self, results):
        """Analyze consistency of measurements across folders."""
        print("📊 CONSISTENCY ANALYSIS")
        print("=" * 50)
        
        # Group by point
        num_points = len(results[0]['points']) if results else 0
        
        for point_idx in range(num_points):
            valid_distances = []
            
            for folder_result in results:
                point = folder_result['points'][point_idx]
                if point['status'] == 'Valid':
                    valid_distances.append(point['distance_m'])
            
            if len(valid_distances) > 1:
                mean_dist = np.mean(valid_distances)
                std_dist = np.std(valid_distances)
                cv = (std_dist / mean_dist) * 100  # Coefficient of variation
                
                print(f"Point {point_idx + 1}:")
                print(f"  Valid measurements: {len(valid_distances)}/{len(results)}")
                print(f"  Mean distance: {mean_dist:.3f} ± {std_dist:.3f} m")
                print(f"  Range: {min(valid_distances):.3f} - {max(valid_distances):.3f} m")
                print(f"  Coefficient of variation: {cv:.1f}%")
                
                if cv < 5:
                    print(f"  Quality: ✅ Excellent (CV < 5%)")
                elif cv < 10:
                    print(f"  Quality: ✅ Good (CV < 10%)")
                elif cv < 20:
                    print(f"  Quality: ⚠️  Fair (CV < 20%)")
                else:
                    print(f"  Quality: ❌ Poor (CV > 20%)")
                print()
    
    def save_results_csv(self, results, output_file):
        """Save results to CSV file."""
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['folder', 'algorithm', 'point_id', 'pixel_x', 'pixel_y', 
                         'depth_m', 'distance_m', 'x_3d', 'y_3d', 'z_3d', 'disparity', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for folder_result in results:
                for point in folder_result['points']:
                    row = {
                        'folder': folder_result['folder'],
                        'algorithm': folder_result['algorithm'],
                        'point_id': point['point_id'],
                        'pixel_x': point['pixel'][0],
                        'pixel_y': point['pixel'][1],
                        'status': point['status']
                    }
                    
                    if point['status'] == 'Valid':
                        row.update({
                            'depth_m': point['depth_m'],
                            'distance_m': point['distance_m'],
                            'x_3d': point['3d_coords'][0],
                            'y_3d': point['3d_coords'][1],
                            'z_3d': point['3d_coords'][2],
                            'disparity': point['disparity']
                        })
                    
                    writer.writerow(row)
        
        print(f"💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch distance measurement tool for stereo vision validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python distance_measurement.py --folders test/000040 test/000080 --algorithm HITNET --points 200,200 400,300
  python distance_measurement.py --folders test/00004* --algorithm SGBM --points 100,100 200,200 300,300 --output distances.csv
        """
    )
    
    parser.add_argument(
        '--folders',
        type=str,
        nargs='+',
        required=True,
        help='Paths to folders containing processed stereo data'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['SGBM', 'HITNET'],
        default='HITNET',
        help='Algorithm to analyze (default: HITNET)'
    )
    
    parser.add_argument(
        '--points',
        type=str,
        nargs='+',
        required=True,
        help='Points to measure in format "x,y" (e.g., 200,200 400,300)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for results (optional)'
    )
    
    args = parser.parse_args()
    
    # Parse points
    points = []
    for point_str in args.points:
        try:
            x, y = map(int, point_str.split(','))
            points.append((x, y))
        except ValueError:
            print(f"Warning: Invalid point format '{point_str}'. Use 'x,y' format.")
    
    if not points:
        print("Error: No valid points provided")
        return 1
    
    # Expand glob patterns in folder names
    import glob
    expanded_folders = []
    for folder_pattern in args.folders:
        matches = glob.glob(folder_pattern)
        if matches:
            expanded_folders.extend(matches)
        else:
            expanded_folders.append(folder_pattern)  # Keep original if no matches
    
    # Remove duplicates and sort
    folders = sorted(list(set(expanded_folders)))
    
    try:
        tool = BatchDistanceMeasurement()
        results = tool.measure_common_points(folders, args.algorithm, points)
        
        if results:
            tool.analyze_consistency(results)
            
            if args.output:
                tool.save_results_csv(results, args.output)
        else:
            print("❌ No valid results found")
            return 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 