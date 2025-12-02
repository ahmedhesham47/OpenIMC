#!/usr/bin/env python3
"""
Scalability benchmark for feature extraction using steinbock.

This script measures RAM usage, wall time, and maximum resident set size (RSS)
for feature extraction using steinbock while varying:
1. Number of images
2. Number of workers (if supported by steinbock)

Runs two separate steinbock commands:
- `steinbock measure intensities` - measures object intensities
- `steinbock measure regionprops` - measures morphological properties

Uses `/usr/bin/time -v` to measure performance metrics for each command,
then combines the metrics (sums wall time, takes max of RAM/RSS).

Usage:
    python steinbock_feature_extraction_scalability.py \
        --images-dir /path/to/images \
        --masks-dir /path/to/masks \
        --output-dir ./results \
        --num-images 1 2 4 8 16 \
        --num-workers 1

The images directory should contain:
    - Image files compatible with steinbock (typically OME-TIFF files)

The masks directory should contain:
    - Mask files (e.g., .tif, .tiff files) matching the images
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_time_output(time_output: str) -> Dict[str, Optional[float]]:
    """
    Parse output from `/usr/bin/time -v` to extract metrics.
    
    Returns:
        Dictionary with metrics: wall_time, peak_ram_mb, max_rss_mb
    """
    metrics = {
        'wall_time': None,
        'peak_ram_mb': None,
        'max_rss_mb': None
    }
    
    # Parse elapsed (wall-clock) time
    # Format: "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:23.45"
    elapsed_match = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:\.]+)', time_output)
    if elapsed_match:
        time_str = elapsed_match.group(1)
        # Convert to seconds
        parts = time_str.split(':')
        if len(parts) == 2:  # m:ss
            minutes, seconds = parts
            metrics['wall_time'] = float(minutes) * 60 + float(seconds)
        elif len(parts) == 3:  # h:mm:ss
            hours, minutes, seconds = parts
            metrics['wall_time'] = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    
    # Parse Maximum resident set size (kbytes)
    # Format: "Maximum resident set size (kbytes): 1234567"
    rss_match = re.search(r'Maximum resident set size \(kbytes\): ([\d]+)', time_output)
    if rss_match:
        rss_kb = int(rss_match.group(1))
        metrics['max_rss_mb'] = rss_kb / 1024  # Convert KB to MB
        metrics['peak_ram_mb'] = metrics['max_rss_mb']  # Use same value for peak RAM
    
    return metrics


def find_image_files(images_dir: Path) -> List[Path]:
    """Find all image files compatible with steinbock."""
    image_extensions = {'.tif', '.tiff', '.ome.tif', '.ome.tiff'}
    image_files = []
    
    # Find TIFF files
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'**/*{ext}'))
    
    return sorted(set(image_files))


def find_mask_files(masks_dir: Path) -> List[Path]:
    """Find all mask files."""
    mask_extensions = {'.tif', '.tiff', '.ome.tif', '.ome.tiff'}
    mask_files = []
    
    for ext in mask_extensions:
        mask_files.extend(masks_dir.glob(f'*{ext}'))
        mask_files.extend(masks_dir.glob(f'**/*{ext}'))
    
    return sorted(set(mask_files))


def create_subset_directories(
    image_files: List[Path],
    mask_files: List[Path],
    num_images: int,
    base_temp_dir: Path
) -> Tuple[Path, Path]:
    """
    Create temporary directories with subset of images and masks.
    
    Returns:
        Tuple of (images_subset_dir, masks_subset_dir)
    """
    selected_images = image_files[:num_images]
    
    if len(selected_images) < num_images:
        print(f"Warning: Only {len(selected_images)} images available, requested {num_images}")
    
    # Create subset directories
    images_subset_dir = base_temp_dir / f'images_subset_{num_images}'
    masks_subset_dir = base_temp_dir / f'masks_subset_{num_images}'
    
    images_subset_dir.mkdir(parents=True, exist_ok=True)
    masks_subset_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy selected images
    for img_path in selected_images:
        # Copy image file
        dest_img = images_subset_dir / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # Try to find matching mask file
        # Common patterns: same name, or with _mask suffix, etc.
        mask_found = False
        for mask_path in mask_files:
            # Try exact name match
            if mask_path.name == img_path.name:
                shutil.copy2(mask_path, masks_subset_dir / mask_path.name)
                mask_found = True
                break
            # Try with common mask naming patterns
            img_stem = img_path.stem
            mask_stem = mask_path.stem
            if (mask_stem == img_stem or 
                mask_stem == f"{img_stem}_mask" or
                mask_stem == f"{img_stem}_cell" or
                img_stem == f"{mask_stem}_mask" or
                img_stem == f"{mask_stem}_cell"):
                shutil.copy2(mask_path, masks_subset_dir / mask_path.name)
                mask_found = True
                break
        
        if not mask_found:
            print(f"Warning: No matching mask found for {img_path.name}")
    
    return images_subset_dir, masks_subset_dir


def run_steinbock_command(
    cmd: List[str],
    description: str,
    timeout: int = 3600
) -> Tuple[Dict[str, Optional[float]], bool, Optional[str]]:
    """
    Run a steinbock command with time measurement.
    
    Returns:
        Tuple of (metrics_dict, success, error_message)
    """
    print(f"Running {description}: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Combine stdout and stderr (time output goes to stderr)
        combined_output = result.stdout + result.stderr
        
        # Parse time output
        metrics = parse_time_output(combined_output)
        
        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            print(f"Error: {description} failed with return code {result.returncode}")
            print(f"stderr: {error_msg}")
            return metrics, False, error_msg
        else:
            return metrics, True, None
    
    except subprocess.TimeoutExpired:
        print(f"Error: {description} timed out after {timeout} seconds")
        return {
            'wall_time': None,
            'peak_ram_mb': None,
            'max_rss_mb': None
        }, False, 'Timeout'
    except Exception as e:
        print(f"Error running {description}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'wall_time': None,
            'peak_ram_mb': None,
            'max_rss_mb': None
        }, False, str(e)


def run_steinbock_benchmark(
    image_files: List[Path],
    mask_files: List[Path],
    num_images: int,
    num_workers: int,
    output_dir: Path,
    base_temp_dir: Path
) -> Dict:
    """
    Run feature extraction benchmark using steinbock.
    
    Runs both 'measure intensities' and 'measure regionprops' commands
    and combines their metrics.
    
    Returns:
        Dictionary with metrics: wall_time, peak_ram_mb, max_rss_mb
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {num_images} images, {num_workers} workers")
    print(f"{'='*60}")
    
    # Create subset directories
    images_subset_dir, masks_subset_dir = create_subset_directories(
        image_files, mask_files, num_images, base_temp_dir
    )
    
    # Create output directories for this run
    intensities_output_dir = base_temp_dir / f'intensities_{num_images}'
    regionprops_output_dir = base_temp_dir / f'regionprops_{num_images}'
    intensities_output_dir.mkdir(parents=True, exist_ok=True)
    regionprops_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build steinbock measure intensities command
    intensities_cmd = [
        '/usr/bin/time', '-v',
        'steinbock', 'measure', 'intensities',
        '--img', str(images_subset_dir),
        '--masks', str(masks_subset_dir),
        '-o', str(intensities_output_dir)
    ]
    
    # Build steinbock measure regionprops command
    # Common regionprops: area, eccentricity, extent, major_axis_length, 
    # minor_axis_length, orientation, perimeter, solidity
    regionprops_cmd = [
        '/usr/bin/time', '-v',
        'steinbock', 'measure', 'regionprops',
        '--img', str(images_subset_dir),
        '--masks', str(masks_subset_dir),
        '-o', str(regionprops_output_dir),
        'area', 'eccentricity', 'extent', 'major_axis_length',
        'minor_axis_length', 'orientation', 'perimeter', 'solidity'
    ]
    
    # Run intensities measurement
    intensities_metrics, intensities_success, intensities_error = run_steinbock_command(
        intensities_cmd,
        'steinbock measure intensities',
        timeout=3600
    )
    
    if not intensities_success:
        return {
            'num_images': num_images,
            'num_workers': num_workers,
            'wall_time': None,
            'peak_ram_mb': None,
            'max_rss_mb': None,
            'success': False,
            'error': f'intensities: {intensities_error}'
        }
    
    # Run regionprops measurement
    regionprops_metrics, regionprops_success, regionprops_error = run_steinbock_command(
        regionprops_cmd,
        'steinbock measure regionprops',
        timeout=3600
    )
    
    if not regionprops_success:
        return {
            'num_images': num_images,
            'num_workers': num_workers,
            'wall_time': intensities_metrics.get('wall_time'),
            'peak_ram_mb': intensities_metrics.get('peak_ram_mb'),
            'max_rss_mb': intensities_metrics.get('max_rss_mb'),
            'success': False,
            'error': f'regionprops: {regionprops_error}'
        }
    
    # Combine metrics from both commands
    # Wall time: sum of both
    combined_wall_time = None
    if intensities_metrics.get('wall_time') is not None and regionprops_metrics.get('wall_time') is not None:
        combined_wall_time = intensities_metrics['wall_time'] + regionprops_metrics['wall_time']
    elif intensities_metrics.get('wall_time') is not None:
        combined_wall_time = intensities_metrics['wall_time']
    elif regionprops_metrics.get('wall_time') is not None:
        combined_wall_time = regionprops_metrics['wall_time']
    
    # Peak RAM and Max RSS: take maximum of both
    combined_peak_ram = None
    if intensities_metrics.get('peak_ram_mb') is not None and regionprops_metrics.get('peak_ram_mb') is not None:
        combined_peak_ram = max(intensities_metrics['peak_ram_mb'], regionprops_metrics['peak_ram_mb'])
    elif intensities_metrics.get('peak_ram_mb') is not None:
        combined_peak_ram = intensities_metrics['peak_ram_mb']
    elif regionprops_metrics.get('peak_ram_mb') is not None:
        combined_peak_ram = regionprops_metrics['peak_ram_mb']
    
    combined_max_rss = None
    if intensities_metrics.get('max_rss_mb') is not None and regionprops_metrics.get('max_rss_mb') is not None:
        combined_max_rss = max(intensities_metrics['max_rss_mb'], regionprops_metrics['max_rss_mb'])
    elif intensities_metrics.get('max_rss_mb') is not None:
        combined_max_rss = intensities_metrics['max_rss_mb']
    elif regionprops_metrics.get('max_rss_mb') is not None:
        combined_max_rss = regionprops_metrics['max_rss_mb']
    
    metrics = {
        'num_images': num_images,
        'num_workers': num_workers,
        'wall_time': combined_wall_time,
        'peak_ram_mb': combined_peak_ram,
        'max_rss_mb': combined_max_rss,
        'success': True
    }
    
    if metrics['wall_time'] is not None:
        print(f"Completed: {metrics['wall_time']:.2f}s, "
              f"Peak RAM: {metrics['peak_ram_mb']:.1f} MB, "
              f"Max RSS: {metrics['max_rss_mb']:.1f} MB")
    else:
        print("Warning: Could not parse all metrics from time output")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Scalability benchmark for feature extraction using steinbock',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing image files (OME-TIFF files)')
    parser.add_argument('--masks-dir', type=str, required=True,
                       help='Directory containing mask files (.tif, .tiff)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--num-images', type=int, nargs='+', default=[50, 100, 200, 500],
                       help='Number of images to test (default: 50 100 200 500)')
    parser.add_argument('--num-workers', type=int, nargs='+', default=[1],
                       help='Number of workers to test (default: 1)')
    parser.add_argument('--repeats', type=int, default=1,
                       help='Number of times to repeat each configuration (default: 1)')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary directories after benchmark (default: False)')
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    
    if not images_dir.exists():
        print(f"Error: Images directory does not exist: {images_dir}")
        sys.exit(1)
    if not masks_dir.exists():
        print(f"Error: Masks directory does not exist: {masks_dir}")
        sys.exit(1)
    
    # Check if steinbock is available
    try:
        result = subprocess.run(
            ['steinbock', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print("Warning: steinbock command may not be available")
    except FileNotFoundError:
        print("Error: steinbock command not found. Please install steinbock.")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not verify steinbock installation: {e}")
    
    # Check if /usr/bin/time is available
    try:
        result = subprocess.run(
            ['/usr/bin/time', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
    except FileNotFoundError:
        print("Error: /usr/bin/time not found. This script requires GNU time.")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image and mask files
    print(f"Scanning for images in {images_dir}...")
    image_files = find_image_files(images_dir)
    
    print(f"Scanning for masks in {masks_dir}...")
    mask_files = find_mask_files(masks_dir)
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {images_dir}")
        sys.exit(1)
    
    if len(mask_files) == 0:
        print(f"Error: No mask files found in {masks_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files")
    print(f"Found {len(mask_files)} mask files")
    
    # Create temporary directory for subsets
    base_temp_dir = output_dir / 'temp_steinbock_benchmark'
    base_temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run benchmarks
        results = []
        for num_images in args.num_images:
            for num_workers in args.num_workers:
                for repeat in range(args.repeats):
                    result = run_steinbock_benchmark(
                        image_files, mask_files, num_images, num_workers,
                        output_dir, base_temp_dir
                    )
                    result['repeat'] = repeat
                    results.append(result)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_csv = output_dir / 'steinbock_scalability_results.csv'
        results_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to {results_csv}")
        
        # Print summary
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        if 'success' in results_df.columns:
            successful = results_df[results_df['success'] == True]
            if len(successful) > 0:
                print(successful.groupby(['num_images', 'num_workers']).agg({
                    'wall_time': ['mean', 'std'],
                    'peak_ram_mb': ['mean', 'std'],
                    'max_rss_mb': ['mean', 'std']
                }))
            else:
                print("No successful runs!")
        
        print(f"\nFull results saved to: {results_csv}")
        print(f"To generate plots, run: python generate_scalability_plots.py {results_csv}")
    
    finally:
        # Clean up temporary directories
        if not args.keep_temp and base_temp_dir.exists():
            print(f"\nCleaning up temporary directory: {base_temp_dir}")
            shutil.rmtree(base_temp_dir)


if __name__ == '__main__':
    main()

