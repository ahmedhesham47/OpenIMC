#!/usr/bin/env python3
"""
Scalability benchmark for feature extraction.

This script measures RAM usage, wall time (elapsed real-world time), and maximum 
resident set size (RSS) for feature extraction while varying:
1. Number of images
2. Number of workers

Metrics:
- wall_time: Elapsed real-world time from start to finish (wall-clock time)
- peak_ram_mb: Peak virtual memory size (total memory allocated)
- max_rss_mb: Maximum resident set size (physical RAM actually used)

Uses the same core functions as the CLI tool `openimc extract-features`.

Usage:
    python feature_extraction_scalability.py \
        --images-dir /path/to/images \
        --masks-dir /path/to/masks \
        --output-dir ./results \
        --num-images 1 2 4 8 16 \
        --num-workers 1 2 4 8

The images directory should contain:
    - Image files (e.g., .mcd, .mcdx files) OR directories of OME-TIFF files

The masks directory should contain:
    - Mask files (e.g., .tif, .tiff files) - matching handled by core functions
"""

import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import psutil

# Add parent directory to path to import openimc
# Path: figures/Scalability_benchmarks/feature_extraction/ -> OpenIMC root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openimc.core import extract_features, load_mcd, _build_feature_selection_dict
from openimc.processing.feature_worker import load_and_extract_features


def get_memory_usage_mb() -> Tuple[float, float]:
    """Get current memory usage in MB.
    
    Returns:
        (rss_mb, vms_mb) - Resident Set Size and Virtual Memory Size
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / 1024 / 1024  # Resident Set Size (physical RAM)
    vms_mb = mem_info.vms / 1024 / 1024  # Virtual Memory Size (total virtual memory)
    return rss_mb, vms_mb


def reset_memory():
    """Reset memory state before a new benchmark run.
    
    Clears caches, forces garbage collection, and allows system to settle.
    This ensures each benchmark starts with a clean memory state.
    """
    # Force multiple garbage collections to clear all references
    for _ in range(3):
        gc.collect()
    
    # Small delay to let system settle and release memory
    time.sleep(0.5)


def find_image_files(images_dir: Path) -> List[Path]:
    """Find all image files/directories - simple file finding only."""
    image_extensions = {'.mcd', '.mcdx'}
    image_files = []
    
    # Find MCD files
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'**/*{ext}'))
    
    # Check if images_dir itself is an OME-TIFF directory (contains TIFF files)
    tiff_files = list(images_dir.glob('*.ome.tif')) + list(images_dir.glob('*.ome.tiff')) + \
                 list(images_dir.glob('*.tif')) + list(images_dir.glob('*.tiff'))
    if tiff_files:
        image_files.append(images_dir)
    
    # Find OME-TIFF subdirectories
    for item in images_dir.iterdir():
        if item.is_dir():
            tiff_files = list(item.glob('*.ome.tif')) + list(item.glob('*.ome.tiff')) + \
                         list(item.glob('*.tif')) + list(item.glob('*.tiff'))
            if tiff_files:
                image_files.append(item)
    
    return sorted(image_files)


def run_feature_extraction_benchmark(
    image_files: List[Path],
    masks_dir: Path,
    num_images: int,
    num_workers: int,
    output_dir: Path,
    repeat: int = 0
) -> Dict:
    """
    Run feature extraction benchmark using core functions.
    
    Returns:
        Dictionary with metrics:
        - wall_time: Wall-clock time (elapsed real-world time) in seconds
        - peak_ram_mb: Peak virtual memory size (VMS) in MB
        - max_rss_mb: Maximum resident set size (physical RAM used) in MB
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {num_images} acquisitions, {num_workers} workers")
    print(f"{'='*60}")
    
    # Reset memory state before starting benchmark
    reset_memory()
    
    # Load images to get acquisitions (metadata only, no image data)
    # Collect acquisitions from all available images until we have enough
    all_acquisitions = []
    acq_to_input_path = {}
    acq_to_loader_type = {}
    
    loaders = []
    for img_path in image_files:
        if len(all_acquisitions) >= num_images:
            break
        
        try:
            loader, loader_type = load_mcd(str(img_path), channel_format='CHW')
            loaders.append(loader)
            
            # Clear any image cache before getting acquisitions (same as GUI)
            if hasattr(loader, '_image_cache'):
                loader._image_cache.clear()
            
            acquisitions = loader.list_acquisitions()
            
            for acq in acquisitions:
                if len(all_acquisitions) >= num_images:
                    break
                all_acquisitions.append(acq)
                acq_to_input_path[acq.id] = str(img_path)
                acq_to_loader_type[acq.id] = loader_type
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue
    
    # Close loaders and clear caches (will reload in workers, same as GUI)
    for loader in loaders:
        # Clear image cache if it exists (same as GUI does)
        if hasattr(loader, '_image_cache'):
            loader._image_cache.clear()
        if hasattr(loader, 'close'):
            loader.close()
    loaders.clear()
    gc.collect()
    
    if len(all_acquisitions) == 0:
        return {
            'num_images': num_images,
            'num_workers': num_workers,
            'wall_time': None,
            'peak_ram_mb': None,
            'max_rss_mb': None,
            'success': False
        }
    
    # Limit to exactly num_images acquisitions
    all_acquisitions = all_acquisitions[:num_images]
    
    # Update mappings to only include selected acquisitions
    selected_acq_ids = {acq.id for acq in all_acquisitions}
    acq_to_input_path = {k: v for k, v in acq_to_input_path.items() if k in selected_acq_ids}
    acq_to_loader_type = {k: v for k, v in acq_to_loader_type.items() if k in selected_acq_ids}
    
    print(f"Processing {len(all_acquisitions)} acquisitions (requested {num_images})")
    
    # Use core function to build feature selection
    selected_features = _build_feature_selection_dict(morphological=True, intensity=True)
    
    # Clear caches before starting benchmark (same as GUI does)
    # This ensures images are not preloaded and we measure actual extraction time
    gc.collect()
    
    # Start timing (after cache clearing, same as GUI)
    start_time = time.time()
    start_rss, start_vms = get_memory_usage_mb()
    peak_vms = start_vms  # Peak virtual memory (total memory allocated)
    max_rss = start_rss   # Max resident set size (physical RAM used)
    
    try:
        if num_workers > 1 and len(all_acquisitions) > 1:
            # Multiprocessing - use existing worker function
            ctx = mp.get_context('spawn')
            mp_args = []
            
            for acq in all_acquisitions:
                acq_info_dict = {
                    'channels': acq.channels,
                    'name': acq.name,
                    'well': acq.well,
                    'id': acq.id,
                    'channel_metals': acq.channel_metals,
                    'channel_labels': acq.channel_labels,
                }
                
                mp_args.append((
                    acq.id,
                    None,  # mask array
                    str(masks_dir),  # mask path - core will handle matching
                    selected_features,
                    acq_info_dict,
                    acq.well if acq.well else acq.name,
                    acq_to_input_path[acq.id],
                    acq_to_loader_type[acq.id],
                    False,  # arcsinh_enabled
                    1.0,  # cofactor
                    "None",  # denoise_source
                    None,  # custom_denoise_settings
                    None,  # spillover_config
                    acq.source_file if hasattr(acq, 'source_file') else None,
                    None  # excluded_channels
                ))
            
            with ctx.Pool(processes=num_workers) as pool:
                futures = [pool.apply_async(load_and_extract_features, args) for args in mp_args]
                
                results = []
                for future in futures:
                    try:
                        result = future.get(timeout=600)
                        if not result.empty:
                            results.append(result)
                    except Exception as e:
                        print(f"Error in worker: {e}")
                
                # Monitor memory
                while any(not f.ready() for f in futures):
                    current_rss, current_vms = get_memory_usage_mb()
                    peak_vms = max(peak_vms, current_vms)
                    max_rss = max(max_rss, current_rss)
                    time.sleep(0.1)
        else:
            # Single-threaded - use core extract_features directly
            # Group acquisitions by input path to process efficiently
            results = []
            input_to_acqs = {}
            for acq in all_acquisitions:
                input_path = acq_to_input_path[acq.id]
                if input_path not in input_to_acqs:
                    input_to_acqs[input_path] = []
                input_to_acqs[input_path].append(acq)
            
            for input_path, acquisitions in input_to_acqs.items():
                try:
                    loader, _ = load_mcd(input_path, channel_format='CHW')
                    try:
                        # Clear image cache before processing (same as GUI)
                        if hasattr(loader, '_image_cache'):
                            loader._image_cache.clear()
                        
                        current_rss, current_vms = get_memory_usage_mb()
                        peak_vms = max(peak_vms, current_vms)
                        max_rss = max(max_rss, current_rss)
                        
                        # Use core function - it handles mask matching
                        # Images will be loaded on-demand during extraction, not preloaded
                        result = extract_features(
                            loader=loader,
                            acquisitions=acquisitions,
                            mask_path=str(masks_dir),
                            output_path=None,
                            morphological=True,
                            intensity=True,
                            arcsinh=False,
                            arcsinh_cofactor=1.0,
                            denoise_settings=None,
                            spillover_config=None,
                            excluded_channels=None
                        )
                        
                        if not result.empty:
                            results.append(result)
                    finally:
                        # Clear cache before closing (same as GUI)
                        if hasattr(loader, '_image_cache'):
                            loader._image_cache.clear()
                        loader.close()
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
                    continue
        
        end_time = time.time()
        wall_time = end_time - start_time
        final_rss, final_vms = get_memory_usage_mb()
        peak_vms = max(peak_vms, final_vms)
        max_rss = max(max_rss, final_rss)
        
        # Combine all results into a single DataFrame
        combined_features = pd.DataFrame()
        if len(results) > 1:
            combined_features = pd.concat(results, ignore_index=True)
        elif len(results) == 1:
            combined_features = results[0]
        
        # Save features to file for reproducibility testing
        if not combined_features.empty:
            output_data_dir = output_dir / 'output_data'
            output_data_dir.mkdir(parents=True, exist_ok=True)
            features_filename = f"{num_images}_{num_workers}_{repeat}.csv"
            features_path = output_data_dir / features_filename
            combined_features.to_csv(features_path, index=False)
            print(f"Saved features to: {features_path} ({len(combined_features)} cells)")
        
        print(f"Completed: {wall_time:.2f}s, Peak VMS: {peak_vms:.1f} MB, Max RSS: {max_rss:.1f} MB")
        
        return {
            'num_images': num_images,
            'num_workers': num_workers,
            'wall_time': wall_time,
            'peak_ram_mb': peak_vms,  # Virtual memory size (total allocated)
            'max_rss_mb': max_rss,    # Resident set size (physical RAM)
            'success': True
        }
    
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {
            'num_images': num_images,
            'num_workers': num_workers,
            'wall_time': time.time() - start_time,
            'peak_ram_mb': peak_vms,
            'max_rss_mb': max_rss,
            'success': False,
            'error': str(e)
        }
    
    finally:
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description='Scalability benchmark for feature extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing image files (MCD files or OME-TIFF directories)')
    parser.add_argument('--masks-dir', type=str, required=True,
                       help='Directory containing mask files (.tif, .tiff)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--num-images', type=int, nargs='+', default=[50, 100, 200, 500],
                       help='Number of images to test (default: 1 2 4 8)')
    parser.add_argument('--num-workers', type=int, nargs='+', default=[1, 4, 8, 16, 22],
                       help='Number of workers to test (default: 1 2 4 8)')
    parser.add_argument('--repeats', type=int, default=3,
                       help='Number of times to repeat each configuration (default: 1)')
    parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW',
                       help='Channel format for OME-TIFF files (default: CHW)')
    
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
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image files
    print(f"Scanning for images in {images_dir}...")
    image_files = find_image_files(images_dir)
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {images_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files/directories")
    print(f"Masks directory: {masks_dir}")
    print("Mask matching handled by core functions")
    
    # Run benchmarks
    results = []
    for num_images in args.num_images:
        for num_workers in args.num_workers:
            for repeat in range(args.repeats):
                # Reset memory between each benchmark run to ensure clean state
                if repeat > 0 or (num_images != args.num_images[0] or num_workers != args.num_workers[0]):
                    print("\nResetting memory before next benchmark...")
                    reset_memory()
                
                result = run_feature_extraction_benchmark(
                    image_files, masks_dir, num_images, num_workers, output_dir, repeat
                )
                result['repeat'] = repeat
                results.append(result)
                
                # Reset memory after each run to ensure clean state for next run
                reset_memory()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / 'scalability_results.csv'
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


if __name__ == '__main__':
    main()
