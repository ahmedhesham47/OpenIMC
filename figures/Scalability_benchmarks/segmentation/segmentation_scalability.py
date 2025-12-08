#!/usr/bin/env python3
"""
Scalability benchmark for segmentation.

This script measures RAM usage, wall time (elapsed real-world time), maximum 
resident set size (RSS), and GPU vRAM for segmentation while varying:
1. Number of images
2. Number of workers (for Cellpose, batch_size = num_workers)

Supports two segmentation methods (can run both in one execution):
- Cellpose: Batch processing where batch_size is automatically set equal to num_workers
- CellSAM: Individual image processing (without WSI mode, without cell size gauging)

Output directories are named: {num_images}_{num_workers}_{method}_{repeat}
This ensures outputs from different experiments don't overwrite each other.

Metrics:
- wall_time: Elapsed real-world time from start to finish (wall-clock time)
- peak_ram_mb: Peak virtual memory size (total memory allocated)
- max_rss_mb: Maximum resident set size (physical RAM actually used)
- peak_vram_mb: Peak GPU memory usage (if GPU is used)

For Cellpose:
- batch_size = num_workers (automatically set)
- Multiprocessing for loading/preprocessing acquisitions
- Model initialized once and reused for all batches
- model.eval() called on entire batch of images

For CellSAM:
- Processes images individually (no batching)
- use_wsi=False, gauge_cell_size=False
- Multiprocessing for loading/preprocessing acquisitions

Usage:
    # Run both methods (default):
    python segmentation_scalability.py \
        --images-dir /path/to/images \
        --output-dir ./results \
        --nuclear-channels "Histone_1261726In113Di,Histone_473968La139Di,..." \
        --cyto-channels "Cytoker_651779Pr141Di,Cytoker_3111576Nd143Di,..." \
        --num-images 50 100 200 500 \
        --num-workers 4 8 16 \
        --deepcell-api-key YOUR_API_KEY
    
    # Run only Cellpose:
    python segmentation_scalability.py \
        --images-dir /path/to/images \
        --output-dir ./results \
        --nuclear-channels "..." \
        --method cellpose \
        --num-images 50 100 200 500 \
        --num-workers 4 8 16
    
    # Run only CellSAM:
    python segmentation_scalability.py \
        --images-dir /path/to/images \
        --output-dir ./results \
        --nuclear-channels "..." \
        --method cellsam \
        --num-images 50 100 200 500 \
        --num-workers 4 8 16 \
        --deepcell-api-key YOUR_API_KEY

The images directory should contain:
    - Image files (e.g., .mcd, .mcdx files) OR directories of OME-TIFF files
"""

import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil

# Add parent directory to path to import openimc
# Path: figures/Scalability_benchmarks/segmentation/ -> OpenIMC root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openimc.core import load_mcd
from openimc.data.mcd_loader import AcquisitionInfo

# Try to import torch for GPU memory tracking
try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


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


def get_gpu_memory_mb(gpu_id: Optional[Union[int, str]] = None) -> Optional[float]:
    """Get current GPU memory usage in MB.
    
    Args:
        gpu_id: GPU ID to check (optional)
    
    Returns:
        GPU memory usage in MB, or None if GPU not available
    """
    if not _HAVE_TORCH:
        return None
    
    try:
        if gpu_id is not None and gpu_id != 'auto':
            device_id = int(gpu_id) if isinstance(gpu_id, str) else gpu_id
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)
                return torch.cuda.memory_allocated(device_id) / 1024 / 1024
        elif torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS (Apple Silicon) doesn't have memory tracking yet
            return None
    except Exception:
        pass
    
    return None


def reset_memory(gpu_id: Optional[Union[int, str]] = None):
    """Reset memory state before a new benchmark run.
    
    Clears caches, forces garbage collection, and allows system to settle.
    This ensures each benchmark starts with a clean memory state.
    """
    # Force multiple garbage collections to clear all references
    for _ in range(3):
        gc.collect()
    
    # Clear GPU cache if available
    if _HAVE_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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


def _load_and_preprocess_acquisition_worker(task_data):
    """
    Worker function to load and preprocess a single acquisition for segmentation.
    This matches the GUI's _load_and_preprocess_acquisition_worker exactly.
    """
    (original_acq_id, unique_acq_id, acq_name, file_path, loader_type, preprocessing_config,
     denoise_source, custom_denoise_settings, source_file, channel_format) = task_data
    
    try:
        # Import inside function to ensure isolation
        import numpy as np
        from openimc.data.mcd_loader import MCDLoader
        from openimc.data.ometiff_loader import OMETIFFLoader
        from openimc.ui.utils import combine_channels, channelwise_minmax_normalize
        from openimc.core import _preprocess_channels_for_segmentation
        
        # Recreate loader (can't pickle loader objects)
        loader = None
        if loader_type == "mcd":
            loader = MCDLoader()
            loader.open(file_path)
        elif loader_type == "ometiff":
            loader = OMETIFFLoader(channel_format=channel_format or 'CHW')
            loader.open(file_path)
        
        if not loader:
            return None
        
        try:
            # Clear image cache before processing (same as feature extraction)
            if hasattr(loader, '_image_cache'):
                loader._image_cache.clear()
            
            config = preprocessing_config
            
            # Get nuclear channels
            nuclear_channels = config.get('nuclear_channels', [])
            if not nuclear_channels:
                return None
            
            # Get cytoplasm channels
            cyto_channels = config.get('cyto_channels', [])
            
            # Get channels, size, and metadata from loader (same as list_acquisitions does)
            channels = loader.get_channels(original_acq_id)
            
            # Get channel_metals, channel_labels, size, metadata, and well from loader's internal dictionaries
            channel_metals = []
            channel_labels = []
            size = (None, None)
            metadata = {}
            well = None
            
            if hasattr(loader, '_acq_channel_metals'):
                channel_metals = loader._acq_channel_metals.get(original_acq_id, [])
            if hasattr(loader, '_acq_channel_labels'):
                channel_labels = loader._acq_channel_labels.get(original_acq_id, [])
            if hasattr(loader, '_acq_size'):
                size = loader._acq_size.get(original_acq_id, (None, None))
            if hasattr(loader, '_acq_metadata'):
                metadata = loader._acq_metadata.get(original_acq_id, {})
            if hasattr(loader, '_acq_well'):
                well = loader._acq_well.get(original_acq_id)
            
            # Create AcquisitionInfo for preprocessing (same structure as list_acquisitions returns)
            acq_info = AcquisitionInfo(
                id=unique_acq_id,
                name=acq_name,
                well=well,
                size=size,
                channels=channels,
                channel_metals=channel_metals if isinstance(channel_metals, list) else [],
                channel_labels=channel_labels if isinstance(channel_labels, list) else [],
                metadata=metadata,
                source_file=source_file
            )
            
            # Use core preprocessing function (imported from core)
            # Note: _preprocess_channels_for_segmentation is used internally by segment()
            # We replicate the same logic here to match GUI behavior
            from openimc.core import _preprocess_channels_for_segmentation as preprocess_channels
            
            nuclear_img, cyto_img = preprocess_channels(
                loader, acq_info,
                nuclear_channels, cyto_channels,
                denoise_settings=custom_denoise_settings,
                normalization_method=config.get('normalization_method', 'channelwise_minmax'),
                arcsinh_cofactor=config.get('arcsinh_cofactor', 1.0),
                percentile_params=config.get('percentile_params', (1.0, 99.0)),
                nuclear_combo_method=config.get('nuclear_combo_method', 'mean'),
                cyto_combo_method=config.get('cyto_combo_method', 'mean'),
                nuclear_weights=config.get('nuclear_weights'),
                cyto_weights=config.get('cyto_weights')
            )
            
            # Return result with acquisition info
            return {
                'acq_id': unique_acq_id,
                'acq_name': acq_name,
                'nuclear_img': nuclear_img,
                'cyto_img': cyto_img,
                'source_file': source_file
            }
        finally:
            # Clear cache before closing (same as feature extraction)
            if hasattr(loader, '_image_cache'):
                loader._image_cache.clear()
            loader.close()
        
    except Exception as e:
        print(f"Error processing acquisition {acq_name} ({unique_acq_id}): {e}")
        import traceback
        traceback.print_exc()
        return None


def run_segmentation_benchmark(
    image_files: List[Path],
    num_images: int,
    output_dir: Path,
    nuclear_channels: List[str],
    cyto_channels: List[str],
    method: str = 'cellpose',  # 'cellpose' or 'cellsam'
    cellpose_model: str = 'cyto3',
    diameter: Optional[int] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu_id: Optional[Union[int, str]] = None,
    denoise_settings: Optional[Dict] = None,
    channel_format: str = 'CHW',
    num_workers: Optional[int] = None,  # Number of workers for multiprocessing (None = auto)
    repeat: int = 0,
    # CellSAM parameters
    deepcell_api_key: Optional[str] = None,
    bbox_threshold: float = 0.4
) -> Dict:
    """
    Run segmentation benchmark using EXACT same approach as GUI.
    
    For Cellpose:
    - batch_size = num_workers (set automatically)
    - Multiprocessing for loading/preprocessing
    - Model initialized once and reused
    - model.eval() called on entire batch
    
    For CellSAM:
    - Processes images individually (no batching)
    - use_wsi=False, gauge_cell_size=False (as requested)
    - Multiprocessing for loading/preprocessing
    
    Returns:
        Dictionary with metrics including vRAM
    """
    # For Cellpose, set batch_size equal to num_workers
    if method == 'cellpose':
        if num_workers is None:
            # Auto-calculate num_workers, then use it as batch_size
            effective_num_workers = max(1, min(mp.cpu_count() - 2, num_images))
            batch_size = effective_num_workers
        else:
            batch_size = num_workers
    else:
        batch_size = None  # Not used for CellSAM
    
    print(f"\n{'='*60}")
    print(f"Benchmark: {num_images} acquisitions, method={method}")
    if method == 'cellpose':
        print(f"  batch_size={batch_size} (equal to num_workers), num_workers={num_workers or 'auto'}")
    else:
        print(f"  num_workers={num_workers or 'auto'}")
    print(f"{'='*60}")
    
    # Reset memory state before starting benchmark
    reset_memory(gpu_id)
    
    # Load images to get acquisitions (metadata only, no image data)
    all_acquisitions = []
    acq_to_input_path = {}
    acq_to_loader_type = {}
    acq_to_original_id = {}
    
    loaders = []
    for img_path in image_files:
        if len(all_acquisitions) >= num_images:
            break
        
        try:
            loader, loader_type = load_mcd(str(img_path), channel_format=channel_format)
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
                # Extract original acquisition ID (handles multi-file unique IDs)
                from openimc.core import _extract_original_acq_id
                acq_to_original_id[acq.id] = _extract_original_acq_id(acq.id)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue
    
    # Close loaders and clear caches (will reload in workers, same as GUI)
    for loader in loaders:
        if hasattr(loader, '_image_cache'):
            loader._image_cache.clear()
        if hasattr(loader, 'close'):
            loader.close()
    loaders.clear()
    gc.collect()
    
    if len(all_acquisitions) == 0:
        effective_num_workers = num_workers or max(1, min(mp.cpu_count() - 2, num_images))
        return {
            'num_images': num_images,
            'method': method,
            'batch_size': batch_size if method == 'cellpose' else None,
            'num_workers': effective_num_workers,
            'wall_time': None,
            'peak_ram_mb': None,
            'max_rss_mb': None,
            'peak_vram_mb': None,
            'num_cells_total': 0,
            'num_acquisitions': 0,
            'success': False
        }
    
    # Limit to exactly num_images acquisitions
    all_acquisitions = all_acquisitions[:num_images]
    
    # Update mappings to only include selected acquisitions
    selected_acq_ids = {acq.id for acq in all_acquisitions}
    acq_to_input_path = {k: v for k, v in acq_to_input_path.items() if k in selected_acq_ids}
    acq_to_loader_type = {k: v for k, v in acq_to_loader_type.items() if k in selected_acq_ids}
    acq_to_original_id = {k: v for k, v in acq_to_original_id.items() if k in selected_acq_ids}
    
    print(f"Processing {len(all_acquisitions)} acquisitions (requested {num_images})")
    
    # Determine effective num_workers for output directory naming
    if num_workers is None:
        effective_num_workers = max(1, min(mp.cpu_count() - 2, num_images))
    else:
        effective_num_workers = num_workers
    
    # Create output directory for masks: {num_images}_{num_workers}_{method}_{repeat}
    masks_output_dir = output_dir / 'output_data' / 'masks' / f"{num_images}_{effective_num_workers}_{method}_{repeat}"
    masks_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear caches before starting benchmark (same as feature extraction)
    # This ensures images are not preloaded and we measure actual segmentation time
    gc.collect()
    
    # Start timing
    start_time = time.time()
    start_rss, start_vms = get_memory_usage_mb()
    peak_vms = start_vms
    max_rss = start_rss
    peak_vram = get_gpu_memory_mb(gpu_id) or 0.0
    
    try:
        if method == 'cellsam':
            # Initialize CellSAM
            try:
                from openimc.processing.custom_cellsam import cellsam_pipeline_custom
            except (ImportError, OSError) as e:
                raise ImportError(f"CellSAM not installed or failed to load: {e}. Install with: pip install git+https://github.com/vanvalenlab/cellSAM.git")
            
            # Set API key from argument or environment variable
            api_key = deepcell_api_key or os.environ.get("DEEPCELL_ACCESS_TOKEN", "")
            if not api_key:
                raise ValueError("DeepCell API key is required for CellSAM. Set deepcell_api_key or DEEPCELL_ACCESS_TOKEN environment variable.")
            os.environ["DEEPCELL_ACCESS_TOKEN"] = api_key
            
            print("Using CellSAM (without WSI mode, without cell size gauging)")
        else:
            # Initialize Cellpose model once (same as GUI)
            from cellpose import models
        
        # Build preprocessing config (same as GUI)
        preprocessing_config = {
            'nuclear_channels': nuclear_channels,
            'cyto_channels': cyto_channels,
            'normalization_method': 'channelwise_minmax',
            'nuclear_combo_method': 'mean',
            'cyto_combo_method': 'mean',
            'nuclear_weights': None,
            'cyto_weights': None
        }
        
        if method == 'cellsam':
            # Process acquisitions individually for CellSAM (no batching)
            successful_segmentations = 0
            all_masks = {}
            
            # Use multiprocessing for loading/preprocessing
            mp_args = []
            for acq in all_acquisitions:
                file_path = acq_to_input_path[acq.id]
                loader_type = acq_to_loader_type[acq.id]
                original_acq_id = acq_to_original_id[acq.id]
                
                mp_args.append((
                    original_acq_id,
                    acq.id,
                    acq.name,
                    file_path,
                    loader_type,
                    preprocessing_config,
                    "none",  # denoise_source
                    denoise_settings,
                    file_path,  # source_file
                    channel_format
                ))
            
            # Use multiprocessing for loading/preprocessing
            if num_workers is None:
                max_workers = max(1, min(mp.cpu_count() - 2, len(mp_args)))
            else:
                max_workers = max(1, min(num_workers, len(mp_args)))
            
            # Load and preprocess all acquisitions
            preprocessed_results = []
            with mp.Pool(processes=max_workers) as pool:
                futures = [pool.apply_async(_load_and_preprocess_acquisition_worker, (args,)) for args in mp_args]
                
                for future in futures:
                    try:
                        result = future.get(timeout=600)
                        if result is not None:
                            preprocessed_results.append(result)
                    except Exception as e:
                        print(f"Error in worker: {e}")
                        continue
            
            # Process each acquisition with CellSAM
            for idx, result in enumerate(preprocessed_results):
                acq_id = result['acq_id']
                acq_name = result['acq_name']
                nuclear_img = result['nuclear_img']
                cyto_img = result['cyto_img']
                
                try:
                    print(f"Processing {acq_name} ({idx + 1}/{len(preprocessed_results)})...")
                    
                    # Prepare input for CellSAM (same as core.py)
                    if nuclear_channels and cyto_channels:
                        # Combined mode: H x W x 3 array
                        h, w = nuclear_img.shape
                        cellsam_input = np.zeros((h, w, 3), dtype=np.float32)
                        cellsam_input[:, :, 1] = nuclear_img  # Channel 1 is nuclear
                        cellsam_input[:, :, 2] = cyto_img if cyto_img is not None else nuclear_img  # Channel 2 is cyto
                    elif nuclear_channels:
                        # Nuclear only mode: H x W array
                        cellsam_input = nuclear_img
                    elif cyto_channels:
                        # Cyto only mode: H x W array
                        cellsam_input = cyto_img if cyto_img is not None else nuclear_img
                    else:
                        print(f"Warning: No channels available for {acq_name}")
                        continue
                    
                    # Monitor memory before segmentation
                    current_rss, current_vms = get_memory_usage_mb()
                    peak_vms = max(peak_vms, current_vms)
                    max_rss = max(max_rss, current_rss)
                    current_vram = get_gpu_memory_mb(gpu_id)
                    if current_vram is not None:
                        peak_vram = max(peak_vram, current_vram)
                    
                    # Run CellSAM (without WSI mode, without cell size gauging)
                    mask = cellsam_pipeline_custom(
                        cellsam_input,
                        bbox_threshold=bbox_threshold,
                        use_wsi=False,  # As requested
                        gauge_cell_size=False,  # As requested
                        low_contrast_enhancement=False
                    )
                    
                    # Store result
                    if isinstance(mask, np.ndarray):
                        mask = mask.copy()
                    
                    all_masks[acq_id] = mask
                    successful_segmentations += 1
                    
                    # Save mask
                    acq_info = next((a for a in all_acquisitions if a.id == acq_id), None)
                    if acq_info:
                        if acq_info.well:
                            output_filename = f"{acq_info.well}_segmentation_masks.tif"
                        else:
                            output_filename = f"{acq_info.name}_segmentation_masks.tif"
                        output_path = masks_output_dir / output_filename
                        
                        import tifffile
                        tifffile.imwrite(str(output_path), mask.astype(np.uint16), compression='lzw')
                    
                    # Clean up
                    del cellsam_input, nuclear_img
                    if cyto_img is not None:
                        del cyto_img
                    gc.collect()
                    if _HAVE_TORCH and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Monitor memory after segmentation
                    current_rss, current_vms = get_memory_usage_mb()
                    peak_vms = max(peak_vms, current_vms)
                    max_rss = max(max_rss, current_rss)
                    current_vram = get_gpu_memory_mb(gpu_id)
                    if current_vram is not None:
                        peak_vram = max(peak_vram, current_vram)
                        
                except Exception as e:
                    print(f"Error processing {acq_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        else:
            # Cellpose method - process in batches
            # Determine GPU usage and device (same as core.py)
            # Note: core.py uses: use_gpu = gpu_id is not None, device=gpu_id
            # But we need to handle "auto" and convert properly
            use_gpu = False
            gpu_device = None
            
            if gpu_id == "auto":
                if _HAVE_TORCH and torch.cuda.is_available():
                    use_gpu = True
                    gpu_device = 0
                elif _HAVE_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    use_gpu = True
                    gpu_device = 'mps'
            elif gpu_id is not None:
                use_gpu = True
                # gpu_id is already parsed in main() - should be int, str, or 'auto'
                # Cellpose expects device to be int for CUDA or 'mps' for Apple Silicon
                # Pass gpu_id directly as device (same as core.py does)
                gpu_device = gpu_id
            
            # Set CUDA device if using GPU and device is specified
            if use_gpu and _HAVE_TORCH and isinstance(gpu_device, int):
                torch.cuda.set_device(gpu_device)
            
            # Print GPU info
            if use_gpu:
                if isinstance(gpu_device, int):
                    print(f"Using GPU: device={gpu_device}")
                else:
                    print(f"Using GPU: device={gpu_device}")
            else:
                print("Using CPU")
            
            # Initialize model (same as GUI - only pass gpu parameter, not device)
            # Cellpose handles device selection internally when gpu=True
            # We set the CUDA device above if a specific GPU was requested
            if cellpose_model == 'nuclei':
                model_obj = models.Cellpose(gpu=use_gpu, model_type='nuclei')
            else:  # cyto3
                model_obj = models.Cellpose(gpu=use_gpu, model_type='cyto3')
            
            # Process acquisitions in batches (same as GUI)
            successful_segmentations = 0
            all_masks = {}
            
            for batch_start in range(0, len(all_acquisitions), batch_size):
                batch_end = min(batch_start + batch_size, len(all_acquisitions))
                batch_acquisitions = all_acquisitions[batch_start:batch_end]
                
                print(f"Processing batch {batch_start//batch_size + 1} ({len(batch_acquisitions)} acquisitions)...")
                
                try:
                    # Load and preprocess all acquisitions in this batch using multiprocessing (same as GUI)
                    mp_args = []
                    acq_id_to_acq_info = {acq.id: acq for acq in batch_acquisitions}
                    
                    for acq in batch_acquisitions:
                        file_path = acq_to_input_path[acq.id]
                        loader_type = acq_to_loader_type[acq.id]
                        original_acq_id = acq_to_original_id[acq.id]
                        
                        mp_args.append((
                            original_acq_id,
                            acq.id,
                            acq.name,
                            file_path,
                            loader_type,
                            preprocessing_config,
                            "none",  # denoise_source
                            denoise_settings,
                            file_path,  # source_file
                            channel_format
                        ))
                    
                    # Use multiprocessing for loading/preprocessing (same as GUI)
                    # Use provided num_workers or auto-calculate (same as GUI default)
                    if num_workers is None:
                        max_workers = max(1, min(mp.cpu_count() - 2, len(mp_args)))
                    else:
                        max_workers = max(1, min(num_workers, len(mp_args)))
                    batch_images = []
                    batch_channels = []
                    acquisition_mapping = []
                    acquisition_info_list = []
                    
                    with mp.Pool(processes=max_workers) as pool:
                        futures = [pool.apply_async(_load_and_preprocess_acquisition_worker, (args,)) for args in mp_args]
                        
                        for future in futures:
                            try:
                                result = future.get(timeout=600)
                                if result is None:
                                    continue
                                
                                acq_id = result['acq_id']
                                nuclear_img = result['nuclear_img']
                                cyto_img = result['cyto_img']
                                
                                if acq_id not in acq_id_to_acq_info:
                                    continue
                                
                                acq = acq_id_to_acq_info[acq_id]
                                acq_idx = len(acquisition_info_list)
                                acquisition_info_list.append(acq)
                                
                                # Prepare input images based on model type (same as GUI)
                                if cyto_img is not None:
                                    # Both nuclear and cytoplasm available
                                    batch_images.extend([cyto_img, nuclear_img])
                                    batch_channels.extend([0, 1])  # cyto, nuclear
                                    acquisition_mapping.extend([acq_idx, acq_idx])
                                else:
                                    # Only nuclear available
                                    batch_images.extend([nuclear_img, nuclear_img])
                                    batch_channels.extend([0, 0])  # nuclear, nuclear
                                    acquisition_mapping.extend([acq_idx, acq_idx])
                            except Exception as e:
                                print(f"Error in worker: {e}")
                                continue
                    
                    if not batch_images:
                        continue
                    
                    # Monitor memory
                    current_rss, current_vms = get_memory_usage_mb()
                    peak_vms = max(peak_vms, current_vms)
                    max_rss = max(max_rss, current_rss)
                    current_vram = get_gpu_memory_mb(gpu_id)
                    if current_vram is not None:
                        peak_vram = max(peak_vram, current_vram)
                    
                    # Run segmentation on the entire batch (same as GUI)
                    masks, flows, styles, diams = model_obj.eval(
                        batch_images,
                        diameter=diameter,
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                        channels=batch_channels
                    )
                    
                    # Store results using acquisition mapping (same as GUI)
                    processed_acquisitions = set()
                    for i, mask in enumerate(masks):
                        if i < len(acquisition_mapping):
                            acq_idx = acquisition_mapping[i]
                            if acq_idx not in processed_acquisitions and acq_idx < len(acquisition_info_list):
                                acq_info = acquisition_info_list[acq_idx]
                                acq_id = acq_info.id
                                
                                if isinstance(mask, np.ndarray):
                                    mask = mask.copy()
                                
                                all_masks[acq_id] = mask
                                processed_acquisitions.add(acq_idx)
                                successful_segmentations += 1
                                
                                # Save mask
                                if acq_info.well:
                                    output_filename = f"{acq_info.well}_segmentation_masks.tif"
                                else:
                                    output_filename = f"{acq_info.name}_segmentation_masks.tif"
                                output_path = masks_output_dir / output_filename
                                
                                import tifffile
                                tifffile.imwrite(str(output_path), mask.astype(np.uint16), compression='lzw')
                    
                    # Explicitly release memory (same as GUI)
                    del batch_images
                    del masks, flows, styles, diams
                    gc.collect()
                    if _HAVE_TORCH and use_gpu:
                        torch.cuda.empty_cache()
                    
                    # Monitor memory after batch
                    current_rss, current_vms = get_memory_usage_mb()
                    peak_vms = max(peak_vms, current_vms)
                    max_rss = max(max_rss, current_rss)
                    current_vram = get_gpu_memory_mb(gpu_id)
                    if current_vram is not None:
                        peak_vram = max(peak_vram, current_vram)
                    
                except Exception as e:
                    print(f"Error processing batch {batch_start//batch_size + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        end_time = time.time()
        wall_time = end_time - start_time
        final_rss, final_vms = get_memory_usage_mb()
        peak_vms = max(peak_vms, final_vms)
        max_rss = max(max_rss, final_rss)
        final_vram = get_gpu_memory_mb(gpu_id)
        if final_vram is not None:
            peak_vram = max(peak_vram, final_vram)
        
        # Calculate total cells
        num_cells_total = sum(len(np.unique(mask)) - 1 for mask in all_masks.values())
        
        print(f"Completed: {wall_time:.2f}s, Peak VMS: {peak_vms:.1f} MB, Max RSS: {max_rss:.1f} MB")
        if peak_vram > 0:
            print(f"  Peak vRAM: {peak_vram:.1f} MB")
        print(f"  Processed {successful_segmentations}/{len(all_acquisitions)} acquisitions successfully")
        print(f"  Total cells detected: {num_cells_total}")
        
        # Determine effective num_workers for return value
        if num_workers is None:
            effective_num_workers = max_workers if 'max_workers' in locals() else max(1, min(mp.cpu_count() - 2, num_images))
        else:
            effective_num_workers = num_workers
        
        return {
            'num_images': num_images,
            'method': method,
            'batch_size': batch_size if method == 'cellpose' else None,
            'num_workers': effective_num_workers,
            'wall_time': wall_time,
            'peak_ram_mb': peak_vms,
            'max_rss_mb': max_rss,
            'peak_vram_mb': peak_vram if peak_vram > 0 else None,
            'num_cells_total': num_cells_total,
            'num_acquisitions': successful_segmentations,
            'success': successful_segmentations > 0
        }
    
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        effective_num_workers = num_workers or max(1, min(mp.cpu_count() - 2, num_images))
        return {
            'num_images': num_images,
            'method': method,
            'batch_size': batch_size if method == 'cellpose' else None,
            'num_workers': effective_num_workers,
            'wall_time': time.time() - start_time,
            'peak_ram_mb': peak_vms,
            'max_rss_mb': max_rss,
            'peak_vram_mb': peak_vram if peak_vram > 0 else None,
            'num_cells_total': 0,
            'num_acquisitions': 0,
            'success': False,
            'error': str(e)
        }
    
    finally:
        gc.collect()
        if _HAVE_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Scalability benchmark for segmentation (matches GUI exactly)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing image files (MCD files or OME-TIFF directories)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--nuclear-channels', type=str, required=True,
                       help='Comma-separated list of nuclear channel names')
    parser.add_argument('--cyto-channels', type=str, default=None,
                       help='Comma-separated list of cytoplasm channel names (optional, for cyto3 model)')
    parser.add_argument('--num-images', type=int, nargs='+', default=[50, 100, 200, 500],
                       help='Number of images to test (default: 50 100 200 500)')
    parser.add_argument('--repeats', type=int, default=3,
                       help='Number of times to repeat each configuration (default: 3)')
    parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW',
                       help='Channel format for OME-TIFF files (default: CHW)')
    parser.add_argument('--method', choices=['cellpose', 'cellsam', 'both'], default='both',
                       help='Segmentation method: cellpose, cellsam, or both (default: both)')
    parser.add_argument('--cellpose-model', choices=['cyto3', 'nuclei'], default='cyto3',
                       help='Cellpose model type (default: cyto3, only used with --method cellpose)')
    parser.add_argument('--diameter', type=int, default=10,
                       help='Cell diameter in pixels (optional, Cellpose will estimate if not provided, only used with --method cellpose)')
    parser.add_argument('--flow-threshold', type=float, default=0.4,
                       help='Flow threshold for Cellpose (default: 0.4, only used with --method cellpose)')
    parser.add_argument('--cellprob-threshold', type=float, default=0.0,
                       help='Cell probability threshold for Cellpose (default: 0.0, only used with --method cellpose)')
    parser.add_argument('--deepcell-api-key', type=str, default=None,
                       help='DeepCell API key for CellSAM (or set DEEPCELL_ACCESS_TOKEN env var, only used with --method cellsam)')
    parser.add_argument('--bbox-threshold', type=float, default=0.4,
                       help='Bbox threshold for CellSAM (default: 0.4, only used with --method cellsam)')
    parser.add_argument('--gpu-id', type=str, default='0',
                       help='GPU ID to use (default: "0", use "auto" for auto-detection, or "None" for CPU)')
    parser.add_argument('--denoise-settings', type=str, default=None,
                       help='JSON file or string with denoise settings (optional)')
    parser.add_argument('--num-workers', type=int, nargs='+', default=[None],
                       help='Number of workers for multiprocessing during loading/preprocessing (default: auto, uses cpu_count - 2). For Cellpose, batch_size is set equal to num_workers. Can specify multiple values to test scalability.')
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    if not images_dir.exists():
        print(f"Error: Images directory does not exist: {images_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse channels
    nuclear_channels = [ch.strip() for ch in args.nuclear_channels.split(',')]
    cyto_channels = None
    if args.cyto_channels:
        cyto_channels = [ch.strip() for ch in args.cyto_channels.split(',')]
    
    # Parse denoise settings if provided
    denoise_settings = None
    if args.denoise_settings:
        from openimc.core import parse_denoise_settings
        denoise_settings = parse_denoise_settings(args.denoise_settings)
    
    # Parse GPU ID
    # args.gpu_id is a string from argparse (default: '0')
    gpu_id = None
    if args.gpu_id and args.gpu_id.lower() != 'none':
        if args.gpu_id.lower() == 'auto':
            gpu_id = 'auto'
        else:
            # Try to convert to int, but keep as string if it's not numeric (e.g., 'mps')
            try:
                gpu_id = int(args.gpu_id)
            except (ValueError, TypeError):
                # Not a number, use as-is (e.g., 'mps' for Apple Silicon)
                gpu_id = args.gpu_id
    
    # Find image files
    print(f"Scanning for images in {images_dir}...")
    image_files = find_image_files(images_dir)
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {images_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files/directories")
    print(f"Nuclear channels: {nuclear_channels}")
    if cyto_channels:
        print(f"Cytoplasm channels: {cyto_channels}")
    print(f"Segmentation method(s): {args.method}")
    if args.method in ['cellpose', 'both']:
        print(f"Cellpose model: {args.cellpose_model}")
        print(f"  Note: batch_size will be set equal to num_workers for Cellpose")
    if args.method in ['cellsam', 'both']:
        print(f"CellSAM bbox_threshold: {args.bbox_threshold}")
        print(f"  CellSAM: use_wsi=False, gauge_cell_size=False (as requested)")
    worker_str = [str(w) if w is not None else 'auto' for w in args.num_workers]
    print(f"Num workers: {', '.join(worker_str)} (auto means cpu_count - 2)")
    
    # Determine which methods to run
    methods_to_run = []
    if args.method == 'both':
        methods_to_run = ['cellpose', 'cellsam']
    else:
        methods_to_run = [args.method]
    
    # Load existing results if CSV exists
    results_csv = output_dir / 'scalability_results.csv'
    existing_results = []
    if results_csv.exists():
        try:
            existing_df = pd.read_csv(results_csv)
            existing_results = existing_df.to_dict('records')
            print(f"Loaded {len(existing_results)} existing results from {results_csv}")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            existing_results = []
    
    # Create a set of completed runs to skip (successful runs only)
    # Format: (num_images, method, num_workers, repeat)
    completed_runs = set()
    for row in existing_results:
        if row.get('success', False):
            num_imgs = row.get('num_images')
            method = row.get('method')
            num_w = row.get('num_workers')
            repeat = row.get('repeat', 0)
            if num_imgs is not None and method and num_w is not None:
                completed_runs.add((num_imgs, method, num_w, repeat))
    
    print(f"Found {len(completed_runs)} completed runs to skip")
    
    # Run benchmarks
    results = []
    first_image = args.num_images[0]
    first_worker = args.num_workers[0] if args.num_workers[0] is not None else 'auto'
    first_method = methods_to_run[0]
    
    for method in methods_to_run:
        for num_images in args.num_images:
            for num_workers in args.num_workers:
                for repeat in range(args.repeats):
                    # Check if this run is already completed
                    effective_num_workers = num_workers if num_workers is not None else max(1, min(mp.cpu_count() - 2, num_images))
                    run_key = (num_images, method, effective_num_workers, repeat)
                    
                    if run_key in completed_runs:
                        print(f"Skipping already completed: {num_images} images, {method}, {effective_num_workers} workers, repeat {repeat}")
                        continue
                    
                    # Reset memory between each benchmark run
                    if repeat > 0 or (num_images != first_image or num_workers != first_worker or method != first_method):
                        print("\nResetting memory before next benchmark...")
                        reset_memory(gpu_id)
                    
                    result = run_segmentation_benchmark(
                        image_files, num_images, output_dir,
                        nuclear_channels, cyto_channels,
                        method=method,
                        cellpose_model=args.cellpose_model,
                        diameter=args.diameter,
                        flow_threshold=args.flow_threshold,
                        cellprob_threshold=args.cellprob_threshold,
                        gpu_id=gpu_id,
                        denoise_settings=denoise_settings,
                        channel_format=args.channel_format,
                        num_workers=num_workers,
                        repeat=repeat,
                        deepcell_api_key=args.deepcell_api_key,
                        bbox_threshold=args.bbox_threshold
                    )
                    result['repeat'] = repeat
                    results.append(result)
                
                # Reset memory after each run
                reset_memory(gpu_id)
    
    # Combine existing and new results
    if results:
        # Convert new results to DataFrame and combine with existing
        new_results_df = pd.DataFrame(results)
        if existing_results:
            existing_df = pd.DataFrame(existing_results)
            # Remove old failed runs for the same configurations we just ran
            # Keep only successful runs or runs we didn't re-run
            new_configs = set()
            for row in results:
                num_imgs = row.get('num_images')
                method = row.get('method')
                num_w = row.get('num_workers')
                repeat = row.get('repeat', 0)
                if num_imgs is not None and method and num_w is not None:
                    new_configs.add((num_imgs, method, num_w, repeat))
            
            # Filter existing: keep if successful OR not in new_configs
            def should_keep(row):
                num_imgs = row.get('num_images')
                method = row.get('method')
                num_w = row.get('num_workers')
                repeat = row.get('repeat', 0)
                if num_imgs is None or not method or num_w is None:
                    return True  # Keep malformed rows
                key = (num_imgs, method, num_w, repeat)
                # Keep if successful OR not being re-run
                return row.get('success', False) or key not in new_configs
            
            existing_df = existing_df[existing_df.apply(should_keep, axis=1)]
            # Combine
            results_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        else:
            results_df = new_results_df
        
        # Save combined results
        results_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to {results_csv} ({len(results_df)} total rows, {len(results)} new)")
    else:
        print("\nNo new results to save (all runs were already completed)")
        if existing_results:
            results_df = pd.DataFrame(existing_results)
        else:
            results_df = pd.DataFrame()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    if 'success' in results_df.columns:
        successful = results_df[results_df['success'] == True]
        if len(successful) > 0:
            agg_dict = {
                'wall_time': ['mean', 'std'],
                'peak_ram_mb': ['mean', 'std'],
                'max_rss_mb': ['mean', 'std'],
                'num_cells_total': ['mean', 'std']
            }
            if 'peak_vram_mb' in successful.columns:
                agg_dict['peak_vram_mb'] = ['mean', 'std']
            groupby_cols = ['num_images', 'method']
            if 'batch_size' in successful.columns:
                groupby_cols.append('batch_size')
            if 'num_workers' in successful.columns:
                groupby_cols.append('num_workers')
            print(successful.groupby(groupby_cols).agg(agg_dict))
        else:
            print("No successful runs!")
    
    print(f"\nFull results saved to: {results_csv}")
    print(f"To generate plots, run: python generate_scalability_plots.py {results_csv}")


if __name__ == '__main__':
    main()
