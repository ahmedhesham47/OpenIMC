# SPDX-License-Identifier: GPL-3.0-or-later
#
# OpenIMC – Interactive analysis toolkit for IMC data
#
# Copyright (C) 2025 University of Southern California
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Dynamic mask manager for handling large datasets with on-demand mask loading.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

# Try to import tifffile
try:
    import tifffile
    _HAVE_TIFFFILE = True
except ImportError:
    _HAVE_TIFFFILE = False


class DynamicMaskManager:
    """
    Manages segmentation masks with dynamic loading for large datasets.
    
    For datasets with >50 ROIs, masks are stored on disk and loaded on-demand
    to avoid memory buildup. For smaller datasets, masks can be kept in memory.
    """
    
    def __init__(self, masks_directory: Optional[str] = None, force_disk_storage: bool = False):
        """
        Initialize the mask manager.
        
        Args:
            masks_directory: Directory where masks are stored on disk
            force_disk_storage: If True, always store masks on disk even for small datasets
        """
        self.masks_directory = masks_directory
        self.force_disk_storage = force_disk_storage
        
        # In-memory masks (for small datasets or recently accessed masks)
        self._in_memory_masks: Dict[str, np.ndarray] = {}
        
        # Mask file paths (for on-disk masks)
        self._mask_file_paths: Dict[str, str] = {}
        
        # Track which masks are currently in memory (for cache management)
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._max_cache_size = 10  # Keep up to 10 masks in memory cache
    
    def set_masks_directory(self, directory: str):
        """Set the directory where masks are stored."""
        self.masks_directory = directory
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def has_mask(self, acq_id: str) -> bool:
        """Check if a mask exists for the given acquisition ID."""
        return (acq_id in self._in_memory_masks or 
                acq_id in self._mask_file_paths or
                self._get_mask_file_path(acq_id) is not None)
    
    def get_mask(self, acq_id: str, acq_info=None) -> Optional[np.ndarray]:
        """
        Get mask for an acquisition, loading from disk if necessary.
        
        Args:
            acq_id: Acquisition ID
            acq_info: Optional AcquisitionInfo object for finding mask files
        
        Returns:
            Mask array or None if not found
        """
        # If forcing disk storage, don't use in-memory cache
        if not self.force_disk_storage:
            # First check in-memory cache
            if acq_id in self._memory_cache:
                return self._memory_cache[acq_id]
            
            # Check in-memory masks
            if acq_id in self._in_memory_masks:
                mask = self._in_memory_masks[acq_id]
                # Add to cache
                self._add_to_cache(acq_id, mask)
                return mask
        
        # Check if we have a file path
        mask_path = self._mask_file_paths.get(acq_id)
        if not mask_path and self.masks_directory:
            # Try to find mask file
            mask_path = self._get_mask_file_path(acq_id, acq_info)
        
        if mask_path and os.path.exists(mask_path):
            # Load from disk
            try:
                if mask_path.endswith('.npy'):
                    mask = np.load(mask_path)
                else:
                    if _HAVE_TIFFFILE:
                        mask = tifffile.imread(mask_path)
                    else:
                        from PIL import Image
                        mask = np.array(Image.open(mask_path))
                
                # Only add to cache if not forcing disk storage
                if not self.force_disk_storage:
                    self._add_to_cache(acq_id, mask)
                return mask
            except Exception as e:
                print(f"Error loading mask from {mask_path}: {e}")
                return None
        
        return None
    
    def set_mask(self, acq_id: str, mask: np.ndarray, save_to_disk: bool = False, 
                 acq_info=None, masks_directory: Optional[str] = None):
        """
        Set mask for an acquisition.
        
        Args:
            acq_id: Acquisition ID
            mask: Mask array
            save_to_disk: If True, save to disk immediately
            acq_info: Optional AcquisitionInfo for generating filename
            masks_directory: Optional directory to save masks (overrides self.masks_directory)
        """
        try:
            from openimc.core import _log_memory_debug
        except ImportError:
            def _log_memory_debug(msg, obj=None, obj_name=None):
                if obj is not None and obj_name and isinstance(obj, np.ndarray):
                    obj_size_mb = obj.nbytes / 1024 / 1024
                    print(f"[MASK_MGR] {msg} | {obj_name}: {obj.shape} {obj.dtype} ({obj_size_mb:.2f} MB)")
                else:
                    print(f"[MASK_MGR] {msg}")
        
        _log_memory_debug(f"[MASK_MGR] set_mask called for {acq_id}, save_to_disk={save_to_disk}, force_disk_storage={self.force_disk_storage}", mask, "input_mask")
        save_dir = masks_directory or self.masks_directory
        
        # If we're forcing disk storage or explicitly requested, save to disk
        if self.force_disk_storage or save_to_disk:
            if save_dir:
                _log_memory_debug(f"[MASK_MGR] Saving mask to disk for {acq_id}")
                mask_path = self._save_mask_to_disk(acq_id, mask, save_dir, acq_info)
                if mask_path:
                    self._mask_file_paths[acq_id] = mask_path
                    _log_memory_debug(f"[MASK_MGR] Mask saved to {mask_path} for {acq_id}")
                    # Don't keep in memory if we're forcing disk storage
                    if self.force_disk_storage:
                        # Explicitly clear any existing in-memory references
                        self._in_memory_masks.pop(acq_id, None)
                        self._memory_cache.pop(acq_id, None)
                        _log_memory_debug(f"[MASK_MGR] Mask NOT kept in memory (force_disk_storage=True) for {acq_id}")
                        return
            else:
                # Can't save to disk without directory
                print(f"Warning: Cannot save mask to disk - no masks directory set")
        
        # Store in memory (for small datasets or caching)
        # Only if not forcing disk storage
        if not self.force_disk_storage:
            self._in_memory_masks[acq_id] = mask
            self._add_to_cache(acq_id, mask)
            _log_memory_debug(f"[MASK_MGR] Mask kept in memory for {acq_id}", mask, "stored_mask")
    
    def remove_mask(self, acq_id: str):
        """Remove mask from memory and cache (does not delete disk files)."""
        self._in_memory_masks.pop(acq_id, None)
        self._memory_cache.pop(acq_id, None)
        # Keep file path in case we need to reload
    
    def clear_memory_cache(self):
        """Clear the memory cache (keeps file paths)."""
        self._memory_cache.clear()
        # Optionally clear in-memory masks too if forcing disk storage
        if self.force_disk_storage:
            self._in_memory_masks.clear()
    
    def get_all_mask_ids(self) -> list:
        """Get list of all acquisition IDs that have masks."""
        ids = set(self._in_memory_masks.keys())
        ids.update(self._mask_file_paths.keys())
        return list(ids)
    
    def get_mask_file_path(self, acq_id: str, acq_info=None) -> Optional[str]:
        """
        Get the file path for a mask if it exists on disk.
        
        Args:
            acq_id: Acquisition ID
            acq_info: Optional AcquisitionInfo object for finding mask files
        
        Returns:
            File path to mask if it exists on disk, None otherwise
        """
        # Check if we have a stored file path
        if acq_id in self._mask_file_paths:
            path = self._mask_file_paths[acq_id]
            if os.path.exists(path):
                return path
        
        # Try to find the file path
        return self._get_mask_file_path(acq_id, acq_info)
    
    def _get_mask_file_path(self, acq_id: str, acq_info=None) -> Optional[str]:
        """Get the file path for a mask, trying various naming conventions."""
        if not self.masks_directory:
            return None
        
        if acq_info:
            # Try to match using acquisition info
            source_basename = None
            if hasattr(acq_info, 'source_file') and acq_info.source_file:
                source_basename = os.path.splitext(os.path.basename(acq_info.source_file))[0]
                source_basename = self._sanitize_filename(source_basename)
            
            # Try different possible filenames
            possible_filenames = []
            
            if acq_info.well and source_basename:
                safe_well = self._sanitize_filename(acq_info.well)
                possible_filenames.append(f"{source_basename}_{safe_well}_segmentation.tiff")
                possible_filenames.append(f"{source_basename}_{safe_well}_segmentation.tif")
                possible_filenames.append(f"{source_basename}_{safe_well}_segmentation_masks.tiff")
                possible_filenames.append(f"{source_basename}_{safe_well}_segmentation_masks.tif")
            
            if source_basename:
                safe_name = self._sanitize_filename(acq_info.name)
                possible_filenames.append(f"{source_basename}_{safe_name}_segmentation.tiff")
                possible_filenames.append(f"{source_basename}_{safe_name}_segmentation.tif")
                possible_filenames.append(f"{source_basename}_{safe_name}_segmentation_masks.tiff")
                possible_filenames.append(f"{source_basename}_{safe_name}_segmentation_masks.tif")
            
            # Try without source prefix
            if acq_info.well:
                safe_well = self._sanitize_filename(acq_info.well)
                possible_filenames.append(f"{safe_well}_segmentation.tiff")
                possible_filenames.append(f"{safe_well}_segmentation.tif")
            
            safe_name = self._sanitize_filename(acq_info.name)
            possible_filenames.append(f"{safe_name}_segmentation.tiff")
            possible_filenames.append(f"{safe_name}_segmentation.tif")
            
            # Try with acquisition ID
            possible_filenames.append(f"{acq_id}_segmentation.tiff")
            possible_filenames.append(f"{acq_id}_segmentation.tif")
            possible_filenames.append(f"{acq_id}_segmentation.npy")
            
            for filename in possible_filenames:
                filepath = os.path.join(self.masks_directory, filename)
                if os.path.exists(filepath):
                    return filepath
        
        # Fallback: try direct ID-based naming
        for ext in ['.tiff', '.tif', '.npy']:
            filepath = os.path.join(self.masks_directory, f"{acq_id}_segmentation{ext}")
            if os.path.exists(filepath):
                return filepath
        
        return None
    
    def _save_mask_to_disk(self, acq_id: str, mask: np.ndarray, masks_directory: str, 
                          acq_info=None) -> Optional[str]:
        """Save mask to disk and return the file path."""
        try:
            from openimc.core import _log_memory_debug
        except ImportError:
            def _log_memory_debug(msg, obj=None, obj_name=None):
                if obj is not None and obj_name and isinstance(obj, np.ndarray):
                    obj_size_mb = obj.nbytes / 1024 / 1024
                    print(f"[MASK_MGR] {msg} | {obj_name}: {obj.shape} {obj.dtype} ({obj_size_mb:.2f} MB)")
                else:
                    print(f"[MASK_MGR] {msg}")
        
        _log_memory_debug(f"[MASK_MGR] _save_mask_to_disk called for {acq_id}", mask, "mask_to_save")
        try:
            if not os.path.exists(masks_directory):
                os.makedirs(masks_directory, exist_ok=True)
            
            # Generate filename
            if acq_info:
                source_basename = None
                if hasattr(acq_info, 'source_file') and acq_info.source_file:
                    source_basename = os.path.splitext(os.path.basename(acq_info.source_file))[0]
                    source_basename = self._sanitize_filename(source_basename)
                
                if acq_info.well and source_basename:
                    safe_well = self._sanitize_filename(acq_info.well)
                    filename = f"{source_basename}_{safe_well}_segmentation_masks.tif"
                elif source_basename:
                    safe_name = self._sanitize_filename(acq_info.name)
                    filename = f"{source_basename}_{safe_name}_segmentation_masks.tif"
                elif acq_info.well:
                    safe_well = self._sanitize_filename(acq_info.well)
                    filename = f"{safe_well}_segmentation_masks.tif"
                else:
                    safe_name = self._sanitize_filename(acq_info.name)
                    filename = f"{safe_name}_segmentation_masks.tif"
            else:
                filename = f"{acq_id}_segmentation_masks.tif"
            
            filepath = os.path.join(masks_directory, filename)
            
            # Save mask
            _log_memory_debug(f"[MASK_MGR] Writing mask to {filepath} for {acq_id}")
            if _HAVE_TIFFFILE:
                tifffile.imwrite(filepath, mask.astype(np.uint16), compression='lzw')
            else:
                from PIL import Image
                Image.fromarray(mask.astype(np.uint16)).save(filepath)
            _log_memory_debug(f"[MASK_MGR] Mask written to disk successfully for {acq_id}")
            
            return filepath
        except Exception as e:
            print(f"Error saving mask to disk: {e}")
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing invalid characters."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename
    
    def _add_to_cache(self, acq_id: str, mask: np.ndarray):
        """Add mask to memory cache, evicting oldest if cache is full."""
        # Don't cache if forcing disk storage
        if self.force_disk_storage:
            return
        
        # Simple LRU: remove oldest entry if cache is full
        if len(self._memory_cache) >= self._max_cache_size:
            # Remove first (oldest) entry
            first_key = next(iter(self._memory_cache))
            del self._memory_cache[first_key]
        
        self._memory_cache[acq_id] = mask

