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
State Manager for OpenIMC Application

This module handles saving and loading the complete application state,
including images, masks, features, and all analysis module states.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterable
import numpy as np
import pandas as pd
import tifffile

from openimc.data.mcd_loader import AcquisitionInfo


class StateManager:
    """
    Manages saving and loading of complete application state.
    
    State is saved to a folder structure:
    - state.json: Main state metadata
    - images/: Saved images (if any)
    - masks/: Segmentation masks
    - features/: Feature dataframes
    - analysis/: Analysis module states (QC, clustering, spatial, etc.)
    
    The state manager ensures that all data structures required by each analysis
    module are properly saved and can be restored, including:
    - Clustering: cluster labels, embeddings, annotations, filtered features
    - Pixel Correlation: correlation results, aggregated results
    - QC Analysis: QC results (pixel and cell level), aggregated results
    - Spatial Analysis: edge dataframes, enrichment results, distance results
    """
    
    STATE_VERSION = "1.1"  # Incremented for improved state management
    TEXT_ENCODING = "utf-8"
    # Heuristics for keeping analysis-state JSON reasonably sized
    _INLINE_NDARRAY_MAX_ELEMENTS = 10_000
    _INLINE_DF_MAX_CELLS = 50_000  # rows * cols; above this we save to CSV
    
    def __init__(self):
        self.state_version = self.STATE_VERSION
    
    def save_state(
        self,
        state_path: Union[str, Path],
        main_window_state: Dict[str, Any],
        overwrite: bool = False
    ) -> bool:
        """
        Save complete application state to a folder.
        
        Args:
            state_path: Path to folder where state will be saved
            main_window_state: Dictionary containing all state from MainWindow
            overwrite: If True, overwrite existing state folder
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state_path = Path(state_path)
            
            # Check if folder exists
            if state_path.exists():
                if overwrite:
                    shutil.rmtree(state_path)
                else:
                    return False
            
            # Create state folder structure
            state_path.mkdir(parents=True, exist_ok=True)
            images_dir = state_path / "images"
            masks_dir = state_path / "masks"
            features_dir = state_path / "features"
            analysis_dir = state_path / "analysis"
            
            images_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)
            features_dir.mkdir(exist_ok=True)
            analysis_dir.mkdir(exist_ok=True)
            
            # Extract state components
            state_data = {
                "version": self.state_version,
                "main_window": {},
                "analysis_modules": {}
            }
            
            # Save main window state (file paths, acquisition info, etc.)
            main_state = main_window_state.get("main_state", {})
            state_data["main_window"] = main_state
            
            # Save images if they exist
            images_state = main_window_state.get("images", {})
            if images_state:
                state_data["main_window"]["images"] = self._save_images(
                    images_state, images_dir
                )
            
            # Save masks
            masks_state = main_window_state.get("masks", {})
            acquisitions_info = main_window_state.get("acquisitions_info", {})
            if masks_state:
                state_data["main_window"]["masks"] = self._save_masks(
                    masks_state, masks_dir, acquisitions_info
                )
            
            # Copy .mcd files to images folder if they exist
            source_files = main_window_state.get("source_files", [])
            if source_files:
                self._copy_source_files(source_files, images_dir)
                state_data["main_window"]["source_files"] = [
                    str(Path(f).name) for f in source_files
                ]
            
            # Save features
            features_state = main_window_state.get("features", {})
            if features_state:
                state_data["main_window"]["features"] = self._save_features(
                    features_state, features_dir
                )
            
            # Save analysis module states
            analysis_state = main_window_state.get("analysis", {})
            if analysis_state:
                state_data["analysis_modules"] = self._save_analysis_states(
                    analysis_state, analysis_dir
                )
            
            # Save main state JSON
            state_json_path = state_path / "state.json"
            with open(state_json_path, 'w', encoding=self.TEXT_ENCODING) as f:
                json.dump(state_data, f, indent=2, default=self._json_serializer)
            
            # Generate README for Zenodo submission
            self._generate_readme(state_path, state_data, main_window_state)
            
            # Generate manifest file
            self._generate_manifest(state_path)
            
            return True
            
        except Exception as e:
            print(f"Error saving state: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_state(
        self,
        state_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """
        Load complete application state from a folder.
        
        Args:
            state_path: Path to folder containing saved state
            
        Returns:
            Dictionary containing loaded state, or None if failed
        """
        try:
            state_path = Path(state_path)
            
            if not state_path.exists():
                return None
            
            # Load main state JSON
            state_json_path = state_path / "state.json"
            if not state_json_path.exists():
                return None
            
            with open(state_json_path, 'r', encoding=self.TEXT_ENCODING) as f:
                state_data = json.load(f)
            
            # Check version compatibility
            version = state_data.get("version", "0.0")
            if version != self.state_version:
                print(f"Warning: State version {version} may not be compatible with current version {self.state_version}")
            
            # Load components
            main_window_data = state_data.get("main_window", {})
            loaded_state = {
                "main_window": main_window_data,
                "images": {},
                "masks": {},
                "features": {},
                "analysis": {}
            }
            
            # Store source files info in main_window for easy access
            if "source_files" in main_window_data:
                loaded_state["main_window"]["source_files"] = main_window_data["source_files"]
            
            # Load images
            images_info = state_data.get("main_window", {}).get("images", {})
            if images_info:
                images_dir = state_path / "images"
                loaded_state["images"] = self._load_images(images_info, images_dir)
            
            # Load masks
            masks_info = state_data.get("main_window", {}).get("masks", {})
            if masks_info:
                masks_dir = state_path / "masks"
                loaded_state["masks"] = self._load_masks(masks_info, masks_dir)
            
            # Load features
            features_info = state_data.get("main_window", {}).get("features", {})
            if features_info:
                features_dir = state_path / "features"
                loaded_state["features"] = self._load_features(features_info, features_dir)
            
            # Load analysis states
            analysis_info = state_data.get("analysis_modules", {})
            if analysis_info:
                analysis_dir = state_path / "analysis"
                loaded_state["analysis"] = self._load_analysis_states(analysis_info, analysis_dir)
            
            return loaded_state
            
        except Exception as e:
            print(f"Error loading state: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_images(
        self,
        images_state: Dict[str, Any],
        images_dir: Path
    ) -> Dict[str, str]:
        """
        Save images to disk.
        
        Args:
            images_state: Dictionary mapping acquisition IDs to image arrays or paths
            images_dir: Directory to save images
            
        Returns:
            Dictionary mapping acquisition IDs to saved image file paths
        """
        saved_images = {}
        
        if not images_state:
            return saved_images
        
        for acq_id, image_data in images_state.items():
            try:
                if isinstance(image_data, dict):
                    # Multi-channel image data
                    for channel, img in image_data.items():
                        if isinstance(img, np.ndarray):
                            img_path = images_dir / f"{acq_id}_{channel}.tif"
                            tifffile.imwrite(str(img_path), img)
                            if acq_id not in saved_images:
                                saved_images[acq_id] = {}
                            # Store paths relative to the state folder for portability (renaming/moving the state folder)
                            saved_images[acq_id][channel] = str(img_path.relative_to(images_dir.parent))
                elif isinstance(image_data, np.ndarray):
                    # Single image array
                    img_path = images_dir / f"{acq_id}.tif"
                    tifffile.imwrite(str(img_path), image_data)
                    saved_images[acq_id] = str(img_path.relative_to(images_dir.parent))
                elif isinstance(image_data, str):
                    # Already a file path - just record it
                    saved_images[acq_id] = image_data
            except Exception as e:
                print(f"Warning: Could not save image for {acq_id}: {e}")
                continue
        
        return saved_images
    
    def _load_images(
        self,
        images_info: Dict[str, Any],
        images_dir: Path
    ) -> Dict[str, Any]:
        """
        Load images from disk.
        
        Args:
            images_info: Dictionary mapping acquisition IDs to image file paths
            images_dir: Directory containing images
            
        Returns:
            Dictionary mapping acquisition IDs to loaded image arrays
        """
        loaded_images = {}
        
        if not images_info:
            return loaded_images
        
        for acq_id, img_path in images_info.items():
            try:
                if isinstance(img_path, dict):
                    # Multi-channel
                    loaded_images[acq_id] = {}
                    for channel, path in img_path.items():
                        full_path = images_dir.parent / path
                        if full_path.exists():
                            loaded_images[acq_id][channel] = tifffile.imread(str(full_path))
                elif isinstance(img_path, str):
                    # Single image
                    full_path = images_dir.parent / img_path
                    if full_path.exists():
                        loaded_images[acq_id] = tifffile.imread(str(full_path))
            except Exception as e:
                print(f"Warning: Could not load image for {acq_id}: {e}")
                continue
        
        return loaded_images
    
    def _save_masks(
        self,
        masks_state: Dict[str, Any],
        masks_dir: Path,
        acquisitions_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save masks to disk with proper naming convention.
        
        Args:
            masks_state: Dictionary mapping acquisition IDs to mask arrays or paths
            masks_dir: Directory to save masks
            acquisitions_info: Optional dictionary mapping acq_id to AcquisitionInfo for proper naming
            
        Returns:
            Dictionary mapping acquisition IDs to saved mask file paths
        """
        saved_masks = {}
        
        def sanitize_filename(filename: str) -> str:
            """Sanitize filename by removing invalid characters."""
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                filename = filename.replace(char, '_')
            return filename
        
        for acq_id, mask_data in masks_state.items():
            try:
                # Get acquisition info for proper naming
                acq_info = None
                if acquisitions_info:
                    acq_info = acquisitions_info.get(acq_id)
                
                # Generate proper filename
                if acq_info:
                    # Use well name if available, otherwise use acquisition name
                    if hasattr(acq_info, 'well') and acq_info.well:
                        label_for_filename = acq_info.well
                    elif hasattr(acq_info, 'name'):
                        label_for_filename = acq_info.name
                    else:
                        label_for_filename = acq_id
                    
                    safe_label = sanitize_filename(label_for_filename)
                    
                    # Include source file name in filename to ensure uniqueness
                    if hasattr(acq_info, 'source_file') and acq_info.source_file:
                        source_basename = os.path.splitext(os.path.basename(acq_info.source_file))[0]
                        safe_source = sanitize_filename(source_basename)
                        filename = f"{safe_source}_{safe_label}_segmentation_masks.tif"
                    else:
                        filename = f"{safe_label}_segmentation_masks.tif"
                else:
                    # Fallback to acq_id if no info available
                    filename = f"{sanitize_filename(acq_id)}_segmentation_masks.tif"
                
                mask_path = masks_dir / filename
                
                if isinstance(mask_data, np.ndarray):
                    # Save mask array
                    tifffile.imwrite(str(mask_path), mask_data.astype(np.uint16), compression='lzw')
                    saved_masks[acq_id] = str(mask_path.relative_to(masks_dir.parent))
                elif isinstance(mask_data, str):
                    # Already a file path - copy it
                    if os.path.exists(mask_data):
                        shutil.copy2(mask_data, mask_path)
                        saved_masks[acq_id] = str(mask_path.relative_to(masks_dir.parent))
                    else:
                        saved_masks[acq_id] = mask_data
            except Exception as e:
                print(f"Warning: Could not save mask for {acq_id}: {e}")
                continue
        
        return saved_masks
    
    def _load_masks(
        self,
        masks_info: Dict[str, str],
        masks_dir: Path
    ) -> Dict[str, np.ndarray]:
        """
        Load masks from disk.
        
        Args:
            masks_info: Dictionary mapping acquisition IDs to mask file paths
            masks_dir: Directory containing masks
            
        Returns:
            Dictionary mapping acquisition IDs to loaded mask arrays
        """
        loaded_masks = {}
        
        for acq_id, mask_path in masks_info.items():
            full_path = masks_dir.parent / mask_path
            if full_path.exists():
                loaded_masks[acq_id] = tifffile.imread(str(full_path))
        
        return loaded_masks
    
    def _copy_source_files(self, source_files: List[str], images_dir: Path):
        """
        Copy .mcd files (or other source files) to images directory.
        
        Args:
            source_files: List of source file paths
            images_dir: Directory to copy files to
        """
        for source_file in source_files:
            if not source_file or not os.path.exists(source_file):
                continue
            
            try:
                # Get just the filename
                filename = os.path.basename(source_file)
                dest_path = images_dir / filename
                
                # Copy the file
                shutil.copy2(source_file, dest_path)
                print(f"Copied source file: {filename}")
            except Exception as e:
                print(f"Warning: Could not copy source file {source_file}: {e}")
                continue
    
    def _save_features(
        self,
        features_state: Dict[str, pd.DataFrame],
        features_dir: Path
    ) -> Dict[str, str]:
        """
        Save feature dataframes to disk with proper naming.
        
        Args:
            features_state: Dictionary mapping feature names to DataFrames
            features_dir: Directory to save features
            
        Returns:
            Dictionary mapping feature names to saved file paths
        """
        saved_features = {}
        
        # Standard feature file names
        feature_names = {
            "original": "features.csv",
            "batch_corrected": "features_batch_corrected.csv"
        }
        
        for name, df in features_state.items():
            if df is not None and not df.empty:
                # Use standard name if available, otherwise use the provided name
                filename = feature_names.get(name, f"{name}_features.csv")
                feature_path = features_dir / filename
                df.to_csv(feature_path, index=False, encoding=self.TEXT_ENCODING)
                saved_features[name] = str(feature_path.relative_to(features_dir.parent))
        
        return saved_features
    
    def _load_features(
        self,
        features_info: Dict[str, str],
        features_dir: Path
    ) -> Dict[str, pd.DataFrame]:
        """
        Load feature dataframes from disk.
        
        Args:
            features_info: Dictionary mapping feature names to file paths
            features_dir: Directory containing features
            
        Returns:
            Dictionary mapping feature names to loaded DataFrames
        """
        loaded_features = {}
        
        for name, feature_path in features_info.items():
            full_path = features_dir.parent / feature_path
            if full_path.exists():
                loaded_features[name] = pd.read_csv(full_path, encoding=self.TEXT_ENCODING)
        
        return loaded_features
    
    def _save_analysis_states(
        self,
        analysis_state: Dict[str, Any],
        analysis_dir: Path
    ) -> Dict[str, Any]:
        """
        Save analysis module states to disk.
        
        Args:
            analysis_state: Dictionary mapping module names to their states
            analysis_dir: Directory to save analysis states
            
        Returns:
            Dictionary mapping module names to saved file paths
        """
        saved_analysis = {}
        
        for module_name, module_state in analysis_state.items():
            if module_state:
                # Save as JSON (with special handling for DataFrames)
                analysis_path = analysis_dir / f"{module_name}.json"
                
                # Convert state to JSON-serializable format
                json_state = self._prepare_analysis_state_for_json(
                    module_state,
                    analysis_dir=analysis_dir,
                    module_name=module_name,
                    path_parts=(),
                )
                
                with open(analysis_path, 'w', encoding=self.TEXT_ENCODING) as f:
                    json.dump(json_state, f, indent=2, default=self._json_serializer)
                
                saved_analysis[module_name] = str(analysis_path.relative_to(analysis_dir.parent))
        
        return saved_analysis
    
    def _load_analysis_states(
        self,
        analysis_info: Dict[str, str],
        analysis_dir: Path
    ) -> Dict[str, Any]:
        """
        Load analysis module states from disk.
        
        Args:
            analysis_info: Dictionary mapping module names to file paths
            analysis_dir: Directory containing analysis states
            
        Returns:
            Dictionary mapping module names to loaded states
        """
        loaded_analysis = {}
        
        for module_name, analysis_path in analysis_info.items():
            full_path = analysis_dir.parent / analysis_path
            if full_path.exists():
                with open(full_path, 'r', encoding=self.TEXT_ENCODING) as f:
                    state = json.load(f)
                    # Restore DataFrames and other special types
                    loaded_analysis[module_name] = self._restore_analysis_state_from_json(state, analysis_dir)
        
        return loaded_analysis
    
    def _prepare_analysis_state_for_json(
        self,
        state: Any,
        *,
        analysis_dir: Optional[Path] = None,
        module_name: Optional[str] = None,
        path_parts: Iterable[str] = (),
    ) -> Any:
        """
        Prepare analysis state for JSON serialization.
        Handles DataFrames, Series, numpy arrays, etc.
        """
        if isinstance(state, dict):
            return {
                k: self._prepare_analysis_state_for_json(
                    v,
                    analysis_dir=analysis_dir,
                    module_name=module_name,
                    path_parts=tuple(path_parts) + (str(k),),
                )
                for k, v in state.items()
            }
        elif isinstance(state, list):
            return [
                self._prepare_analysis_state_for_json(
                    item,
                    analysis_dir=analysis_dir,
                    module_name=module_name,
                    path_parts=tuple(path_parts) + (str(i),),
                )
                for i, item in enumerate(state)
            ]
        elif isinstance(state, pd.DataFrame):
            # Inline small DataFrames; save larger ones to disk for portability/perf.
            n_cells = int(state.shape[0]) * int(state.shape[1])
            if analysis_dir is not None and module_name is not None and n_cells > self._INLINE_DF_MAX_CELLS:
                blob_dir = analysis_dir / "_blobs"
                blob_dir.mkdir(parents=True, exist_ok=True)
                safe_name = self._make_blob_name(module_name, path_parts, ext="csv")
                df_path = blob_dir / safe_name
                state.to_csv(df_path, index=False, encoding=self.TEXT_ENCODING)
                return {
                    "__type__": "DataFrame_file",
                    "__path__": str(df_path.relative_to(analysis_dir.parent)),
                }
            return {"__type__": "DataFrame", "__data__": state.to_dict('records')}
        elif isinstance(state, pd.Series):
            # Convert Series to dict format with index and values
            return {
                "__type__": "Series",
                "__data__": state.to_dict(),
                "__index__": state.index.tolist() if hasattr(state.index, 'tolist') else list(state.index),
                "__name__": state.name if state.name is not None else None
            }
        elif isinstance(state, pd.Index):
            # Convert Index to list
            return {"__type__": "Index", "__data__": state.tolist() if hasattr(state, 'tolist') else list(state)}
        elif isinstance(state, np.ndarray):
            # Inline small arrays; save larger ones to .npy under analysis/_blobs.
            if state.size <= self._INLINE_NDARRAY_MAX_ELEMENTS:
                return {
                    "__type__": "ndarray",
                    "__data__": state.tolist(),
                    "__shape__": list(state.shape),
                    "__dtype__": str(state.dtype),
                }
            if analysis_dir is not None and module_name is not None:
                blob_dir = analysis_dir / "_blobs"
                blob_dir.mkdir(parents=True, exist_ok=True)
                safe_name = self._make_blob_name(module_name, path_parts, ext="npy")
                arr_path = blob_dir / safe_name
                np.save(str(arr_path), state)
                return {
                    "__type__": "ndarray_file",
                    "__path__": str(arr_path.relative_to(analysis_dir.parent)),
                    "__shape__": list(state.shape),
                    "__dtype__": str(state.dtype),
                }
            # Fallback (shouldn't happen in normal save flow)
            return {
                "__type__": "ndarray",
                "__data__": state.tolist(),
                "__shape__": list(state.shape),
                "__dtype__": str(state.dtype),
            }
        elif isinstance(state, (np.integer, np.floating)):
            return state.item()
        elif isinstance(state, (set, frozenset)):
            # Convert sets to lists for JSON serialization
            return {"__type__": "set", "__data__": list(state)}
        elif isinstance(state, tuple):
            # Convert tuples to lists (will be restored as lists, which is usually fine)
            return list(state)
        else:
            return state

    def _make_blob_name(self, module_name: str, path_parts: Iterable[str], *, ext: str) -> str:
        """Create a deterministic, filesystem-safe blob filename for an analysis-state leaf."""
        def _sanitize(s: str) -> str:
            s = str(s)
            # Keep it readable; collapse anything risky to underscores.
            return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)[:80]

        parts = [_sanitize(module_name)] + [_sanitize(p) for p in path_parts if p is not None]
        stem = "__".join([p for p in parts if p])
        if not stem:
            stem = "analysis_state"
        return f"{stem}.{ext}"
    
    def _restore_analysis_state_from_json(self, state: Any, analysis_dir: Path) -> Any:
        """
        Restore analysis state from JSON, handling special types.
        """
        if isinstance(state, dict):
            if "__type__" in state:
                if state["__type__"] == "DataFrame":
                    return pd.DataFrame(state["__data__"])
                elif state["__type__"] == "DataFrame_file":
                    # Load DataFrame from CSV stored relative to state folder
                    if state.get("__path__"):
                        full_path = analysis_dir.parent / state["__path__"]
                        if full_path.exists():
                            return pd.read_csv(full_path, encoding=self.TEXT_ENCODING)
                    return pd.DataFrame()
                elif state["__type__"] == "Series":
                    # Restore Series from dict format
                    series_data = state.get("__data__", {})
                    series_index = state.get("__index__", None)
                    series_name = state.get("__name__", None)
                    if series_index is not None:
                        restored_series = pd.Series(series_data, index=series_index, name=series_name)
                    else:
                        restored_series = pd.Series(series_data, name=series_name)
                    return restored_series
                elif state["__type__"] == "Index":
                    # Restore Index from list
                    return pd.Index(state.get("__data__", []))
                elif state["__type__"] == "set":
                    # Restore set from list
                    return set(state.get("__data__", []))
                elif state["__type__"] == "ndarray":
                    return np.array(state["__data__"], dtype=state.get("__dtype__", float)).reshape(state["__shape__"])
                elif state["__type__"] == "ndarray_file":
                    # Load from file if path is provided
                    if state.get("__path__"):
                        full_path = analysis_dir.parent / state["__path__"]
                        if full_path.exists():
                            return np.load(str(full_path), allow_pickle=False)
                    return None
            else:
                return {k: self._restore_analysis_state_from_json(v, analysis_dir) for k, v in state.items()}
        elif isinstance(state, list):
            return [self._restore_analysis_state_from_json(item, analysis_dir) for item in state]
        else:
            return state
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            # Convert Series to dict format
            return {
                "__type__": "Series",
                "__data__": obj.to_dict(),
                "__index__": obj.index.tolist() if hasattr(obj.index, 'tolist') else list(obj.index),
                "__name__": obj.name if obj.name is not None else None
            }
        elif isinstance(obj, pd.Index):
            # Convert Index to list
            return {"__type__": "Index", "__data__": obj.tolist() if hasattr(obj, 'tolist') else list(obj)}
        elif isinstance(obj, (set, frozenset)):
            # Convert sets to lists for JSON serialization
            return {"__type__": "set", "__data__": list(obj)}
        elif isinstance(obj, tuple):
            # Convert tuples to lists (will be restored as lists)
            return list(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _generate_readme(self, state_path: Path, state_data: Dict[str, Any], main_window_state: Dict[str, Any]):
        """Generate README.md file for Zenodo submission."""
        readme_path = state_path / "README.md"
        
        # Get metadata from state
        main_state = state_data.get("main_window", {})
        acquisitions = main_state.get("acquisitions", [])
        n_masks = len(state_data.get("main_window", {}).get("masks", {}))
        n_features = len(main_window_state.get("features", {}))
        analysis_modules = list(state_data.get("analysis_modules", {}).keys())
        
        readme_content = f"""# OpenIMC Analysis State

This folder contains a complete analysis state from OpenIMC that can be loaded to reproduce the analysis.

## Contents

- `state.json`: Main state metadata and configuration
- `masks/`: Segmentation masks ({n_masks} acquisition(s))
- `features/`: Feature dataframes ({n_features} file(s))
- `analysis/`: Analysis module states ({len(analysis_modules)} module(s))
- `images/`: Saved images (if any)

## Analysis Modules Included

"""
        
        for module in analysis_modules:
            readme_content += f"- {module.replace('_', ' ').title()}\n"
        
        readme_content += f"""
## Acquisitions

"""
        for acq in acquisitions:
            readme_content += f"- {acq.get('name', acq.get('id', 'Unknown'))} (ID: {acq.get('id', 'N/A')})\n"
        
        readme_content += f"""
## Loading This State

To load this state in OpenIMC:

1. Open OpenIMC
2. Go to File → Load State…
3. Select this folder
4. The application will restore all masks, features, and analysis states

## OpenIMC Version

State format version: {self.state_version}

## Notes

This state was saved for reproducibility and can be uploaded to Zenodo or other data repositories as supplementary material for publications.

**For Zenodo Submission:**
- This folder contains all data needed to reproduce the analysis
- Consider also exporting "Analysis Steps" (File → Export Analysis Steps…) to include a human-readable methods description
- The analysis steps file can be included alongside this state folder

## File Structure

```
state_folder/
├── README.md              # This file
├── manifest.txt           # Complete file listing
├── state.json             # Main state metadata
├── masks/                 # Segmentation masks
├── features/              # Feature dataframes
├── analysis/              # Analysis module states
└── images/                # Saved images (if any)
```

For questions or issues, please refer to the OpenIMC documentation or GitHub repository.
"""
        
        with open(readme_path, 'w', encoding=self.TEXT_ENCODING, newline='\n') as f:
            f.write(readme_content)
    
    def _generate_manifest(self, state_path: Path):
        """Generate manifest.txt file listing all files."""
        manifest_path = state_path / "manifest.txt"
        
        files_list = []
        for root, dirs, files in os.walk(state_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.startswith('.'):
                    continue
                rel_path = Path(root).relative_to(state_path) / file
                file_path = state_path / rel_path
                if file_path.is_file():
                    size = file_path.stat().st_size
                    files_list.append(f"{rel_path}\t{size} bytes")
        
        with open(manifest_path, 'w', encoding=self.TEXT_ENCODING, newline='\n') as f:
            f.write("File Manifest\n")
            f.write("=" * 80 + "\n\n")
            f.write("This file lists all files in this state directory.\n\n")
            for line in sorted(files_list):
                f.write(line + "\n")
