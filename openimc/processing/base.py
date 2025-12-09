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
Base classes for OpenIMC processing algorithms.

This module provides abstract base classes that define the interface for
segmentation, clustering, and feature extraction algorithms. These classes
ensure consistent input/output formats and make it easy for developers to
integrate novel algorithms into OpenIMC.

Example:
    To create a custom segmenter::
    
        from openimc.processing.base import BaseSegmenter
        import numpy as np
        
        class MyCustomSegmenter(BaseSegmenter):
            def __init__(self):
                super().__init__()
                self.name = "my_custom_segmenter"
            
            def segment(self, nuclear_image, cyto_image=None, **kwargs):
                # Your segmentation logic here
                mask = np.zeros_like(nuclear_image, dtype=np.uint32)
                # ... perform segmentation ...
                return mask
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path


class BaseSegmenter(ABC):
    """
    Abstract base class for segmentation algorithms.
    
    This class defines the interface that all segmentation algorithms must
    implement to integrate with OpenIMC. It ensures consistent input/output
    formats and provides validation.
    
    Expected Inputs:
        - nuclear_image: np.ndarray, shape (H, W), dtype float32
            Preprocessed nuclear channel image (0-1 normalized)
        - cyto_image: np.ndarray, shape (H, W), dtype float32, optional
            Preprocessed cytoplasm channel image (0-1 normalized)
        - **kwargs: Additional algorithm-specific parameters
    
    Expected Output:
        - mask: np.ndarray, shape (H, W), dtype uint32
            Segmentation mask where each cell has a unique integer label
            (0 = background, 1+ = cell labels)
    
    Example:
        >>> class MySegmenter(BaseSegmenter):
        ...     def segment(self, nuclear_image, cyto_image=None, **kwargs):
        ...         # Your implementation
        ...         return mask
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the segmenter.
        
        Args:
            name: Name identifier for this segmenter (used in logs/UI)
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def segment(
        self,
        nuclear_image: np.ndarray,
        cyto_image: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Perform cell segmentation.
        
        This is the main method that must be implemented by all segmenters.
        It takes preprocessed images and returns a segmentation mask.
        
        Args:
            nuclear_image: Preprocessed nuclear channel image
                - Shape: (H, W)
                - Dtype: float32
                - Range: 0.0-1.0 (normalized)
            cyto_image: Preprocessed cytoplasm channel image (optional)
                - Shape: (H, W) or None
                - Dtype: float32
                - Range: 0.0-1.0 (normalized)
            **kwargs: Additional algorithm-specific parameters
                Common parameters include:
                - diameter: Expected cell diameter in pixels (int, optional)
                - threshold: Segmentation threshold (float, optional)
                - model_path: Path to pre-trained model (str, optional)
        
        Returns:
            Segmentation mask
                - Shape: (H, W), matching nuclear_image
                - Dtype: uint32
                - Values: 0 = background, 1+ = cell labels
                - Each cell must have a unique integer label
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If segmentation fails
        """
        pass
    
    def validate_inputs(
        self,
        nuclear_image: np.ndarray,
        cyto_image: Optional[np.ndarray] = None
    ) -> None:
        """
        Validate input images before segmentation.
        
        This method can be overridden to add custom validation, but the
        base implementation checks common requirements.
        
        Args:
            nuclear_image: Nuclear channel image to validate
            cyto_image: Cytoplasm channel image to validate (optional)
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(nuclear_image, np.ndarray):
            raise ValueError(f"nuclear_image must be numpy array, got {type(nuclear_image)}")
        
        if nuclear_image.ndim != 2:
            raise ValueError(f"nuclear_image must be 2D (H, W), got shape {nuclear_image.shape}")
        
        if nuclear_image.dtype != np.float32:
            raise ValueError(f"nuclear_image must be float32, got {nuclear_image.dtype}")
        
        if cyto_image is not None:
            if not isinstance(cyto_image, np.ndarray):
                raise ValueError(f"cyto_image must be numpy array, got {type(cyto_image)}")
            
            if cyto_image.ndim != 2:
                raise ValueError(f"cyto_image must be 2D (H, W), got shape {cyto_image.shape}")
            
            if cyto_image.shape != nuclear_image.shape:
                raise ValueError(
                    f"cyto_image shape {cyto_image.shape} must match "
                    f"nuclear_image shape {nuclear_image.shape}"
                )
            
            if cyto_image.dtype != np.float32:
                raise ValueError(f"cyto_image must be float32, got {cyto_image.dtype}")
    
    def validate_output(self, mask: np.ndarray, expected_shape: Tuple[int, int]) -> None:
        """
        Validate segmentation mask output.
        
        Args:
            mask: Segmentation mask to validate
            expected_shape: Expected (H, W) shape
        
        Raises:
            ValueError: If mask is invalid
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"mask must be numpy array, got {type(mask)}")
        
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D (H, W), got shape {mask.shape}")
        
        if mask.shape != expected_shape:
            raise ValueError(
                f"mask shape {mask.shape} must match expected shape {expected_shape}"
            )
        
        if mask.dtype != np.uint32:
            raise ValueError(f"mask must be uint32, got {mask.dtype}")
        
        if mask.min() < 0:
            raise ValueError(f"mask values must be >= 0, got min={mask.min()}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this segmenter.
        
        Returns:
            Dictionary with segmenter metadata
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__
        }


class BaseClusterer(ABC):
    """
    Abstract base class for clustering algorithms.
    
    This class defines the interface that all clustering algorithms must
    implement to integrate with OpenIMC. It ensures consistent input/output
    formats and provides validation.
    
    Expected Inputs:
        - features_df: pd.DataFrame
            Feature matrix with one row per cell and one column per feature
            Required columns: None (all numeric columns are used)
            Excluded columns: 'cell_id', 'acquisition_id', 'acquisition_name',
                             'well', 'cluster', 'label', 'source_file', etc.
        - columns: List[str], optional
            Specific feature columns to use for clustering
            If None, auto-detects all numeric columns
        - **kwargs: Additional algorithm-specific parameters
    
    Expected Output:
        - features_df: pd.DataFrame
            Same DataFrame as input with 'cluster' column added
            - 'cluster' column: int, 1-based cluster labels (0 = unassigned/noise)
    
    Example:
        >>> class MyClusterer(BaseClusterer):
        ...     def cluster(self, features_df, columns=None, **kwargs):
        ...         # Your implementation
        ...         features_df['cluster'] = cluster_labels
        ...         return features_df
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the clusterer.
        
        Args:
            name: Name identifier for this clusterer (used in logs/UI)
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def cluster(
        self,
        features_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform clustering on feature data.
        
        This is the main method that must be implemented by all clusterers.
        It takes a feature DataFrame and returns it with cluster labels added.
        
        Args:
            features_df: Feature matrix with one row per cell
                - Required: Numeric feature columns
                - Excluded: Metadata columns (cell_id, acquisition_id, etc.)
            columns: Specific feature columns to use (None = auto-detect)
                - If None: Uses all numeric columns except metadata
                - If specified: Uses only these columns
            **kwargs: Additional algorithm-specific parameters
                Common parameters include:
                - n_clusters: Number of clusters (int, optional)
                - resolution: Resolution parameter (float, optional)
                - seed: Random seed (int, default=42)
                - metric: Distance metric (str, default='euclidean')
        
        Returns:
            DataFrame with 'cluster' column added
                - Same shape as input
                - 'cluster' column: int, 1-based labels (0 = unassigned/noise)
                - All original columns preserved
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If clustering fails
        """
        pass
    
    def validate_inputs(
        self,
        features_df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and prepare input features for clustering.
        
        Args:
            features_df: Feature DataFrame to validate
            columns: Specific columns to use (None = auto-detect)
        
        Returns:
            Tuple of (data_subset, column_names)
            - data_subset: DataFrame with only feature columns
            - column_names: List of column names used
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError(f"features_df must be pandas DataFrame, got {type(features_df)}")
        
        if features_df.empty:
            raise ValueError("features_df cannot be empty")
        
        # Auto-detect columns if not specified
        if columns is None:
            exclude_cols = {
                'cell_id', 'acquisition_id', 'acquisition_name', 'well', 'cluster',
                'label', 'source_file', 'source_well', 'acquisition_label'
            }
            columns = [
                col for col in features_df.columns
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(features_df[col])
            ]
        
        if not columns:
            raise ValueError("No valid feature columns found for clustering")
        
        # Validate columns exist
        missing = [col for col in columns if col not in features_df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        
        # Extract feature data
        data = features_df[columns].copy()
        
        # Check for empty data
        if data.empty:
            raise ValueError("Feature data is empty after column selection")
        
        return data, columns
    
    def validate_output(
        self,
        features_df: pd.DataFrame,
        original_shape: Tuple[int, int]
    ) -> None:
        """
        Validate clustering output.
        
        Args:
            features_df: Output DataFrame with cluster labels
            original_shape: Original (n_rows, n_cols) shape
        
        Raises:
            ValueError: If output is invalid
        """
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError(f"features_df must be pandas DataFrame, got {type(features_df)}")
        
        if features_df.shape[0] != original_shape[0]:
            raise ValueError(
                f"Output DataFrame must have same number of rows as input "
                f"({original_shape[0]}), got {features_df.shape[0]}"
            )
        
        if 'cluster' not in features_df.columns:
            raise ValueError("Output DataFrame must have 'cluster' column")
        
        cluster_col = features_df['cluster']
        if not pd.api.types.is_integer_dtype(cluster_col):
            raise ValueError(f"'cluster' column must be integer type, got {cluster_col.dtype}")
        
        if cluster_col.isna().any():
            raise ValueError("'cluster' column cannot contain NaN values")
        
        if (cluster_col < 0).any():
            raise ValueError("'cluster' column must contain non-negative values (0 = unassigned)")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this clusterer.
        
        Returns:
            Dictionary with clusterer metadata
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__
        }


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extraction algorithms.
    
    This class defines the interface that all feature extraction algorithms
    must implement to integrate with OpenIMC. It ensures consistent input/output
    formats and provides validation.
    
    Expected Inputs:
        - mask: np.ndarray, shape (H, W), dtype uint32
            Segmentation mask with cell labels (0 = background, 1+ = cells)
        - image_stack: np.ndarray, shape (H, W, C), dtype float32
            Image stack with C channels
        - channel_names: List[str], length C
            Names of each channel in image_stack
        - **kwargs: Additional algorithm-specific parameters
    
    Expected Output:
        - features_df: pd.DataFrame
            Feature matrix with one row per cell
            Required columns:
                - 'cell_id': int, unique identifier for each cell (1-based)
                - 'label': int, cell label from mask (1-based)
                - Additional feature columns (algorithm-specific)
    
    Example:
        >>> class MyFeatureExtractor(BaseFeatureExtractor):
        ...     def extract(self, mask, image_stack, channel_names, **kwargs):
        ...         # Your implementation
        ...         return features_df
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the feature extractor.
        
        Args:
            name: Name identifier for this extractor (used in logs/UI)
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def extract(
        self,
        mask: np.ndarray,
        image_stack: np.ndarray,
        channel_names: List[str],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from segmented cells.
        
        This is the main method that must be implemented by all feature extractors.
        It takes a segmentation mask and image stack and returns a feature DataFrame.
        
        Args:
            mask: Segmentation mask
                - Shape: (H, W)
                - Dtype: uint32
                - Values: 0 = background, 1+ = cell labels
            image_stack: Image stack with all channels
                - Shape: (H, W, C)
                - Dtype: float32
                - C = number of channels
            channel_names: List of channel names
                - Length: C (must match image_stack channels)
                - Example: ['DNA1_Ir191', 'CD3_1841', ...]
            **kwargs: Additional algorithm-specific parameters
                Common parameters include:
                - morphological: Extract morphological features (bool, default=True)
                - intensity: Extract intensity features (bool, default=True)
                - selected_features: Dict of feature flags (Dict[str, bool], optional)
        
        Returns:
            DataFrame with extracted features
                - One row per cell (excluding background)
                - Required columns: 'cell_id', 'label'
                - Additional columns: Feature-specific (e.g., 'area_um2', 'mean_DNA1_Ir191', ...)
                - 'cell_id': int, 1-based unique identifier
                - 'label': int, cell label from mask (1-based)
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If feature extraction fails
        """
        pass
    
    def validate_inputs(
        self,
        mask: np.ndarray,
        image_stack: np.ndarray,
        channel_names: List[str]
    ) -> None:
        """
        Validate input data before feature extraction.
        
        Args:
            mask: Segmentation mask to validate
            image_stack: Image stack to validate
            channel_names: Channel names to validate
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"mask must be numpy array, got {type(mask)}")
        
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D (H, W), got shape {mask.shape}")
        
        if mask.dtype != np.uint32:
            raise ValueError(f"mask must be uint32, got {mask.dtype}")
        
        if not isinstance(image_stack, np.ndarray):
            raise ValueError(f"image_stack must be numpy array, got {type(image_stack)}")
        
        if image_stack.ndim != 3:
            raise ValueError(f"image_stack must be 3D (H, W, C), got shape {image_stack.shape}")
        
        if image_stack.shape[:2] != mask.shape:
            raise ValueError(
                f"image_stack spatial dimensions {image_stack.shape[:2]} must match "
                f"mask shape {mask.shape}"
            )
        
        if image_stack.dtype != np.float32:
            raise ValueError(f"image_stack must be float32, got {image_stack.dtype}")
        
        if not isinstance(channel_names, list):
            raise ValueError(f"channel_names must be list, got {type(channel_names)}")
        
        if len(channel_names) != image_stack.shape[2]:
            raise ValueError(
                f"channel_names length {len(channel_names)} must match "
                f"image_stack channels {image_stack.shape[2]}"
            )
    
    def validate_output(
        self,
        features_df: pd.DataFrame,
        expected_n_cells: int
    ) -> None:
        """
        Validate feature extraction output.
        
        Args:
            features_df: Output DataFrame to validate
            expected_n_cells: Expected number of cells (non-zero labels in mask)
        
        Raises:
            ValueError: If output is invalid
        """
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError(f"features_df must be pandas DataFrame, got {type(features_df)}")
        
        if features_df.empty:
            raise ValueError("features_df cannot be empty")
        
        # Check required columns
        required_cols = {'cell_id', 'label'}
        missing = required_cols - set(features_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Validate cell_id
        if not pd.api.types.is_integer_dtype(features_df['cell_id']):
            raise ValueError("'cell_id' column must be integer type")
        
        if features_df['cell_id'].isna().any():
            raise ValueError("'cell_id' column cannot contain NaN values")
        
        if (features_df['cell_id'] < 1).any():
            raise ValueError("'cell_id' must be >= 1 (1-based indexing)")
        
        # Validate label
        if not pd.api.types.is_integer_dtype(features_df['label']):
            raise ValueError("'label' column must be integer type")
        
        if features_df['label'].isna().any():
            raise ValueError("'label' column cannot contain NaN values")
        
        if (features_df['label'] < 1).any():
            raise ValueError("'label' must be >= 1 (1-based indexing)")
        
        # Check number of cells (approximate, may vary slightly)
        n_cells = len(features_df)
        if n_cells == 0:
            raise ValueError("No cells found in features_df")
        
        # Note: We don't enforce exact match because some extractors may filter cells
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this feature extractor.
        
        Returns:
            Dictionary with extractor metadata
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__
        }

