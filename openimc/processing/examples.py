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
Example implementations of base classes for OpenIMC.

This module provides example implementations showing how to create custom
segmentation, clustering, and feature extraction algorithms that integrate
with OpenIMC. These examples demonstrate the expected interface and can
serve as templates for novel algorithms.

Note: These are simplified examples for demonstration purposes. Real
implementations would include more sophisticated algorithms.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from skimage import measure, morphology, segmentation
from scipy import ndimage

from openimc.processing.base import (
    BaseSegmenter,
    BaseClusterer,
    BaseFeatureExtractor
)


class ExampleThresholdSegmenter(BaseSegmenter):
    """
    Example segmentation algorithm using simple thresholding.
    
    This is a minimal example showing how to implement a custom segmenter.
    It uses Otsu thresholding followed by watershed to separate touching cells.
    
    Example Usage:
        >>> segmenter = ExampleThresholdSegmenter()
        >>> mask = segmenter.segment(nuclear_image, cyto_image=None, threshold=0.5)
    """
    
    def __init__(self):
        super().__init__(name="example_threshold")
    
    def segment(
        self,
        nuclear_image: np.ndarray,
        cyto_image: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Perform segmentation using thresholding.
        
        Args:
            nuclear_image: Preprocessed nuclear channel (H, W), float32, 0-1
            cyto_image: Optional cytoplasm channel (H, W), float32, 0-1
            **kwargs: Additional parameters
                - threshold: Threshold value (float, default=0.5)
                - min_cell_area: Minimum cell area in pixels (int, default=50)
        
        Returns:
            Segmentation mask (H, W), uint32, 0=background, 1+=cells
        """
        # Validate inputs (inherited from BaseSegmenter)
        self.validate_inputs(nuclear_image, cyto_image)
        
        # Get parameters
        threshold = kwargs.get('threshold', 0.5)
        min_cell_area = kwargs.get('min_cell_area', 50)
        
        # Use nuclear image (primary) or cyto image if nuclear not provided
        # Note: In practice, nuclear_image should always be provided
        img = nuclear_image
        
        # Apply threshold
        binary = (img > threshold).astype(np.uint8)
        
        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=min_cell_area)
        
        # Separate touching cells using watershed
        distance = ndimage.distance_transform_edt(binary)
        local_maxima = morphology.local_maxima(distance, min_distance=5)
        markers = measure.label(local_maxima)
        
        # Apply watershed
        mask = segmentation.watershed(-distance, markers, mask=binary)
        
        # Ensure uint32 and validate
        mask = mask.astype(np.uint32)
        self.validate_output(mask, nuclear_image.shape)
        
        return mask


class ExampleKMeansClusterer(BaseClusterer):
    """
    Example clustering algorithm using K-means.
    
    This is a minimal example showing how to implement a custom clusterer.
    It uses scikit-learn's KMeans for demonstration.
    
    Example Usage:
        >>> clusterer = ExampleKMeansClusterer()
        >>> features_df = clusterer.cluster(features_df, n_clusters=5, seed=42)
    """
    
    def __init__(self):
        super().__init__(name="example_kmeans")
    
    def cluster(
        self,
        features_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform K-means clustering.
        
        Args:
            features_df: Feature DataFrame with one row per cell
            columns: Specific columns to use (None = auto-detect)
            **kwargs: Additional parameters
                - n_clusters: Number of clusters (int, required)
                - seed: Random seed (int, default=42)
                - n_init: Number of initializations (int, default=10)
        
        Returns:
            DataFrame with 'cluster' column added (1-based labels)
        """
        from sklearn.cluster import KMeans
        
        # Validate and prepare inputs
        data, column_names = self.validate_inputs(features_df, columns)
        original_shape = features_df.shape
        
        # Get parameters
        n_clusters = kwargs.get('n_clusters')
        if n_clusters is None:
            raise ValueError("n_clusters parameter is required")
        
        seed = kwargs.get('seed', 42)
        n_init = kwargs.get('n_init', 10)
        
        # Handle missing/infinite values
        data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median())
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=n_init,
            n_jobs=1
        )
        cluster_labels = kmeans.fit_predict(data.values)
        
        # Convert to 1-based labels (0 = unassigned, 1+ = clusters)
        cluster_labels = (cluster_labels + 1).astype(int)
        
        # Add cluster column to original DataFrame
        result_df = features_df.copy()
        result_df['cluster'] = cluster_labels
        
        # Validate output
        self.validate_output(result_df, original_shape)
        
        return result_df


class ExampleBasicFeatureExtractor(BaseFeatureExtractor):
    """
    Example feature extraction algorithm.
    
    This is a minimal example showing how to implement a custom feature extractor.
    It extracts basic morphological and intensity features.
    
    Example Usage:
        >>> extractor = ExampleBasicFeatureExtractor()
        >>> features_df = extractor.extract(mask, image_stack, channel_names)
    """
    
    def __init__(self):
        super().__init__(name="example_basic")
    
    def extract(
        self,
        mask: np.ndarray,
        image_stack: np.ndarray,
        channel_names: List[str],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract basic features from segmented cells.
        
        Args:
            mask: Segmentation mask (H, W), uint32, 0=background, 1+=cells
            image_stack: Image stack (H, W, C), float32
            channel_names: List of channel names, length C
            **kwargs: Additional parameters
                - morphological: Extract morphological features (bool, default=True)
                - intensity: Extract intensity features (bool, default=True)
        
        Returns:
            DataFrame with features (one row per cell)
        """
        # Validate inputs
        self.validate_inputs(mask, image_stack, channel_names)
        
        # Get parameters
        extract_morphological = kwargs.get('morphological', True)
        extract_intensity = kwargs.get('intensity', True)
        
        # Get unique cell labels (exclude background = 0)
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) == 0:
            # No cells found, return empty DataFrame with required columns
            return pd.DataFrame(columns=['cell_id', 'label'])
        
        # Prepare feature list
        features_list = []
        
        for idx, label in enumerate(unique_labels):
            cell_id = idx + 1  # 1-based cell ID
            features = {'cell_id': cell_id, 'label': int(label)}
            
            # Create binary mask for this cell
            cell_mask = (mask == label)
            
            # Extract morphological features
            if extract_morphological:
                # Area (in pixels)
                features['area_pixels'] = np.sum(cell_mask)
                
                # Perimeter (approximate)
                from skimage import measure
                contours = measure.find_contours(cell_mask.astype(float), 0.5)
                if len(contours) > 0:
                    perimeter = np.sum([
                        np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)).sum()
                        for contour in contours
                    ])
                    features['perimeter_pixels'] = perimeter
                else:
                    features['perimeter_pixels'] = 0.0
                
                # Bounding box
                coords = np.where(cell_mask)
                if len(coords[0]) > 0:
                    features['bbox_min_row'] = int(coords[0].min())
                    features['bbox_min_col'] = int(coords[1].min())
                    features['bbox_max_row'] = int(coords[0].max())
                    features['bbox_max_col'] = int(coords[1].max())
            
            # Extract intensity features
            if extract_intensity:
                for ch_idx, ch_name in enumerate(channel_names):
                    channel_img = image_stack[:, :, ch_idx]
                    cell_pixels = channel_img[cell_mask]
                    
                    if len(cell_pixels) > 0:
                        features[f'mean_{ch_name}'] = float(np.mean(cell_pixels))
                        features[f'median_{ch_name}'] = float(np.median(cell_pixels))
                        features[f'std_{ch_name}'] = float(np.std(cell_pixels))
                        features[f'max_{ch_name}'] = float(np.max(cell_pixels))
                        features[f'min_{ch_name}'] = float(np.min(cell_pixels))
                        features[f'sum_{ch_name}'] = float(np.sum(cell_pixels))
                    else:
                        # Cell has no pixels (shouldn't happen, but handle gracefully)
                        features[f'mean_{ch_name}'] = 0.0
                        features[f'median_{ch_name}'] = 0.0
                        features[f'std_{ch_name}'] = 0.0
                        features[f'max_{ch_name}'] = 0.0
                        features[f'min_{ch_name}'] = 0.0
                        features[f'sum_{ch_name}'] = 0.0
            
            features_list.append(features)
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Validate output
        expected_n_cells = len(unique_labels)
        self.validate_output(features_df, expected_n_cells)
        
        return features_df

