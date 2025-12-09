# Custom Algorithm Integration Guide

This directory contains base classes and examples for integrating custom segmentation, clustering, and feature extraction algorithms into OpenIMC.

## Quick Start

### Creating a Custom Segmenter

```python
from openimc.processing.base import BaseSegmenter
import numpy as np

class MySegmenter(BaseSegmenter):
    def __init__(self):
        super().__init__(name="my_segmenter")
    
    def segment(self, nuclear_image, cyto_image=None, **kwargs):
        # Input: nuclear_image (H, W), float32, 0-1 normalized
        # Output: mask (H, W), uint32, 0=background, 1+=cells
        mask = your_segmentation_algorithm(nuclear_image)
        return mask.astype(np.uint32)
```

### Creating a Custom Clusterer

```python
from openimc.processing.base import BaseClusterer
import pandas as pd

class MyClusterer(BaseClusterer):
    def __init__(self):
        super().__init__(name="my_clusterer")
    
    def cluster(self, features_df, columns=None, **kwargs):
        # Input: features_df (DataFrame with features)
        # Output: features_df with 'cluster' column added (1-based labels)
        cluster_labels = your_clustering_algorithm(features_df)
        result_df = features_df.copy()
        result_df['cluster'] = cluster_labels.astype(int)
        return result_df
```

### Creating a Custom Feature Extractor

```python
from openimc.processing.base import BaseFeatureExtractor
import pandas as pd

class MyExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__(name="my_extractor")
    
    def extract(self, mask, image_stack, channel_names, **kwargs):
        # Input: mask (H, W), uint32; image_stack (H, W, C), float32
        # Output: DataFrame with 'cell_id', 'label', and feature columns
        features_list = []
        for label in np.unique(mask[mask > 0]):
            features = {
                'cell_id': int(label),
                'label': int(label),
                # ... your features ...
            }
            features_list.append(features)
        return pd.DataFrame(features_list)
```

## Key Requirements

### Segmentation
- **Input**: `nuclear_image` (H, W), float32, 0-1 normalized
- **Output**: `mask` (H, W), uint32, 0=background, 1+=cell labels
- Each cell must have a unique integer label

### Clustering
- **Input**: `features_df` (DataFrame with numeric feature columns)
- **Output**: Same DataFrame with `'cluster'` column added
- Cluster labels: 1-based integers (0 = unassigned/noise)

### Feature Extraction
- **Input**: `mask` (H, W), uint32; `image_stack` (H, W, C), float32
- **Output**: DataFrame with required columns: `'cell_id'`, `'label'`
- Cell IDs: 1-based integers

## Files

- `base.py`: Abstract base classes with validation
- `examples.py`: Example implementations showing usage patterns
- See `docs/source/custom_algorithms.rst` for full documentation

## Validation

All base classes provide automatic input/output validation:

```python
# Automatic validation (called internally)
segmenter.validate_inputs(nuclear_image, cyto_image)
segmenter.validate_output(mask, expected_shape)
```

## Integration

After implementing your algorithm, integrate it into OpenIMC by modifying:
- `openimc.core.segment()` for segmenters
- `openimc.core.cluster()` for clusterers  
- `openimc.processing.feature_worker.extract_features_for_acquisition()` for extractors

See `docs/source/custom_algorithms.rst` for detailed integration instructions.

