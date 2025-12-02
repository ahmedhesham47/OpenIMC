# Feature Extraction Scalability Benchmarks

This directory contains scripts to benchmark the scalability of feature extraction in OpenIMC, measuring RAM usage, wall time, and maximum resident set size (RSS) while varying the number of images and workers.

## Overview

The benchmark suite consists of two main scripts:

1. **`feature_extraction_scalability.py`** - Runs the benchmarks and collects metrics
2. **`generate_scalability_plots.py`** - Generates visualization plots from the results

## Requirements

- Python 3.7+
- OpenIMC installed (or run from the OpenIMC repository root)
- Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `psutil`, `tifffile`

## Data Preparation

Prepare separate directories for images and masks:

```
images_directory/
├── image1.mcd
├── image2.mcd
├── ometiff_dir1/  (directory containing .ome.tif files)
└── ...

masks_directory/
├── image1_mask.tif (or image1_seg.tif, image1_segmentation.tif)
├── image2_mask.tif
└── ...
```

**Supported image formats:**
- `.mcd`, `.mcdx` - MCD files
- Directories containing `.ome.tif`, `.ome.tiff`, `.tif`, `.tiff` files (OME-TIFF format)

**Mask naming conventions:**
The core functions handle mask matching automatically by:
- First trying to match by well name (if available)
- Then falling back to acquisition name or ID

**Note:** Masks should be in `.tif` or `.tiff` format.

## Running Benchmarks

### Basic Usage

```bash
python feature_extraction_scalability.py \
    --images-dir /path/to/images \
    --masks-dir /path/to/masks \
    --output-dir ./results
```

### Custom Configuration

```bash
python feature_extraction_scalability.py \
    --images-dir /path/to/images \
    --masks-dir /path/to/masks \
    --output-dir ./results \
    --num-images 50 100 200 500 \
    --num-workers 1 4 8 16 22 \
    --repeats 3
```

**Parameters:**
- `--images-dir`: Directory containing image files (MCD files or OME-TIFF directories) (required)
- `--masks-dir`: Directory containing mask files (.tif, .tiff) (required)
- `--output-dir`: Output directory for results (default: `./results`)
- `--num-images`: List of acquisition counts to test (default: `50 100 200 500`)
- `--num-workers`: List of worker counts to test (default: `1 4 8 16 22`)
- `--repeats`: Number of times to repeat each configuration (default: `3`)
- `--channel-format`: Channel format for OME-TIFF files (`CHW` or `HWC`, default: `CHW`)

**Note:** `--num-images` refers to the number of acquisitions to process, not the number of image files/directories.

## Generating Plots

After running the benchmarks, generate visualization plots:

```bash
python generate_scalability_plots.py results/scalability_results.csv
```

Or specify a custom output directory:

```bash
python generate_scalability_plots.py results/scalability_results.csv --output-dir ./plots
```

### Generated Plots

The script generates the following plots:

1. **`wall_time_scalability.png`** - Wall time vs number of images and workers
2. **`ram_usage_scalability.png`** - Peak RAM usage vs number of images and workers
3. **`rss_scalability.png`** - Maximum RSS vs number of images and workers
4. **`scalability_heatmaps.png`** - Heatmaps showing all metrics across both dimensions

## Output Files

### `scalability_results.csv`

Contains the raw benchmark results with columns:
- `num_images`: Number of acquisitions processed
- `num_workers`: Number of parallel workers
- `wall_time`: Total wall clock time (seconds)
- `peak_ram_mb`: Peak virtual memory usage (MB)
- `max_rss_mb`: Maximum resident set size (MB)
- `success`: Whether the benchmark completed successfully
- `repeat`: Repeat number (if `--repeats > 1`)

### `output_data/` Directory

Contains extracted features for each benchmark run:
- `{num_images}_{num_workers}_{repeat}.csv` - Features extracted for that specific run

These files can be used to test reproducibility of feature extraction (deterministic behavior).

## Metrics Explained

- **Wall Time**: Total elapsed time from start to finish (what you'd measure with a stopwatch)
- **Peak RAM (VMS)**: Peak virtual memory size - the maximum total memory allocated (including swapped memory)
- **Max RSS**: Maximum resident set size - the maximum amount of physical RAM actually used

## Reproducibility Testing

The benchmark saves extracted features for each run in `output_data/`. You can compare features across repeats to verify that feature extraction is deterministic:

```python
import pandas as pd

# Load features from different repeats
features_0 = pd.read_csv('results/output_data/50_8_0.csv')
features_1 = pd.read_csv('results/output_data/50_8_1.csv')

# Compare to verify reproducibility
assert features_0.equals(features_1), "Features differ between runs!"
```

## Tips

1. **Start small**: Begin with fewer images and workers to ensure everything works
2. **Multiple repeats**: Use `--repeats 3` or more to get average performance and reduce variance
3. **Monitor system**: Ensure you have enough RAM for the largest configuration
4. **Clean environment**: Close other applications to get accurate memory measurements
5. **Memory reset**: The benchmark automatically resets memory between runs for accurate measurements

## Troubleshooting

### "No image files found"
- Check that your `--images-dir` path is correct
- Ensure image files have supported extensions (`.mcd`, `.mcdx`) or are directories containing OME-TIFF files

### "No mask found"
- Check mask file naming (see Data Preparation section)
- Ensure mask files are in the `--masks-dir` directory
- Verify mask files are readable TIFF files (`.tif`, `.tiff`)
- Check that mask file names match acquisition names/IDs (core functions handle matching)

### Memory errors
- Reduce the number of images or workers
- Ensure sufficient RAM is available
- Check that images aren't too large

### Import errors
- Ensure you're running from the OpenIMC repository root, or
- Install OpenIMC: `pip install -e .`

## Example Workflow

```bash
# 1. Prepare your data directories with images and masks
#    images/
#    ├── sample1.mcd
#    └── sample2.mcd
#    
#    masks/
#    ├── sample1_mask.tif
#    └── sample2_mask.tif

# 2. Run benchmarks
python feature_extraction_scalability.py \
    --images-dir ./images \
    --masks-dir ./masks \
    --output-dir ./results \
    --num-images 50 100 \
    --num-workers 1 4 8 \
    --repeats 3

# 3. Generate plots
python generate_scalability_plots.py ./results/scalability_results.csv

# 4. View results
#    - Check scalability_results.csv for raw data
#    - View generated PNG plots
#    - Compare features in output_data/ for reproducibility testing
```

