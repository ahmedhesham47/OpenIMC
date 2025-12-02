# Segmentation Scalability Benchmarks

This directory contains scripts to benchmark the scalability of segmentation in OpenIMC, measuring RAM usage, wall time, maximum resident set size (RSS), and GPU vRAM while varying the number of images, batch size, and number of workers.

## Overview

The benchmark suite consists of two main scripts:

1. **`segmentation_scalability.py`** - Runs the benchmarks and collects metrics
2. **`generate_scalability_plots.py`** - Generates visualization plots from the results

## Requirements

- Python 3.7+
- OpenIMC installed (or run from the OpenIMC repository root)
- Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `psutil`, `tifffile`, `cellpose`
- Optional: GPU support for faster Cellpose segmentation

## Data Preparation

Prepare a directory containing image files:

```
images_directory/
├── image1.mcd
├── image2.mcd
├── ometiff_dir1/  (directory containing .ome.tif files)
└── ...
```

**Supported image formats:**
- `.mcd`, `.mcdx` - MCD files
- Directories containing `.ome.tif`, `.ome.tiff`, `.tif`, `.tiff` files (OME-TIFF format)

## Running Benchmarks

### Basic Usage

```bash
python segmentation_scalability.py \
    --images-dir /path/to/images \
    --output-dir ./results \
    --nuclear-channels "Histone_1261726In113Di,Histone_473968La139Di,Histone_phospho_383738Eu153Di,Iridium_10331253Ir191Di,Iridium_10331254Ir193Di" \
    --cyto-channels "Cytoker_651779Pr141Di,Cytoker_3111576Nd143Di,Cytoker_971099Nd144Di,Keratin_346876Sm147Di,CD68_77877Nd146Di,SMA_174864Nd148Di,Vimenti_1921755Sm149Di,c-erbB-_201487Eu151Di,CD3epsi_8001752Sm152Di,Progest_312878Gd158Di,CD44_6967Gd160Di,CD45_71790Dy162Di,CD20_361077Dy164Di,E-Cadhe_1031747Er167Di,panCyto_234832Lu175Di,Cytoker_98922Yb174Di"
```

**Note:** The benchmark uses:
- **Mean combination** for both nuclear and cytoplasmic channels
- **Channelwise min-max scaling** for normalization (applied automatically)

### Custom Configuration

```bash
python segmentation_scalability.py \
    --images-dir /path/to/images \
    --output-dir ./results \
    --nuclear-channels DNA1_Ir191,DNA2_Ir193 \
    --cyto-channels CD3_1841,CD4_2293 \
    --num-images 50 100 200 500 \
    --num-workers 1 4 8 16 22 \
    --repeats 3 \
    --cellpose-model cyto3 \
    --gpu-id 0
```

**Parameters:**
- `--images-dir`: Directory containing image files (MCD files or OME-TIFF directories) (required)
- `--output-dir`: Output directory for results (default: ./results)
- `--nuclear-channels`: Comma-separated list of nuclear channel names (required)
- `--cyto-channels`: Comma-separated list of cytoplasm channel names (optional, required for cyto3 model)
- `--num-images`: Number of images to test (default: 50 100 200 500)
- `--num-workers`: Number of workers for multiprocessing during loading/preprocessing. Can specify multiple values to test scalability (e.g., `1 4 8 16`). Use `None` or omit for auto (default: auto, uses cpu_count - 2)
- `--repeats`: Number of times to repeat each configuration (default: 3)
- `--cellpose-model`: Cellpose model type - 'cyto3' or 'nuclei' (default: cyto3)
- `--diameter`: Cell diameter in pixels (optional, Cellpose will estimate if not provided)
- `--flow-threshold`: Flow threshold for Cellpose (default: 0.4)
- `--cellprob-threshold`: Cell probability threshold for Cellpose (default: 0.0)
- `--gpu-id`: GPU ID to use (optional, e.g., "0" or "auto")
- `--denoise-settings`: JSON file or string with denoise settings (optional)
- `--channel-format`: Channel format for OME-TIFF files - 'CHW' or 'HWC' (default: CHW)

## Metrics Collected

The benchmark collects the following metrics:

- **wall_time**: Elapsed real-world time from start to finish (wall-clock time) in seconds
- **peak_ram_mb**: Peak virtual memory size (total memory allocated) in MB
- **max_rss_mb**: Maximum resident set size (physical RAM actually used) in MB
- **num_cells_total**: Total number of cells detected across all acquisitions
- **num_acquisitions**: Number of acquisitions successfully processed

## Generating Plots

After running the benchmark, generate visualization plots:

```bash
python generate_scalability_plots.py results/scalability_results.csv
```

This will generate:
- `wall_time_scalability.png` - Wall time vs number of images/workers
- `ram_usage_scalability.png` - RAM usage vs number of images/workers
- `rss_scalability.png` - Maximum RSS vs number of images/workers
- `scalability_heatmaps.png` - Heatmaps showing scalability across both dimensions

## Implementation Details

The benchmark uses the same core functions as the CLI tool `openimc segment`:

- **Core function**: `openimc.core.segment()` - Unified segmentation function used by both GUI and CLI
- **Data loading**: `openimc.core.load_mcd()` - Unified data loading function
- **Parallelization**: Uses multiprocessing with spawn context (same as feature extraction)
- **Memory management**: Clears image caches before and after processing (same as GUI)

The benchmark tests scalability by:
1. Varying the number of images (acquisitions) to process
2. Varying the batch size for Cellpose (default: 16, matches GUI)
3. Varying the number of workers for multiprocessing during loading/preprocessing
4. Measuring performance metrics (wall time, RAM, RSS, vRAM) for each configuration

## Notes

- The benchmark uses Cellpose with the cyto3 model by default
- **Normalization**: Channelwise min-max scaling is applied automatically to all channels before combination
- **Channel combination**: Both nuclear and cytoplasmic channels are combined using mean (default)
- Masks are saved to `output_data/masks/` for reproducibility
- Each benchmark run starts with a clean memory state (cache clearing and garbage collection)
- The benchmark follows the same patterns as the feature extraction benchmark for consistency
- See `example_command.sh` for a complete example with the specified channels

