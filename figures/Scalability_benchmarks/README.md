# Scalability Benchmarks

This directory contains scalability benchmarks for different components of OpenIMC, measuring performance metrics (RAM usage, wall time, maximum resident set size) while varying key parameters.

## Structure

Each subdirectory contains benchmarks for a specific component:

- **`feature_extraction/`** - Feature extraction scalability benchmarks
- **`segmentation/`** - Segmentation scalability benchmarks (Cellpose cyto3)
- (More components will be added here)

## Metrics

All benchmarks measure:
- **wall_time**: Elapsed real-world time (wall-clock time) in seconds
- **peak_ram_mb**: Peak virtual memory size (total memory allocated) in MB
- **max_rss_mb**: Maximum resident set size (physical RAM actually used) in MB

## Usage

Navigate to the specific component's directory for detailed usage instructions.

