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

## Main-Text Runtime and Memory Summary

Values are mean +/- SD across three successful benchmark repeats. Peak RAM is the `peak_ram_mb` metric converted to GB.

| Workflow | Dataset size (acquisitions) | Workers | Runtime (s) | Peak RAM (GB) |
| --- | ---: | ---: | ---: | ---: |
| Segmentation | 50 | 22 | 164.8 +/- 1.0 | 20.45 +/- 0.01 |
| Segmentation | 500 | 22 | 1586.0 +/- 17.4 | 21.03 +/- 0.04 |
| Feature Extraction | 50 | 22 | 41.3 +/- 0.3 | 3.25 +/- 0.06 |
| Feature Extraction | 500 | 22 | 322.8 +/- 1.0 | 8.65 +/- 1.02 |

## Usage

Navigate to the specific component's directory for detailed usage instructions.

