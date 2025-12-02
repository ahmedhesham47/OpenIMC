#!/usr/bin/env python3
"""
Generate scalability plots from benchmark results.

Usage:
    python generate_scalability_plots.py results/scalability_results.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def load_results(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load benchmark results from CSV.
    
    Returns:
        (raw_df, summary_df) - Raw data and summary with mean/std
    """
    df = pd.read_csv(csv_path)
    
    # Filter to successful runs only
    if 'success' in df.columns:
        df = df[df['success'] == True]
    
    # Ensure num_images and num_workers are numeric
    df['num_images'] = pd.to_numeric(df['num_images'])
    df['num_workers'] = pd.to_numeric(df['num_workers'])
    
    # Create summary with mean and std
    if 'repeat' in df.columns:
        summary_df = df.groupby(['num_images', 'num_workers']).agg({
            'wall_time': ['mean', 'std'],
            'peak_ram_mb': ['mean', 'std'],
            'max_rss_mb': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        summary_df.columns = ['num_images', 'num_workers',
                            'wall_time_mean', 'wall_time_std',
                            'peak_ram_mb_mean', 'peak_ram_mb_std',
                            'max_rss_mb_mean', 'max_rss_mb_std']
    else:
        # No repeats, just use values as mean with zero std
        summary_df = df.copy()
        summary_df['wall_time_mean'] = summary_df['wall_time']
        summary_df['wall_time_std'] = 0
        summary_df['peak_ram_mb_mean'] = summary_df['peak_ram_mb']
        summary_df['peak_ram_mb_std'] = 0
        summary_df['max_rss_mb_mean'] = summary_df['max_rss_mb']
        summary_df['max_rss_mb_std'] = 0
    
    return df, summary_df


def plot_wall_time(raw_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path):
    """Plot wall time vs number of images for different worker counts with boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Wall time vs number of images (different lines for different workers)
    ax1 = axes[0]
    for num_workers in sorted(summary_df['num_workers'].unique()):
        subset = summary_df[summary_df['num_workers'] == num_workers].sort_values('num_images')
        
        # Plot line with error bars
        ax1.errorbar(
            subset['num_images'],
            subset['wall_time_mean'],
            yerr=subset['wall_time_std'],
            marker='o',
            label=f'{num_workers} worker{"s" if num_workers > 1 else ""}',
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            elinewidth=2
        )
        
        # Add boxplots at each point if we have raw data
        if 'repeat' in raw_df.columns:
            for _, row in subset.iterrows():
                num_imgs = row['num_images']
                raw_subset = raw_df[(raw_df['num_workers'] == num_workers) & 
                                   (raw_df['num_images'] == num_imgs)]['wall_time']
                if len(raw_subset) > 1:
                    # Create small boxplot offset slightly
                    bp = ax1.boxplot([raw_subset.values], 
                                    positions=[num_imgs],
                                    widths=num_imgs * 0.05,  # Scale width with x-axis
                                    patch_artist=True,
                                    showfliers=False,
                                    boxprops=dict(alpha=0.3, facecolor='none', edgecolor='gray', linewidth=1),
                                    medianprops=dict(visible=False),
                                    whiskerprops=dict(visible=False),
                                    capprops=dict(visible=False))
    
    ax1.set_xlabel('Number of Images', fontsize=12)
    ax1.set_ylabel('Wall Time (seconds)', fontsize=12)
    ax1.set_title('Wall Time vs Number of Images', fontsize=14, fontweight='bold')
    ax1.legend(title='Workers', fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Set x-axis ticks to actual image counts
    ax1.set_xticks(sorted(summary_df['num_images'].unique()))
    ax1.set_xticklabels([str(x) for x in sorted(summary_df['num_images'].unique())])
    
    # Plot 2: Wall time vs number of workers (different lines for different image counts)
    ax2 = axes[1]
    for num_images in sorted(summary_df['num_images'].unique()):
        subset = summary_df[summary_df['num_images'] == num_images].sort_values('num_workers')
        
        # Plot line with error bars
        ax2.errorbar(
            subset['num_workers'],
            subset['wall_time_mean'],
            yerr=subset['wall_time_std'],
            marker='s',
            label=f'{num_images} image{"s" if num_images > 1 else ""}',
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            elinewidth=2
        )
        
        # Add boxplots at each point if we have raw data
        if 'repeat' in raw_df.columns:
            for _, row in subset.iterrows():
                num_wrks = row['num_workers']
                raw_subset = raw_df[(raw_df['num_images'] == num_images) & 
                                   (raw_df['num_workers'] == num_wrks)]['wall_time']
                if len(raw_subset) > 1:
                    # Create small boxplot
                    bp = ax2.boxplot([raw_subset.values], 
                                    positions=[num_wrks],
                                    widths=num_wrks * 0.1,
                                    patch_artist=True,
                                    showfliers=False,
                                    boxprops=dict(alpha=0.3, facecolor='none', edgecolor='gray', linewidth=1),
                                    medianprops=dict(visible=False),
                                    whiskerprops=dict(visible=False),
                                    capprops=dict(visible=False))
    
    ax2.set_xlabel('Number of Workers', fontsize=12)
    ax2.set_ylabel('Wall Time (seconds)', fontsize=12)
    ax2.set_title('Wall Time vs Number of Workers', fontsize=14, fontweight='bold')
    ax2.legend(title='Images', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'wall_time_scalability.png', bbox_inches='tight')
    print(f"Saved: {output_path / 'wall_time_scalability.png'}")
    plt.close()


def plot_ram_usage(raw_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path):
    """Plot RAM usage vs number of images for different worker counts with boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Peak RAM vs number of images
    ax1 = axes[0]
    for num_workers in sorted(summary_df['num_workers'].unique()):
        subset = summary_df[summary_df['num_workers'] == num_workers].sort_values('num_images')
        
        # Plot line with error bars
        ax1.errorbar(
            subset['num_images'],
            subset['peak_ram_mb_mean'],
            yerr=subset['peak_ram_mb_std'],
            marker='o',
            label=f'{num_workers} worker{"s" if num_workers > 1 else ""}',
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            elinewidth=2
        )
        
        # Add boxplots at each point if we have raw data
        if 'repeat' in raw_df.columns:
            for _, row in subset.iterrows():
                num_imgs = row['num_images']
                raw_subset = raw_df[(raw_df['num_workers'] == num_workers) & 
                                   (raw_df['num_images'] == num_imgs)]['peak_ram_mb']
                if len(raw_subset) > 1:
                    bp = ax1.boxplot([raw_subset.values], 
                                    positions=[num_imgs],
                                    widths=num_imgs * 0.05,
                                    patch_artist=True,
                                    showfliers=False,
                                    boxprops=dict(alpha=0.3, facecolor='none', edgecolor='gray', linewidth=1),
                                    medianprops=dict(visible=False),
                                    whiskerprops=dict(visible=False),
                                    capprops=dict(visible=False))
    
    ax1.set_xlabel('Number of Images', fontsize=12)
    ax1.set_ylabel('Peak RAM Usage (MB)', fontsize=12)
    ax1.set_title('Peak RAM Usage vs Number of Images', fontsize=14, fontweight='bold')
    ax1.legend(title='Workers', fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Set x-axis ticks to actual image counts
    ax1.set_xticks(sorted(summary_df['num_images'].unique()))
    ax1.set_xticklabels([str(x) for x in sorted(summary_df['num_images'].unique())])
    
    # Plot 2: Peak RAM vs number of workers
    ax2 = axes[1]
    for num_images in sorted(summary_df['num_images'].unique()):
        subset = summary_df[summary_df['num_images'] == num_images].sort_values('num_workers')
        
        # Plot line with error bars
        ax2.errorbar(
            subset['num_workers'],
            subset['peak_ram_mb_mean'],
            yerr=subset['peak_ram_mb_std'],
            marker='s',
            label=f'{num_images} image{"s" if num_images > 1 else ""}',
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            elinewidth=2
        )
        
        # Add boxplots at each point if we have raw data
        if 'repeat' in raw_df.columns:
            for _, row in subset.iterrows():
                num_wrks = row['num_workers']
                raw_subset = raw_df[(raw_df['num_images'] == num_images) & 
                                   (raw_df['num_workers'] == num_wrks)]['peak_ram_mb']
                if len(raw_subset) > 1:
                    bp = ax2.boxplot([raw_subset.values], 
                                    positions=[num_wrks],
                                    widths=num_wrks * 0.1,
                                    patch_artist=True,
                                    showfliers=False,
                                    boxprops=dict(alpha=0.3, facecolor='none', edgecolor='gray', linewidth=1),
                                    medianprops=dict(visible=False),
                                    whiskerprops=dict(visible=False),
                                    capprops=dict(visible=False))
    
    ax2.set_xlabel('Number of Workers', fontsize=12)
    ax2.set_ylabel('Peak RAM Usage (MB)', fontsize=12)
    ax2.set_title('Peak RAM Usage vs Number of Workers', fontsize=14, fontweight='bold')
    ax2.legend(title='Images', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ram_usage_scalability.png', bbox_inches='tight')
    print(f"Saved: {output_path / 'ram_usage_scalability.png'}")
    plt.close()


def plot_rss(raw_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path):
    """Plot maximum RSS vs number of images for different worker counts with boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Max RSS vs number of images
    ax1 = axes[0]
    for num_workers in sorted(summary_df['num_workers'].unique()):
        subset = summary_df[summary_df['num_workers'] == num_workers].sort_values('num_images')
        
        # Plot line with error bars
        ax1.errorbar(
            subset['num_images'],
            subset['max_rss_mb_mean'],
            yerr=subset['max_rss_mb_std'],
            marker='o',
            label=f'{num_workers} worker{"s" if num_workers > 1 else ""}',
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            elinewidth=2
        )
        
        # Add boxplots at each point if we have raw data
        if 'repeat' in raw_df.columns:
            for _, row in subset.iterrows():
                num_imgs = row['num_images']
                raw_subset = raw_df[(raw_df['num_workers'] == num_workers) & 
                                   (raw_df['num_images'] == num_imgs)]['max_rss_mb']
                if len(raw_subset) > 1:
                    bp = ax1.boxplot([raw_subset.values], 
                                    positions=[num_imgs],
                                    widths=num_imgs * 0.05,
                                    patch_artist=True,
                                    showfliers=False,
                                    boxprops=dict(alpha=0.3, facecolor='none', edgecolor='gray', linewidth=1),
                                    medianprops=dict(visible=False),
                                    whiskerprops=dict(visible=False),
                                    capprops=dict(visible=False))
    
    ax1.set_xlabel('Number of Images', fontsize=12)
    ax1.set_ylabel('Maximum RSS (MB)', fontsize=12)
    ax1.set_title('Maximum RSS vs Number of Images', fontsize=14, fontweight='bold')
    ax1.legend(title='Workers', fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Set x-axis ticks to actual image counts
    ax1.set_xticks(sorted(summary_df['num_images'].unique()))
    ax1.set_xticklabels([str(x) for x in sorted(summary_df['num_images'].unique())])
    
    # Plot 2: Max RSS vs number of workers
    ax2 = axes[1]
    for num_images in sorted(summary_df['num_images'].unique()):
        subset = summary_df[summary_df['num_images'] == num_images].sort_values('num_workers')
        
        # Plot line with error bars
        ax2.errorbar(
            subset['num_workers'],
            subset['max_rss_mb_mean'],
            yerr=subset['max_rss_mb_std'],
            marker='s',
            label=f'{num_images} image{"s" if num_images > 1 else ""}',
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            elinewidth=2
        )
        
        # Add boxplots at each point if we have raw data
        if 'repeat' in raw_df.columns:
            for _, row in subset.iterrows():
                num_wrks = row['num_workers']
                raw_subset = raw_df[(raw_df['num_images'] == num_images) & 
                                   (raw_df['num_workers'] == num_wrks)]['max_rss_mb']
                if len(raw_subset) > 1:
                    bp = ax2.boxplot([raw_subset.values], 
                                    positions=[num_wrks],
                                    widths=num_wrks * 0.1,
                                    patch_artist=True,
                                    showfliers=False,
                                    boxprops=dict(alpha=0.3, facecolor='none', edgecolor='gray', linewidth=1),
                                    medianprops=dict(visible=False),
                                    whiskerprops=dict(visible=False),
                                    capprops=dict(visible=False))
    
    ax2.set_xlabel('Number of Workers', fontsize=12)
    ax2.set_ylabel('Maximum RSS (MB)', fontsize=12)
    ax2.set_title('Maximum RSS vs Number of Workers', fontsize=14, fontweight='bold')
    ax2.legend(title='Images', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'rss_scalability.png', bbox_inches='tight')
    print(f"Saved: {output_path / 'rss_scalability.png'}")
    plt.close()


def plot_heatmaps(summary_df: pd.DataFrame, output_path: Path):
    """Create heatmaps showing scalability across both dimensions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create pivot tables using mean values
    wall_time_pivot = summary_df.pivot_table(
        values='wall_time_mean',
        index='num_images',
        columns='num_workers',
        aggfunc='mean'
    )
    
    ram_pivot = summary_df.pivot_table(
        values='peak_ram_mb_mean',
        index='num_images',
        columns='num_workers',
        aggfunc='mean'
    )
    
    rss_pivot = summary_df.pivot_table(
        values='max_rss_mb_mean',
        index='num_images',
        columns='num_workers',
        aggfunc='mean'
    )
    
    # Plot heatmaps
    sns.heatmap(
        wall_time_pivot,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        ax=axes[0],
        cbar_kws={'label': 'Wall Time (s)'}
    )
    axes[0].set_title('Wall Time Heatmap', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Workers', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    
    sns.heatmap(
        ram_pivot,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        ax=axes[1],
        cbar_kws={'label': 'Peak RAM (MB)'}
    )
    axes[1].set_title('Peak RAM Usage Heatmap', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Workers', fontsize=12)
    axes[1].set_ylabel('Number of Images', fontsize=12)
    
    sns.heatmap(
        rss_pivot,
        annot=True,
        fmt='.0f',
        cmap='Greens',
        ax=axes[2],
        cbar_kws={'label': 'Max RSS (MB)'}
    )
    axes[2].set_title('Maximum RSS Heatmap', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Number of Workers', fontsize=12)
    axes[2].set_ylabel('Number of Images', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scalability_heatmaps.png', bbox_inches='tight')
    print(f"Saved: {output_path / 'scalability_heatmaps.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate scalability plots from benchmark results'
    )
    parser.add_argument(
        'results_csv',
        type=str,
        help='Path to scalability_results.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as CSV directory)'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {csv_path}...")
    raw_df, summary_df = load_results(csv_path)
    
    if len(raw_df) == 0:
        print("Error: No valid results found")
        sys.exit(1)
    
    print(f"Loaded {len(raw_df)} benchmark results")
    print(f"\nConfigurations tested:")
    print(f"  Images: {sorted(summary_df['num_images'].unique())}")
    print(f"  Workers: {sorted(summary_df['num_workers'].unique())}")
    if 'repeat' in raw_df.columns:
        print(f"  Repeats per configuration: {raw_df.groupby(['num_images', 'num_workers']).size().iloc[0]}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_wall_time(raw_df, summary_df, output_dir)
    plot_ram_usage(raw_df, summary_df, output_dir)
    plot_rss(raw_df, summary_df, output_dir)
    plot_heatmaps(summary_df, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

