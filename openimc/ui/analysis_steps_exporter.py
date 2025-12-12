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
Analysis Steps Exporter for OpenIMC

This module exports a human-readable text file documenting all analysis steps
performed on the dataset, suitable for inclusion in methods sections.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from openimc.utils.logger import get_logger


class AnalysisStepsExporter:
    """
    Exports analysis steps to a formatted text file.
    """
    
    def __init__(self):
        self.logger = get_logger()
    
    def export_analysis_steps(
        self,
        output_path: str,
        include_timestamps: bool = True,
        include_parameters: bool = True,
        session_start_time: Optional[datetime] = None
    ) -> bool:
        """
        Export analysis steps to a text file.
        If session_start_time is provided, only exports operations from that session.
        
        Args:
            output_path: Path to output text file
            include_timestamps: Whether to include timestamps
            include_parameters: Whether to include detailed parameters
            session_start_time: If provided, only export entries after this time
            
        Returns:
            True if successful, False otherwise
        """
        try:
            log_file = self.logger.get_log_file_path()
            if not Path(log_file).exists():
                return False
            
            # Read log entries, filtering by session if provided
            entries = self._read_log_entries(log_file, session_start_time=session_start_time)
            
            if not entries:
                return False
            
            # Group entries by type
            grouped = self._group_entries(entries)
            
            # Generate formatted text
            text = self._format_analysis_steps(grouped, include_timestamps, include_parameters)
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(text)
            
            return True
            
        except Exception as e:
            print(f"Error exporting analysis steps: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _read_log_entries(self, log_file: str, session_start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Read log entries from the log file.
        If session_start_time is provided, only returns entries from that session.
        """
        entries = []
        
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Skip metadata entries
                    if entry.get("type") == "log_metadata":
                        continue
                    
                    # Filter by session start time if provided
                    if session_start_time:
                        entry_timestamp_str = entry.get("timestamp")
                        if entry_timestamp_str:
                            try:
                                # Parse timestamp (handle ISO format with or without timezone)
                                entry_timestamp = datetime.fromisoformat(entry_timestamp_str.replace('Z', '+00:00'))
                                # Remove timezone for comparison if present
                                if entry_timestamp.tzinfo:
                                    entry_timestamp = entry_timestamp.replace(tzinfo=None)
                                session_start = session_start_time.replace(tzinfo=None) if session_start_time.tzinfo else session_start_time
                                
                                # Only include entries from this session
                                if entry_timestamp < session_start:
                                    continue
                            except (ValueError, AttributeError):
                                # If timestamp parsing fails, include the entry (better to include than exclude)
                                pass
                    
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        
        return entries
    
    def _group_entries(self, entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group log entries by type."""
        grouped = {}
        
        for entry in entries:
            entry_type = entry.get("type", "unknown")
            if entry_type not in grouped:
                grouped[entry_type] = []
            grouped[entry_type].append(entry)
        
        return grouped
    
    def _format_analysis_steps(
        self,
        grouped: Dict[str, List[Dict[str, Any]]],
        include_timestamps: bool,
        include_parameters: bool
    ) -> str:
        """Format grouped entries into readable text."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("OpenIMC Analysis Steps")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("This document describes all analysis steps performed on the dataset.")
        lines.append("")
        
        # Order of sections
        section_order = [
            "segmentation",
            "feature_extraction",
            "spillover_matrix",
            "batch_correction",
            "clustering",
            "class_annotation",
            "spatial_analysis",
            "qc_analysis",
            "pixel_correlation",
            "deconvolution",
            "export",
            "gating"
        ]
        
        # Process each section
        for section_type in section_order:
            if section_type in grouped:
                lines.extend(self._format_section(section_type, grouped[section_type], include_timestamps, include_parameters))
                lines.append("")
        
        # Add any remaining sections
        for section_type in sorted(grouped.keys()):
            if section_type not in section_order:
                lines.extend(self._format_section(section_type, grouped[section_type], include_timestamps, include_parameters))
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_section(
        self,
        section_type: str,
        entries: List[Dict[str, Any]],
        include_timestamps: bool,
        include_parameters: bool
    ) -> List[str]:
        """Format a section of entries."""
        lines = []
        
        # Section header
        section_title = section_type.replace("_", " ").title()
        lines.append("-" * 80)
        lines.append(f"{section_title}")
        lines.append("-" * 80)
        lines.append("")
        
        for i, entry in enumerate(entries, 1):
            # Entry header
            operation = entry.get("operation", "unknown")
            lines.append(f"{i}. {operation.replace('_', ' ').title()}")
            
            if include_timestamps and "timestamp" in entry:
                timestamp = entry["timestamp"]
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    lines.append(f"   Date/Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    lines.append(f"   Date/Time: {timestamp}")
            
            # Parameters
            params = entry.get("parameters", {})
            if params and include_parameters:
                lines.append("   Parameters:")
                for key, value in params.items():
                    if key == "features_extracted" and isinstance(value, list):
                        lines.append(f"      - {key}: {len(value)} features")
                    elif key == "features_used" and isinstance(value, list):
                        lines.append(f"      - {key}: {len(value)} features")
                    elif isinstance(value, dict):
                        lines.append(f"      - {key}:")
                        for sub_key, sub_value in value.items():
                            lines.append(f"        * {sub_key}: {sub_value}")
                    elif isinstance(value, list) and len(value) > 5:
                        lines.append(f"      - {key}: {len(value)} items")
                    else:
                        lines.append(f"      - {key}: {value}")
            
            # Acquisitions
            acquisitions = entry.get("acquisitions", [])
            if acquisitions:
                lines.append(f"   Acquisitions: {len(acquisitions)} acquisition(s)")
                if len(acquisitions) <= 5:
                    for acq in acquisitions:
                        lines.append(f"      - {acq}")
            
            # Source file
            source_file = entry.get("source_file")
            if source_file:
                lines.append(f"   Source file: {source_file}")
            
            # Output path
            output_path = entry.get("output_path")
            if output_path:
                lines.append(f"   Output: {output_path}")
            
            # Notes
            notes = entry.get("notes")
            if notes:
                lines.append(f"   Notes: {notes}")
            
            lines.append("")
        
        return lines
    
    def export_from_main_window(self, main_window, output_path: str) -> bool:
        """
        Export analysis steps with additional context from MainWindow.
        Only exports operations from the current session.
        
        Args:
            main_window: MainWindow instance
            output_path: Path to output text file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get session start time to filter log entries
            session_start_time = None
            if hasattr(main_window, 'session_start_time'):
                session_start_time = main_window.session_start_time
            
            # Export only session-specific operations
            success = self.export_analysis_steps(
                output_path, 
                include_timestamps=True, 
                include_parameters=True,
                session_start_time=session_start_time
            )
            
            if not success:
                return False
            
            # Read the exported file and enhance it
            with open(output_path, 'r') as f:
                content = f.read()
            
            # Add dataset information
            enhanced_content = self._add_dataset_info(content, main_window)
            
            # Write enhanced content
            with open(output_path, 'w') as f:
                f.write(enhanced_content)
            
            return True
            
        except Exception as e:
            print(f"Error exporting analysis steps: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_dataset_info(self, content: str, main_window) -> str:
        """Add dataset information to the content."""
        lines = content.split("\n")
        
        # Find the header section
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith("This document describes"):
                header_end = i + 1
                break
        
        # Insert dataset info
        dataset_info = []
        dataset_info.append("")
        dataset_info.append("Dataset Information")
        dataset_info.append("-" * 80)
        
        # Get file paths - list all files explicitly
        source_files = []
        if hasattr(main_window, 'acq_to_file') and main_window.acq_to_file:
            unique_files = set(main_window.acq_to_file.values())
            source_files = sorted(list(unique_files))
        
        # If no files from acq_to_file, try current_path
        if not source_files and hasattr(main_window, 'current_path') and main_window.current_path:
            source_files = [main_window.current_path]
        
        # List all source files
        if source_files:
            if len(source_files) == 1:
                dataset_info.append(f"Source file: {source_files[0]}")
            else:
                dataset_info.append(f"Source files ({len(source_files)} files):")
                for file_path in source_files:
                    dataset_info.append(f"  - {file_path}")
        else:
            dataset_info.append("Source file(s): Not available")
        
        # Get acquisitions
        if hasattr(main_window, 'acquisitions') and main_window.acquisitions:
            dataset_info.append(f"Number of acquisitions: {len(main_window.acquisitions)}")
        
        # Get features info
        if hasattr(main_window, 'feature_dataframe') and main_window.feature_dataframe is not None:
            df = main_window.feature_dataframe
            dataset_info.append(f"Feature data: {len(df)} cells, {len(df.columns)} features")
        
        if hasattr(main_window, 'batch_corrected_dataframe') and main_window.batch_corrected_dataframe is not None:
            dataset_info.append("Batch correction: Applied")
        
        # Get masks info
        if hasattr(main_window, 'mask_manager') and main_window.mask_manager:
            mask_ids = main_window.mask_manager.get_all_mask_ids()
            dataset_info.append(f"Segmentation masks: {len(mask_ids)} acquisition(s)")
        
        dataset_info.append("")
        
        # Insert into content
        new_lines = lines[:header_end] + dataset_info + lines[header_end:]
        return "\n".join(new_lines)

