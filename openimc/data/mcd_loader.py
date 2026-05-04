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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import re
import threading

import numpy as np

_ACQ_ID_PARSE = re.compile(r"^slide_(\d+)_acq_(\d+)$")


def _ion_stripe_class(acquisition: object) -> str:
    """MCD ion-list bounds from metadata: ``nonempty``, ``empty``, or ``unknown`` if unreadable."""

    try:
        md = acquisition.metadata  # type: ignore[union-attr]
        if not isinstance(md, dict):
            return "unknown"
        a, b = int(md["DataStartOffset"]), int(md["DataEndOffset"])
        return "nonempty" if a != b else "empty"
    except (KeyError, ValueError, TypeError, AttributeError):
        return "unknown"


def _slide_idx_from_acq_id(acq_id: str) -> Optional[int]:
    m = _ACQ_ID_PARSE.match(acq_id)
    return int(m.group(1)) if m else None


def _acq_id_sort_tuple(acq_id: str) -> Tuple[int, int]:
    m = _ACQ_ID_PARSE.match(acq_id)
    return (int(m.group(1)), int(m.group(2))) if m else (10**9, 10**9)


def _description_group_key(slide_idx: int, acq: object, _slide: object) -> Optional[Tuple[int, str]]:
    text = ""
    md = getattr(acq, "metadata", None)
    if isinstance(md, dict):
        d = md.get("Description")
        if d is not None:
            text = str(d).strip().lower()
    if not text:
        dr = getattr(acq, "description", None)
        if dr is not None:
            text = str(dr).strip().lower()
    return (slide_idx, text) if text else None


_HAVE_READIMC = False
try:
    from readimc import MCDFile as McdFile  # type: ignore
    _HAVE_READIMC = True
except Exception:
    _HAVE_READIMC = False


@dataclass
class AcquisitionInfo:
    id: str
    name: str
    well: Optional[str]
    size: Tuple[Optional[int], Optional[int]]  # (H, W)
    channels: List[str]
    channel_metals: List[str]
    channel_labels: List[str]
    metadata: Dict
    source_file: Optional[str] = None  # Path to the source .mcd file


class MCDLoader:
    """Loader for IMC .mcd files using the readimc library with f.read_acquisition() method.

    All acquisitions are indexed first. Duplicate entries that share the same ROI label within
    one slide then collapse according to ion-stripe metadata: when any carries a nonempty
    ``DataStartOffset``≠``DataEndOffset`` range, placeholders are discarded and ties among nonempty
    entries use brightest ``readimc`` data. Without usable metadata—or no nonempty entry—explicit
    empty placeholders are skipped in favor of the first remaining acquisition order from the slide.

    Unlabeled lone acquisitions always remain (including when metadata cannot be interpreted).

    Note: The readimc library's McdFile context manager is not thread-safe.
    This class uses a lock to serialize access to file operations.
    """

    def __init__(self):
        if not _HAVE_READIMC:
            raise RuntimeError("readimc is not installed. Run: pip install readimc")
        self.mcd: Optional[McdFile] = None
        self._acq_map: Dict[str, object] = {}
        self._acq_channels: Dict[str, List[str]] = {}
        self._acq_channel_metals: Dict[str, List[str]] = {}
        self._acq_channel_labels: Dict[str, List[str]] = {}
        self._acq_size: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        self._acq_name: Dict[str, str] = {}
        self._acq_well: Dict[str, Optional[str]] = {}
        self._acq_metadata: Dict[str, Dict] = {}
        # Lock to serialize access to file operations (readimc is not thread-safe)
        self._file_lock = threading.Lock()

    def open(self, path: str):
        """Open an .mcd file."""
        self.mcd = McdFile(path)
        if hasattr(self.mcd, "open"):
            self.mcd.open()
        self._index()

    def _index(self):
        """Index all acquisitions in the .mcd file."""
        self._acq_map.clear()
        self._acq_channels.clear()
        self._acq_channel_metals.clear()
        self._acq_channel_labels.clear()
        self._acq_size.clear()
        self._acq_name.clear()
        self._acq_well.clear()
        self._acq_metadata.clear()

        slides = getattr(self.mcd, "slides", [])

        if slides:
            for slide_idx, slide in enumerate(slides):
                for acq_idx, acq in enumerate(getattr(slide, "acquisitions", [])):
                    acq_id = f"slide_{slide_idx}_acq_{acq_idx}"

                    name = getattr(acq, "name", f"Slide {slide_idx + 1} Acquisition {acq_idx + 1}")

                    well = getattr(acq, "well", getattr(slide, "well", None))
                    if well is None and hasattr(acq, "metadata"):
                        metadata = acq.metadata
                        if isinstance(metadata, dict) and 'Description' in metadata:
                            well = metadata['Description']

                    channel_metals = getattr(acq, "channel_names", [])
                    channel_labels = getattr(acq, "channel_labels", [])

                    channels: List[str] = []
                    for i, (metal, label) in enumerate(zip(channel_metals, channel_labels)):
                        if label and metal:
                            channels.append(f"{label}_{metal}")
                        elif label:
                            channels.append(label)
                        elif metal:
                            channels.append(metal)
                        else:
                            channels.append(f"Channel_{i+1}")

                    try:
                        H = getattr(acq, "height", None) or getattr(acq, "rows", None)
                        W = getattr(acq, "width", None) or getattr(acq, "cols", None)
                        size = (int(H), int(W)) if H and W else (None, None)
                    except Exception:
                        size = (None, None)

                    metadata = getattr(acq, "metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}

                    self._acq_map[acq_id] = acq
                    self._acq_channels[acq_id] = channels
                    self._acq_channel_metals[acq_id] = channel_metals
                    self._acq_channel_labels[acq_id] = channel_labels
                    self._acq_size[acq_id] = size
                    self._acq_name[acq_id] = name
                    self._acq_well[acq_id] = well
                    self._acq_metadata[acq_id] = metadata

        self._collapse_duplicate_roi_labels()

        if not self._acq_map:
            raise RuntimeError("No acquisitions found in this .mcd file.")

    def _collapse_duplicate_roi_labels(self) -> None:
        """Merge duplicate acquisitions on the same slide that share ROI text (see class docstring)."""

        slides = getattr(self.mcd, "slides", []) if self.mcd else []
        if not slides or len(self._acq_map) < 2:
            return

        groups: Dict[Tuple[int, str], List[str]] = {}
        for aid in list(self._acq_map.keys()):
            si = _slide_idx_from_acq_id(aid)
            if si is None or si < 0 or si >= len(slides):
                continue
            gkey = _description_group_key(si, self._acq_map[aid], slides[si])
            if gkey is None:
                continue
            groups.setdefault(gkey, []).append(aid)

        losers: List[str] = []
        for ids in groups.values():
            if len(ids) < 2:
                continue
            keeper = self._keeper_for_duplicate_roi_group(ids)
            losers.extend(i for i in ids if i != keeper)

        for aid in losers:
            self._acq_map.pop(aid, None)
            self._acq_channels.pop(aid, None)
            self._acq_channel_metals.pop(aid, None)
            self._acq_channel_labels.pop(aid, None)
            self._acq_size.pop(aid, None)
            self._acq_name.pop(aid, None)
            self._acq_well.pop(aid, None)
            self._acq_metadata.pop(aid, None)

    def _keeper_for_duplicate_roi_group(self, ids: List[str]) -> str:
        classes = {_i: _ion_stripe_class(self._acq_map[_i]) for _i in ids}
        nonempty_ids = [_i for _i in ids if classes[_i] == "nonempty"]
        if nonempty_ids:
            if len(nonempty_ids) == 1:
                return nonempty_ids[0]
            return self._brightest_acq_id(nonempty_ids)

        order = sorted(ids, key=_acq_id_sort_tuple)
        prioritized = [_i for _i in order if classes[_i] != "empty"]
        return prioritized[0] if prioritized else order[0]

    def _brightest_acq_id(self, ids: List[str]) -> str:
        ids_sorted = sorted(ids, key=_acq_id_sort_tuple)
        best_id = ids_sorted[0]
        best_peak = -1.0
        with self._file_lock:
            try:
                with self.mcd as f:  # type: ignore[misc]
                    for aid in ids_sorted:
                        data = f.read_acquisition(self._acq_map[aid])
                        pk = float(np.nanmax(np.asarray(data)))
                        if pk > best_peak:
                            best_peak = pk
                            best_id = aid
            except (OSError, ValueError, TypeError):
                return ids_sorted[0]
        return best_id

    def list_acquisitions(self, source_file: Optional[str] = None) -> List[AcquisitionInfo]:
        """List all acquisitions in the .mcd file.
        
        Args:
            source_file: Optional path to the source .mcd file to include in AcquisitionInfo
        """
        infos: List[AcquisitionInfo] = []
        for acq_id in self._acq_map:
            infos.append(
                AcquisitionInfo(
                    id=acq_id,
                    name=self._acq_name.get(acq_id, acq_id),
                    well=self._acq_well.get(acq_id),
                    size=self._acq_size.get(acq_id, (None, None)),
                    channels=self._acq_channels.get(acq_id, []),
                    channel_metals=self._acq_channel_metals.get(acq_id, []),
                    channel_labels=self._acq_channel_labels.get(acq_id, []),
                    metadata=self._acq_metadata.get(acq_id, {}),
                    source_file=source_file,
                )
            )
        return infos

    def get_channels(self, acq_id: str) -> List[str]:
        """Get channel names for a specific acquisition."""
        return self._acq_channels[acq_id]

    def get_image(self, acq_id: str, channel: str) -> np.ndarray:
        """Get image data for a specific acquisition and channel.
        
        This method is thread-safe - uses a lock to serialize file access.
        """
        if self.mcd is None:
            raise RuntimeError("MCD file is not open. Call open() first.")
        
        acq = self._acq_map[acq_id]
        channels = self._acq_channels[acq_id]
        if channel not in channels:
            raise ValueError(f"Channel '{channel}' not found in acquisition {acq_id}.")
        ch_idx = channels.index(channel)
        
        # Serialize access to file operations (readimc is not thread-safe)
        with self._file_lock:
            try:
                with self.mcd as f:  # type: ignore
                    img = f.read_acquisition(acq)
                    img = np.transpose(img, (1, 2, 0))
                    return img[..., ch_idx]
            except OSError as e:
                if e.errno == 9:  # Bad file descriptor
                    raise RuntimeError(
                        f"File descriptor error when reading acquisition {acq_id}, channel {channel}. "
                        "The file may have been closed or is in an invalid state. "
                        "Try reloading the file."
                    ) from e
                raise

    def get_all_channels(self, acq_id: str) -> np.ndarray:
        """Get all channels for a specific acquisition as a 3D array (H, W, C).
        
        This method is thread-safe - uses a lock to serialize file access.
        """
        if self.mcd is None:
            raise RuntimeError("MCD file is not open. Call open() first.")
        
        acq = self._acq_map[acq_id]
        
        # Serialize access to file operations (readimc is not thread-safe)
        with self._file_lock:
            try:
                with self.mcd as f:  # type: ignore
                    img = f.read_acquisition(acq)
                    img = np.transpose(img, (1, 2, 0))
                    return img
            except OSError as e:
                if e.errno == 9:  # Bad file descriptor
                    raise RuntimeError(
                        f"File descriptor error when reading acquisition {acq_id}. "
                        "The file may have been closed or is in an invalid state. "
                        "Try reloading the file."
                    ) from e
                raise

    def close(self):
        """Close the .mcd file."""
        if self.mcd and hasattr(self.mcd, "close"):
            try:
                self.mcd.close()
            except OSError as e:
                # Ignore errors when closing (file may already be closed)
                if e.errno != 9:  # Only ignore Bad file descriptor, raise others
                    raise
        self.mcd = None



