"""Lazy (per-worker) recording-window dataset without crossing activity boundaries."""
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from crossguard_fi.config import CONTRACT
from crossguard_fi.data.manifest import Recording
from crossguard_fi.data.normalization import ChannelNormalizer
from crossguard_fi.data.validation import load_recording


class CSIDataset(Dataset):
    def __init__(self, root: Path, records: tuple[Recording, ...],
                 normalizer: ChannelNormalizer | None = None, cache_size: int = 2):
        self.root = Path(root)
        self.records = records
        self.normalizer = normalizer
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.windows: list[tuple[int, int]] = []
        # Validate once at construction, then lazy-load actual arrays in workers.
        for ri, record in enumerate(records):
            csi, _ = load_recording(self.root, record)
            for offset in range(0, len(csi) - CONTRACT.window_samples + 1,
                                CONTRACT.stride_samples):
                self.windows.append((ri, offset))
        if not self.windows:
            raise ValueError("No windows generated")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int):
        ri, offset = self.windows[index]
        rec = self.records[ri]
        if rec.recording_id in self._cache:
            csi = self._cache[rec.recording_id]
            self._cache.move_to_end(rec.recording_id)
        else:
            csi, _ = load_recording(self.root, rec)
            if self.cache_size > 0:
                self._cache[rec.recording_id] = csi
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        window = csi[offset:offset + CONTRACT.window_samples]
        if self.normalizer is not None:
            window = self.normalizer.transform(window)
        return (torch.as_tensor(np.array(window, dtype=np.float32)),
                torch.tensor(rec.label, dtype=torch.long))
