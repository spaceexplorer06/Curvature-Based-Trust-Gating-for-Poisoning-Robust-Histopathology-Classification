"""
PatchCamelyon (PCam) dataset loader.

PCam ships as six HDF5 files (standard distribution from
github.com/basveeling/pcam), one x/y pair per official split:

    camelyonpatch_level_2_split_train_x.h5   (x: uint8, N x 96 x 96 x 3)
    camelyonpatch_level_2_split_train_y.h5   (y: uint8, N x 1 x 1 x 1)
    camelyonpatch_level_2_split_valid_x.h5 / _y.h5
    camelyonpatch_level_2_split_test_x.h5  / _y.h5

NOTE: verify these exact filenames/dataset-internal key names against
whatever release you actually downloaded -- naming has varied slightly
across PCam mirrors/versions. This loader glob-matches on the
train/valid/test + x/y substrings rather than requiring an exact
filename, and reads the first dataset key found in each h5 file, so
minor naming variations should still work; if your files use different
internal HDF5 dataset key names, check with `h5py.File(path).keys()`.

Unlike BreaKHis, PCam ships pre-defined train/valid/test splits that
are already constructed to avoid whole-slide-image (WSI) leakage across
splits -- there is no patient/WSI grouping metadata in the public
patch-level release for us to split by ourselves the way
`patientwise_split` does for BreaKHis. We therefore use the official
splits as-is rather than re-deriving our own; each 15-run-experiments
"seed" for this dataset varies model initialization, batch order, and
poisoning-flip selection, but not which patches land in which split
(unlike BreaKHis, where --seed also controls the patient-wise split
itself). This is a methodological difference worth noting explicitly
if comparing seed-to-seed variance across the two datasets.

PCam is a balanced binary task (tumor / no-tumor, roughly 50/50) by
construction, unlike BreaKHis's ~31/69 benign/malignant imbalance --
a useful contrast when interpreting cross-dataset results, since
several of the failure modes observed on BreaKHis (e.g. single-class
collapse under poisoning) may interact with class imbalance in ways
that a balanced second dataset can help disentangle.
"""

import os
import glob
import gzip
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PCamSample:
    """Mirrors dataset_loader.Sample's interface (label, indexable) so
    poison.make_label_flip_overrides works unchanged on PCam data.
    `path` here is not a filesystem path but the (h5_file_path, index)
    location of the patch within its split's x-file; `patient_id` and
    `magnification` are not meaningful for PCam and are left as None."""
    path: Tuple[str, int]
    label: int
    patient_id: None = None
    magnification: None = None


import hashlib


def _content_fingerprint(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Cheap fingerprint (file size + hash of the first 1MB) used to key
    the decompression cache to actual file CONTENT rather than just the
    current filename. A plain filename-based cache (e.g. always writing
    to 'test_x.h5') would silently reuse a stale decompressed file left
    over from before a source .gz was renamed/replaced -- exactly the
    scenario that happens when fixing a train/valid/test naming mixup
    after an MD5 checksum check, and which produces a plausible-looking
    but wrong result (desynced images/labels) rather than a crash."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        chunk = f.read(chunk_size)
    h = hashlib.md5(chunk).hexdigest()[:10]
    return f"{size}_{h}"


def _maybe_decompress(path: str) -> str:
    """Decompresses a .gz file to a content-fingerprinted sibling path,
    reusing an existing decompressed output only if it was produced from
    a .gz with the exact same size+content-hash fingerprint (see
    _content_fingerprint). This is deliberately more conservative than a
    plain filename-based cache."""
    if not path.endswith(".gz"):
        return path
    fp = _content_fingerprint(path)
    base = path[:-3]  # strip ".gz", e.g. ".../test_x.h5"
    root, ext = os.path.splitext(base)
    out_path = f"{root}.{fp}{ext}"
    if os.path.exists(out_path):
        return out_path
    print(f"[pcam] Decompressing {path} -> {out_path} "
          f"(one-time; cached for future runs)...")
    with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
    print(f"[pcam] Decompression finished: {out_path}")
    return out_path


def _pick_and_decompress(candidates: List[str]) -> str:
    """Given candidate paths for one file, prefer decompressing from a .gz
    candidate if one is present, deriving the exact expected output path
    from that specific .gz file and only skipping decompression if that
    exact output already exists. We deliberately do NOT prefer an
    already-existing plain .h5 candidate found by the split-name glob
    independent of which .gz it came from: if source .gz files are ever
    renamed (e.g. fixing a train/valid/test naming mixup after checking
    MD5 checksums against the official values), a stale plain .h5 left
    over from decompressing the OLD .gz under the same split name would
    otherwise be silently reused, desyncing images from labels without
    any error -- exactly the kind of bug that produces a plausible-looking
    but wrong result (e.g. chance-level test performance despite good
    validation performance) rather than a crash."""
    gz_candidates = [c for c in candidates if c.endswith(".gz")]
    if gz_candidates:
        return _maybe_decompress(gz_candidates[0])
    plain = [c for c in candidates if not c.endswith(".gz")]
    if plain:
        return plain[0]
    raise FileNotFoundError(f"No usable candidates in {candidates}")


def _find_split_files(root_dir: str, split: str) -> Tuple[str, str]:
    """Locates the x and y h5(.gz) files for one split (train/valid/test)
    under root_dir, tolerant of minor filename variations across PCam
    releases and of whether files have been pre-decompressed."""
    x_candidates = sorted(glob.glob(os.path.join(root_dir, f"**/*{split}*x*.h5*"), recursive=True))
    y_candidates = sorted(glob.glob(os.path.join(root_dir, f"**/*{split}*y*.h5*"), recursive=True))
    if not x_candidates or not y_candidates:
        raise FileNotFoundError(
            f"Could not find PCam '{split}' x/y .h5 or .h5.gz files under {root_dir}. "
            f"Expected filenames containing '{split}', 'x'/'y', and '.h5' "
            f"(e.g. camelyonpatch_level_2_split_{split}_x.h5.gz). "
            f"Found x candidates: {x_candidates}, y candidates: {y_candidates}."
        )
    x_path = _pick_and_decompress(x_candidates)
    y_path = _pick_and_decompress(y_candidates)
    return x_path, y_path


def _h5_dataset_key(h5_path: str) -> str:
    """Returns the first (and normally only) dataset key inside an h5 file.
    PCam's official releases use 'x' and 'y', but we read the key
    dynamically rather than hardcoding it, in case a mirror uses different
    internal naming."""
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
    if not keys:
        raise ValueError(f"{h5_path} contains no datasets.")
    if len(keys) > 1:
        print(f"[pcam] Warning: {h5_path} contains multiple datasets "
              f"{keys}; using '{keys[0]}'.")
    return keys[0]


def index_pcam_split(root_dir: str, split: str) -> List[PCamSample]:
    """Builds a list of PCamSample for one official split (train/valid/test).
    Only reads the y (label) file into memory -- x stays on disk and is
    read lazily per-sample in PCamH5Dataset.__getitem__, since the x files
    are large (tens of thousands of 96x96x3 patches)."""
    x_path, y_path = _find_split_files(root_dir, split)
    y_key = _h5_dataset_key(y_path)
    with h5py.File(y_path, "r") as f:
        labels = np.asarray(f[y_key]).reshape(-1)  # squeeze (N,1,1,1) -> (N,)
    labels = labels.astype(int)

    x_key = _h5_dataset_key(x_path)
    with h5py.File(x_path, "r") as f:
        n_x = f[x_key].shape[0]
    if n_x != len(labels):
        raise ValueError(
            f"PCam '{split}' split: x file has {n_x} samples but y file has "
            f"{len(labels)} labels ({x_path} vs {y_path}). These files do not "
            f"correspond to each other -- do not proceed with training on "
            f"this split until resolved. This usually means the wrong x/y "
            f"pairing was picked up (e.g. after a manual file rename); "
            f"double-check MD5 checksums against the official PCam values."
        )

    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    print(f"[pcam] Split '{split}': {len(labels)} patches "
          f"({n_neg} class-0 / {n_pos} class-1) from {os.path.basename(x_path)}")

    return [PCamSample(path=(x_path, i), label=int(labels[i])) for i in range(len(labels))]


def load_pcam_splits(root_dir: str) -> Tuple[List[PCamSample], List[PCamSample], List[PCamSample]]:
    """Loads PCam's official train/valid/test splits. Unlike
    patientwise_split for BreaKHis, this does NOT take a --seed argument --
    the official splits are fixed and already leakage-safe; --seed still
    affects model init, batch order, and which training samples get
    poisoned (see module docstring)."""
    train = index_pcam_split(root_dir, "train")
    val = index_pcam_split(root_dir, "valid")
    test = index_pcam_split(root_dir, "test")
    return train, val, test


class PCamH5Dataset(Dataset):
    """
    Wraps a list of PCamSample. Matches BreakHisDataset's __getitem__
    contract exactly: (transformed_image, label_tensor, idx). Opens its
    own h5py.File handle lazily per-worker (h5py file handles are not
    safely shared across DataLoader worker processes if opened in the
    main process before forking).
    """

    def __init__(self, samples: List[PCamSample], transform=None,
                 label_override: Dict[int, int] = None):
        self.samples = samples
        self.transform = transform
        self.label_override = label_override or {}
        self._h5_handles = {}  # opened lazily per worker, keyed by file path

    def _get_handle(self, h5_path: str):
        # Re-open per worker process: a handle inherited via fork from the
        # main process is not safe to read from concurrently in other
        # worker processes.
        pid = os.getpid()
        key = (h5_path, pid)
        if key not in self._h5_handles:
            self._h5_handles[key] = h5py.File(h5_path, "r")
        return self._h5_handles[key]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        h5_path, row = s.path
        f = self._get_handle(h5_path)
        x_key = _h5_dataset_key(h5_path) if not hasattr(self, "_x_key") else self._x_key
        self._x_key = x_key
        img_arr = np.asarray(f[x_key][row])  # (96, 96, 3) uint8

        from PIL import Image
        img = Image.fromarray(img_arr, mode="RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label_override.get(idx, s.label)
        return img, torch.tensor(label, dtype=torch.long), idx