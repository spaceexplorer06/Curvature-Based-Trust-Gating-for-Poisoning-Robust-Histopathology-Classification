"""
BreaKHis dataset loader with patient-wise train/val/test splitting.

BreaKHis is distributed with a directory layout roughly like:

    BreaKHis_v1/histology_slides/breast/
        benign/SOB/<tumor_type>/<patient_id>/<magnification>/*.png
        malignant/SOB/<tumor_type>/<patient_id>/<magnification>/*.png

Filenames encode class, tumor subtype, patient/slide id, and magnification, e.g.:
    SOB_B_TA-14-4659-40-001.png
    SOB_M_DC-14-2523-100-014.png

We do NOT split by image, because many images per patient/slide are highly
correlated (same tissue, adjacent crops) — an image-wise random split leaks
patient identity across train/test and inflates reported accuracy. We split
by patient id instead, which is the standard practice for BreaKHis.
"""

import os
import re
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset

# Matches e.g. SOB_B_TA-14-4659-40-001.png -> class=B, subtype=TA, patient=14-4659, mag=40
FILENAME_RE = re.compile(
    r"SOB_([BM])_([A-Z]+)-(\d+-\d+)-(\d+)-(\d+)\.png$", re.IGNORECASE
)

CLASS_MAP = {"B": 0, "M": 1}  # benign=0, malignant=1


@dataclass
class Sample:
    path: str
    label: int
    patient_id: str
    magnification: str


def index_dataset(root_dir: str) -> List[Sample]:
    """
    Walk root_dir and build a list of Sample objects. root_dir should point at
    the folder containing benign/ and malignant/ subfolders (or any ancestor
    of them — we glob recursively for *.png).
    """
    samples: List[Sample] = []
    all_pngs = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    if not all_pngs:
        raise FileNotFoundError(
            f"No .png files found under {root_dir}. Check the path points at "
            f"your extracted BreaKHis_v1 folder."
        )

    unmatched = 0
    for p in all_pngs:
        fname = os.path.basename(p)
        m = FILENAME_RE.search(fname)
        if not m:
            unmatched += 1
            continue
        cls, subtype, patient_id, mag, _idx = m.groups()
        samples.append(
            Sample(
                path=p,
                label=CLASS_MAP[cls.upper()],
                patient_id=patient_id,
                magnification=mag,
            )
        )

    if unmatched:
        print(
            f"[dataset] Warning: {unmatched} files did not match the expected "
            f"BreaKHis filename pattern and were skipped."
        )
    if not samples:
        raise RuntimeError(
            "No samples parsed. The BreaKHis filename format may differ from "
            "what FILENAME_RE expects -- print a few filenames and adjust the regex."
        )
    return samples


def patientwise_split(
    samples: List[Sample],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Split by unique patient_id so no patient's images appear in more than one
    split. Stratifies loosely by class at the patient level.
    """
    rng = random.Random(seed)

    patients_by_class: Dict[int, List[str]] = {0: [], 1: []}
    patient_label: Dict[str, int] = {}
    for s in samples:
        if s.patient_id not in patient_label:
            patient_label[s.patient_id] = s.label
            patients_by_class[s.label].append(s.patient_id)

    train_ids, val_ids, test_ids = set(), set(), set()
    for cls, pids in patients_by_class.items():
        pids = pids[:]
        rng.shuffle(pids)
        n = len(pids)
        n_val = max(1, int(round(n * val_frac)))
        n_test = max(1, int(round(n * test_frac)))
        n_train = n - n_val - n_test
        train_ids.update(pids[:n_train])
        val_ids.update(pids[n_train:n_train + n_val])
        test_ids.update(pids[n_train + n_val:])

    train = [s for s in samples if s.patient_id in train_ids]
    val = [s for s in samples if s.patient_id in val_ids]
    test = [s for s in samples if s.patient_id in test_ids]

    print(
        f"[dataset] Patients: train={len(train_ids)} val={len(val_ids)} "
        f"test={len(test_ids)} | Images: train={len(train)} val={len(val)} "
        f"test={len(test)}"
    )
    return train, val, test


class BreakHisDataset(Dataset):
    """
    Wraps a list of Sample objects. `label_override` is an optional dict
    mapping sample index -> flipped label, used by the poisoning module
    to inject label-flip poisoning without duplicating the dataset.
    """

    def __init__(self, samples: List[Sample], transform=None, label_override: Dict[int, int] = None):
        self.samples = samples
        self.transform = transform
        self.label_override = label_override or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label_override.get(idx, s.label)
        return img, torch.tensor(label, dtype=torch.long), idx