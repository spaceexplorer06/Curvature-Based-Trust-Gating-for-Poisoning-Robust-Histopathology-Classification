"""
Dataset module wrapper - re-exports all symbols from dataset_loader for clean imports.
"""

from dataset_loader import (
    Sample,
    index_dataset,
    patientwise_split,
    BreakHisDataset,
    CLASS_MAP,
    FILENAME_RE,
)

__all__ = [
    "Sample",
    "index_dataset",
    "patientwise_split",
    "BreakHisDataset",
    "CLASS_MAP",
    "FILENAME_RE",
]