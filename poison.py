"""
Label-flipping poisoning attack.

This is the attack used for baseline 2 (poisoned, no defense) in the
experiment design: a fixed fraction of the TRAINING set has its label
flipped (benign <-> malignant). This is a standard, simple, well-cited
poisoning threat model and gives a clean dose-response curve across
poison rates, which is what the paper's headline figure needs.

Only the training split should ever be poisoned. Val/test stay clean so
that "attack success" and "clean accuracy" are both measurable.
"""

import random
from typing import List, Dict

from dataset import Sample


def make_label_flip_overrides(
    samples: List[Sample],
    poison_rate: float,
    seed: int = 42,
) -> Dict[int, int]:
    """
    Returns a dict {index_in_samples: flipped_label} for a random poison_rate
    fraction of the given sample list (indices are positional, matching the
    order `samples` will be passed into BreakHisDataset).

    poison_rate=0.0 -> empty dict -> clean baseline.
    poison_rate=0.05 -> ~5% of samples get their label flipped.
    """
    if not (0.0 <= poison_rate < 1.0):
        raise ValueError("poison_rate must be in [0, 1)")

    n = len(samples)
    n_poison = int(round(n * poison_rate))
    if n_poison == 0:
        return {}

    rng = random.Random(seed)
    poisoned_idx = rng.sample(range(n), n_poison)

    overrides = {}
    for idx in poisoned_idx:
        true_label = samples[idx].label
        flipped = 1 - true_label  # binary task: benign(0) <-> malignant(1)
        overrides[idx] = flipped

    print(f"[poison] Flipped {n_poison}/{n} training labels "
          f"(requested rate={poison_rate:.2%}).")
    return overrides