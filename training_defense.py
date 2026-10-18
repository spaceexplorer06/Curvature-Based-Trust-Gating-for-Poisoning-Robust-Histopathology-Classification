"""
Training script for poisoning-defense comparison on BreaKHis.

Supports four --defense modes:
    none        Plain ResNet-50, no reweighting (baseline)
    ricci       Ollivier-Ricci curvature trust gate (core contribution --
                see ricci_gate.py for the method and its rationale)
    small_loss  Small-loss trust baseline (non-geometric ablation control)
    bloch       Bloch-sphere/quantum-framed gate (kept only as a rejected-
                alternative ablation -- see bloch_gate.py's module docstring
                and /areas/breakhis-histopathology.md for why this was
                dropped as the headline method)

Usage:
    python training_defense.py --data_root /path/to/BreaKHis_v1 --poison_rate 0.1 --defense ricci --run_name test --epochs 20
"""

import os
import sys
import csv
import json
import random
import argparse
import time

# Must be set before torch initializes a CUDA context, so it has to happen
# before `import torch`. Without this, cuBLAS falls back to a
# non-deterministic algorithm selection for some matmul/cdist shapes even
# with torch.use_deterministic_algorithms(True) requested below -- this was
# part of why identical --seed values were producing different final
# ricci-gate results run to run (the rest was scatter_add_ in ricci_gate.py,
# fixed there; this covers the matmul/cdist side).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import index_dataset, patientwise_split, BreakHisDataset
from pcam_dataset import load_pcam_splits, PCamH5Dataset
from poison import make_label_flip_overrides
from model import ResNet50Classifier
from bloch_gate import BlochSphereClassifier
from ricci_gate import RicciSphereClassifier
from loss_trust_gate import SmallLossClassifier


def set_seed(seed: int):
    """Seed every RNG that affects run-to-run reproducibility, not just torch's."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True: use deterministic kernels wherever available (this is
    # what actually fixes GPU run-to-run drift) without hard-crashing on the
    # rare op with no deterministic CUDA implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int):
    """Per-worker seed for DataLoader workers, so RandomHorizontalFlip/
    RandomVerticalFlip augmentation is reproducible across runs instead of
    drawing from an unseeded per-process RNG (this was the cause of
    identical configs producing opposite class-collapse outcomes)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_auc(labels, malignant_probs):
    """ROC-AUC for the malignant class via the Mann-Whitney U statistic.
    Avoids adding an sklearn dependency for one metric. Returns 0.5
    (chance) if only one class is present in `labels`."""
    labels = np.asarray(labels)
    scores = np.asarray(malignant_probs)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg_rank = (sums / counts)[inv]
    sum_ranks_pos = avg_rank[labels == 1].sum()
    auc = (sum_ranks_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def compute_confusion_metrics(class_correct, class_total, class_pred_total):
    """Precision/recall/F1 per class + macro-F1 + balanced accuracy from
    accumulated confusion counts. class_correct[c] = TP_c,
    class_total[c] = actual count of class c (TP_c + FN_c),
    class_pred_total[c] = predicted count of class c (TP_c + FP_c)."""
    metrics = {}
    f1s = []
    for c in (0, 1):
        tp = class_correct[c]
        recall = tp / max(1, class_total[c])
        precision = tp / max(1, class_pred_total[c])
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        metrics[f"precision_{c}"] = precision
        metrics[f"recall_{c}"] = recall
        metrics[f"f1_{c}"] = f1
        f1s.append(f1)
    metrics["macro_f1"] = sum(f1s) / len(f1s)
    metrics["balanced_accuracy"] = (
        class_correct[0] / max(1, class_total[0]) + class_correct[1] / max(1, class_total[1])
    ) / 2
    return metrics


def metrics_from_probs(labels, probs, threshold=0.5):
    """Recompute the same confusion-based metrics as compute_confusion_metrics,
    but from raw (label, P(malignant)) pairs at an arbitrary decision
    threshold instead of from already-thresholded prediction counts. Used
    for threshold calibration: pick the val-optimal threshold, then report
    test metrics at that threshold instead of the default 0.5 cutoff."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    class_pred_total = {0: 0, 1: 0}
    for c in (0, 1):
        mask = labels == c
        class_total[c] = int(mask.sum())
        class_correct[c] = int(((preds == c) & mask).sum())
        class_pred_total[c] = int((preds == c).sum())

    metrics = {
        "benign_acc": class_correct[0] / max(1, class_total[0]),
        "malignant_acc": class_correct[1] / max(1, class_total[1]),
    }
    metrics.update(compute_confusion_metrics(class_correct, class_total, class_pred_total))
    metrics["accuracy"] = (preds == labels).mean().item()
    metrics["threshold"] = threshold
    return metrics


def find_best_threshold(labels, probs, metric="balanced_accuracy", n_steps=199):
    """Sweeps decision thresholds on (labels, probs) -- meant to be called on
    the VALIDATION set only -- and returns the threshold maximizing the given
    metric, plus the metrics dict at that threshold. Never call this on the
    test set: that would be tuning on the test set, which invalidates the
    reported numbers."""
    best_threshold, best_score, best_metrics = 0.5, -1.0, None
    for t in np.linspace(0.01, 0.99, n_steps):
        m = metrics_from_probs(labels, probs, threshold=float(t))
        if m[metric] > best_score:
            best_score, best_threshold, best_metrics = m[metric], float(t), m
    return best_threshold, best_metrics



def get_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def compute_class_weights(samples):
    """Compute inverse-frequency weights for CrossEntropyLoss."""
    n_benign = sum(s.label == 0 for s in samples)
    n_malign = sum(s.label == 1 for s in samples)
    total = n_benign + n_malign
    w0 = total / (2 * max(n_benign, 1))
    w1 = total / (2 * max(n_malign, 1))
    weights = torch.tensor([w0, w1], dtype=torch.float32)
    print(f"[train] Class weights: benign={w0:.3f}, malignant={w1:.3f} "
          f"(counts: {n_benign} vs {n_malign})")
    return weights


def run_epoch(model, loader, criterion, optimizer, device, train, use_gate=False, scaler=None):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    gate_stats_accum = {}
    per_class_correct = {0: 0, 1: 0}
    per_class_total = {0: 0, 1: 0}
    per_class_pred_total = {0: 0, 1: 0}
    use_amp = scaler is not None and device.type == "cuda"

    torch.set_grad_enabled(train)
    pbar = tqdm(loader, desc="Train" if train else "Val  ", leave=False)
    for images, labels, _idx in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            if use_gate:
                logits, gate_out = model(images, labels, return_gate=True)
                loss, gate_stats = model.compute_loss(logits, labels, gate_out)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        if use_gate:
            for k, v in gate_stats.items():
                gate_stats_accum[k] = gate_stats_accum.get(k, 0.0) + v

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        n += images.size(0)

        for c in (0, 1):
            mask = labels == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (preds[mask] == c).sum().item()
            per_class_pred_total[c] += (preds == c).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/n:.4f}"})

    avg_gate_stats = {k: v / len(loader) for k, v in gate_stats_accum.items()} if gate_stats_accum else {}
    per_class_acc = {
        "benign_acc": per_class_correct[0] / max(1, per_class_total[0]),
        "malignant_acc": per_class_correct[1] / max(1, per_class_total[1]),
    }
    per_class_acc.update(compute_confusion_metrics(per_class_correct, per_class_total, per_class_pred_total))
    return total_loss / n, correct / n, avg_gate_stats, per_class_acc


def evaluate_attack_success(model, test_loader, criterion, device, use_gate=False, use_amp=False):
    model.eval()
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    class_pred_total = {0: 0, 1: 0}
    gate_stats_accum = {}
    total_loss, correct, n = 0.0, 0, 0
    all_labels, all_malignant_probs = [], []
    use_amp = use_amp and device.type == "cuda"

    with torch.no_grad():
        for images, labels, _idx in tqdm(test_loader, desc="Test ", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                if use_gate:
                    logits, gate_out = model(images, labels, return_gate=True)
                    loss, _ = model.compute_loss(logits, labels, gate_out)
                    for k, v in gate_out.items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            gate_stats_accum[k] = gate_stats_accum.get(k, 0.0) + v.item()
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            n += labels.size(0)

            probs = F.softmax(logits, dim=1)[:, 1]  # P(malignant)
            all_labels.extend(labels.cpu().tolist())
            all_malignant_probs.extend(probs.cpu().tolist())

            for c in (0, 1):
                mask = labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == c).sum().item()
                class_pred_total[c] += (preds == c).sum().item()

    per_class_acc = {
        "benign_acc": class_correct[0] / max(1, class_total[0]),
        "malignant_acc": class_correct[1] / max(1, class_total[1]),
    }
    per_class_acc.update(compute_confusion_metrics(class_correct, class_total, class_pred_total))
    per_class_acc["auc"] = compute_auc(all_labels, all_malignant_probs)

    avg_gate_stats = {k: v / len(test_loader) for k, v in gate_stats_accum.items()} if gate_stats_accum else {}
    avg_loss = total_loss / max(n, 1)
    avg_acc = correct / max(n, 1)
    return avg_loss, avg_acc, per_class_acc, avg_gate_stats, (all_labels, all_malignant_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="breakhis", choices=["breakhis", "pcam"],
                        help="Which dataset to train/evaluate on")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Required for --dataset breakhis: path to extracted BreaKHis_v1/")
    parser.add_argument("--pcam_root", type=str, default=None,
                        help="Required for --dataset pcam: directory containing the six PCam .h5 files")
    parser.add_argument("--poison_rate", type=float, default=0.0)
    parser.add_argument("--defense", type=str, default="none",
                        choices=["none", "bloch", "ricci", "small_loss"])
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Skip ImageNet-pretrained weight download (random init) -- "
                             "for offline smoke-testing only, not real experiment runs")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed-precision training (torch.autocast + GradScaler). "
                             "CUDA only -- silently ignored on CPU. Roughly 1.5-2x speedup "
                             "on modern GPUs with no reduction in epochs/data, worth enabling "
                             "for any large-dataset run (e.g. PCam).")
    parser.add_argument("--gate_mode", type=str, default="soft", choices=["soft", "hard", "curvature_only"])
    parser.add_argument("--gate_loss_weight", type=float, default=0.1)
    parser.add_argument("--curvature_threshold", type=float, default=0.3)
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weighting")
    # Ricci-gate-specific knobs (ignored by other defenses)
    parser.add_argument("--ricci_k_neighbors", type=int, default=5)
    parser.add_argument("--ricci_proj_dim", type=int, default=32)
    parser.add_argument("--ricci_alpha", type=float, default=0.5,
                        help="Idleness mass at each node in its own Ollivier-Ricci neighbor distribution")
    parser.add_argument("--ricci_curvature_threshold", type=float, default=0.0)
    # small-loss-gate-specific knob (ignored by other defenses)
    parser.add_argument("--keep_fraction", type=float, default=0.8,
                        help="Assumed-clean fraction of each batch for the small-loss baseline")
    args = parser.parse_args()

    if args.dataset == "breakhis" and not args.data_root:
        parser.error("--dataset breakhis requires --data_root")
    if args.dataset == "pcam" and not args.pcam_root:
        parser.error("--dataset pcam requires --pcam_root")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}, dataset={args.dataset}, defense={args.defense}, poison_rate={args.poison_rate}")

    out_dir = os.path.join("runs", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Data
    print("[train] Loading dataset...")
    train_tf, eval_tf = get_transforms()

    if args.dataset == "breakhis":
        samples = index_dataset(args.data_root)
        train_s, val_s, test_s = patientwise_split(samples, seed=args.seed)
        label_overrides = make_label_flip_overrides(train_s, args.poison_rate, seed=args.seed)
        train_ds = BreakHisDataset(train_s, transform=train_tf, label_override=label_overrides)
        val_ds = BreakHisDataset(val_s, transform=eval_tf)
        test_ds = BreakHisDataset(test_s, transform=eval_tf)
    else:  # pcam
        # PCam ships pre-defined, already leakage-safe train/valid/test
        # splits -- unlike BreaKHis, --seed does not affect which patches
        # land in which split here (see pcam_dataset.py module docstring).
        train_s, val_s, test_s = load_pcam_splits(args.pcam_root)
        label_overrides = make_label_flip_overrides(train_s, args.poison_rate, seed=args.seed)
        train_ds = PCamH5Dataset(train_s, transform=train_tf, label_override=label_overrides)
        val_ds = PCamH5Dataset(val_s, transform=eval_tf)
        test_ds = PCamH5Dataset(test_s, transform=eval_tf)

    pin = torch.cuda.is_available()
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    # persistent_workers avoids respawning worker processes (and, for
    # PCam, re-opening h5py file handles) every single epoch -- a real
    # per-epoch overhead that compounds over many epochs and is pure
    # waste, unlike the actual forward/backward compute.
    persistent = args.num_workers > 0
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=pin,
                          persistent_workers=persistent)
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               worker_init_fn=seed_worker, generator=loader_gen,
                               **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # Class weights for imbalance
    if args.no_class_weights:
        class_weights = None
        print("[train] Class weighting disabled")
    else:
        class_weights = compute_class_weights(train_s).to(device)

    # Model
    if args.defense == "bloch":
        gate_config = {
            "gate_mode": args.gate_mode,
            "curvature_threshold": args.curvature_threshold,
        }
        model = BlochSphereClassifier(
            num_classes=2,
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone,
            gate_config=gate_config,
            gate_loss_weight=args.gate_loss_weight,
        ).to(device)
        model.set_class_weights(class_weights)
        use_gate = True
        criterion = None
        print(f"[train] Bloch-sphere defense enabled (mode={args.gate_mode})")
    elif args.defense == "ricci":
        gate_config = {
            "gate_mode": args.gate_mode,
            "curvature_threshold": args.ricci_curvature_threshold,
            "k_neighbors": args.ricci_k_neighbors,
            "proj_dim": args.ricci_proj_dim,
            "alpha": args.ricci_alpha,
        }
        model = RicciSphereClassifier(
            num_classes=2,
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone,
            gate_config=gate_config,
            gate_loss_weight=args.gate_loss_weight,
        ).to(device)
        model.set_class_weights(class_weights)
        use_gate = True
        criterion = None
        print(f"[train] Ollivier-Ricci curvature defense enabled (mode={args.gate_mode}, "
              f"k={args.ricci_k_neighbors})")
    elif args.defense == "small_loss":
        gate_config = {
            "gate_mode": args.gate_mode if args.gate_mode in ("soft", "hard") else "soft",
            "keep_fraction": args.keep_fraction,
        }
        model = SmallLossClassifier(
            num_classes=2,
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone,
            gate_config=gate_config,
        ).to(device)
        model.set_class_weights(class_weights)
        use_gate = True
        criterion = None
        print(f"[train] Small-loss trust baseline enabled (keep_fraction={args.keep_fraction})")
    else:
        model = ResNet50Classifier(num_classes=2, pretrained=not args.no_pretrained,
                                    freeze_backbone=args.freeze_backbone).to(device)
        use_gate = False
        criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        print("[train] No defense (standard ResNet-50)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # Train loop
    # NOTE: model selection uses balanced accuracy (mean of per-class recall),
    # not raw val_acc. With class imbalance, raw accuracy is trivially gamed
    # by a model that collapses to predicting the majority class every time
    # (this is what several earlier runs' benign_acc=0.0/malignant_acc=1.0-
    # style results were doing).
    best_val_acc = 0.0
    best_val_balanced_acc = 0.0
    metrics_path = os.path.join(out_dir, "metrics.csv")

    amp_active = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_active)
    if args.amp and not amp_active:
        print("[train] --amp requested but no CUDA device available; running in full precision.")
    elif amp_active:
        print("[train] Mixed-precision (AMP) training enabled.")

    header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_benign", "val_malignant",
               "val_balanced_acc", "val_macro_f1", "seconds"]
    # Each defense's compute_loss() reports a different set of gate_stats
    # keys, so the extra CSV columns depend on which defense is active
    # (previously hardcoded to bloch's key names only, which silently wrote
    # zeros/garbage for other defenses).
    gate_stat_fields = {
        "bloch": ["mean_trust", "mean_purity", "mean_curvature", "ce_loss", "bloch_loss"],
        "ricci": ["mean_trust", "mean_curvature", "mean_agreement", "ce_loss", "ricci_loss"],
        "small_loss": ["mean_trust", "mean_loss_percentile", "ce_loss"],
    }
    if use_gate:
        header.extend(gate_stat_fields.get(args.defense, []))

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*60}")

            train_loss, train_acc, train_gate, train_cls = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True, use_gate=use_gate, scaler=scaler
            )
            val_loss, val_acc, val_gate, val_cls = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False, use_gate=use_gate, scaler=scaler
            )
            dt = time.time() - t0

            row = [epoch, train_loss, train_acc, val_loss, val_acc, val_cls["benign_acc"], val_cls["malignant_acc"],
                   val_cls["balanced_accuracy"], val_cls["macro_f1"], dt]
            if use_gate:
                row.extend(train_gate.get(k, 0) for k in gate_stat_fields.get(args.defense, []))

            writer.writerow(row)
            f.flush()

            gate_str = ""
            if use_gate:
                gate_str = " | " + " ".join(
                    f"{k}={train_gate.get(k, 0):.3f}" for k in gate_stat_fields.get(args.defense, [])
                )

            print(f"\n[epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"train_benign={train_cls['benign_acc']:.4f} train_malignant={train_cls['malignant_acc']:.4f}")
            print(f"           val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"val_benign={val_cls['benign_acc']:.4f} val_malignant={val_cls['malignant_acc']:.4f} "
                  f"val_balanced_acc={val_cls['balanced_accuracy']:.4f} val_macro_f1={val_cls['macro_f1']:.4f} "
                  f"({dt:.1f}s){gate_str}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if val_cls["balanced_accuracy"] > best_val_balanced_acc:
                best_val_balanced_acc = val_cls["balanced_accuracy"]
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    # Final clean-test evaluation
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))

    # Threshold calibration: pick the decision threshold that maximizes
    # balanced accuracy on the VALIDATION set (never the test set), then
    # report test metrics at both the default 0.5 cutoff and this
    # calibrated threshold. This separates "is the trust/curvature signal
    # good" (AUC, threshold-independent) from "is the default 0.5 cutoff
    # well-calibrated for this defense" (accuracy/balanced-accuracy at a
    # fixed threshold) -- a defense can rank samples well but still lose on
    # accuracy if its probability outputs sit off-center from 0.5.
    val_loss, val_acc_eval, val_cls_eval, _, (val_labels, val_probs) = evaluate_attack_success(
        model, val_loader, criterion, device, use_gate=use_gate, use_amp=args.amp
    )
    best_threshold, val_metrics_at_best = find_best_threshold(val_labels, val_probs)

    test_loss, test_acc, test_cls, test_gate, (test_labels, test_probs) = evaluate_attack_success(
        model, test_loader, criterion, device, use_gate=use_gate, use_amp=args.amp
    )
    calibrated_test_cls = metrics_from_probs(test_labels, test_probs, threshold=best_threshold)

    report = {
        "run_name": args.run_name,
        "poison_rate": args.poison_rate,
        "defense": args.defense,
        "seed": args.seed,
        "clean_test_accuracy": test_acc,
        "test_balanced_accuracy": test_cls["balanced_accuracy"],
        "test_macro_f1": test_cls["macro_f1"],
        "best_val_accuracy": best_val_acc,
        "best_val_balanced_accuracy": best_val_balanced_acc,
        "test_benign_acc": test_cls["benign_acc"],
        "test_malignant_acc": test_cls["malignant_acc"],
        **test_cls,
        # threshold-calibrated numbers (threshold chosen on val, applied to test)
        "calibrated_threshold": best_threshold,
        "calibrated_val_balanced_accuracy": val_metrics_at_best["balanced_accuracy"],
        "calibrated_test_accuracy": calibrated_test_cls["accuracy"],
        "calibrated_test_balanced_accuracy": calibrated_test_cls["balanced_accuracy"],
        "calibrated_test_macro_f1": calibrated_test_cls["macro_f1"],
        "calibrated_test_benign_acc": calibrated_test_cls["benign_acc"],
        "calibrated_test_malignant_acc": calibrated_test_cls["malignant_acc"],
    }
    if test_gate:
        report["gate_stats"] = test_gate

    with open(os.path.join(out_dir, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Per-sample test predictions, saved separately from the aggregate report.
    # Needed for a paired DeLong AUC significance test between two defenses
    # (patientwise_split is keyed on --seed, not on --defense, so two runs
    # with the same seed share the exact same test patients/samples --
    # that's what makes a *paired* test valid here, rather than treating
    # each run's test set as an independent sample).
    with open(os.path.join(out_dir, "test_probs.json"), "w") as f:
        json.dump({"labels": list(test_labels), "malignant_probs": list(test_probs)}, f)

    print(f"\n[train] TEST (threshold=0.5)         acc={test_acc:.4f} balanced_acc={test_cls['balanced_accuracy']:.4f} "
          f"macro_f1={test_cls['macro_f1']:.4f} auc={test_cls['auc']:.4f} "
          f"benign={test_cls['benign_acc']:.4f} malignant={test_cls['malignant_acc']:.4f}")
    print(f"[train] TEST (threshold={best_threshold:.3f}, val-calibrated)  "
          f"acc={calibrated_test_cls['accuracy']:.4f} balanced_acc={calibrated_test_cls['balanced_accuracy']:.4f} "
          f"macro_f1={calibrated_test_cls['macro_f1']:.4f} "
          f"benign={calibrated_test_cls['benign_acc']:.4f} malignant={calibrated_test_cls['malignant_acc']:.4f}")
    print("[train] Final report:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()