#!/usr/bin/env python3
"""
Quick diagnostic: verify dataset loading, model creation, and one forward pass.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("DIAGNOSTIC: Checking imports and basic functionality")
print("=" * 60)

print("\n[1/5] Checking imports...")
try:
    import torch
    print(f"  ok torch {torch.__version__}")
except Exception as e:
    print(f"  FAIL torch import: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"  ok torchvision {torchvision.__version__}")
except Exception as e:
    print(f"  FAIL torchvision import: {e}")
    sys.exit(1)

try:
    from dataset_loader import index_dataset, patientwise_split, BreakHisDataset
    print("  ok dataset_loader")
except Exception as e:
    print(f"  FAIL dataset_loader import: {e}")
    sys.exit(1)

try:
    from pcam_dataset import load_pcam_splits, PCamH5Dataset
    print("  ok pcam_dataset")
except Exception as e:
    print(f"  FAIL pcam_dataset import: {e}")
    sys.exit(1)

try:
    from model import ResNet50Classifier
    print("  ok model")
except Exception as e:
    print(f"  FAIL model import: {e}")
    sys.exit(1)

try:
    from bloch_gate import BlochSphereClassifier
    print("  ok bloch_gate")
except Exception as e:
    print(f"  FAIL bloch_gate import: {e}")
    sys.exit(1)

try:
    from ricci_gate import RicciSphereClassifier
    print("  ok ricci_gate")
except Exception as e:
    print(f"  FAIL ricci_gate import: {e}")
    sys.exit(1)

try:
    from loss_trust_gate import SmallLossClassifier
    print("  ok loss_trust_gate")
except Exception as e:
    print(f"  FAIL loss_trust_gate import: {e}")
    sys.exit(1)

print("\n[2/5] Checking dataset...")
dataset_kind = sys.argv[2] if len(sys.argv) > 2 else "breakhis"
data_root = sys.argv[1] if len(sys.argv) > 1 else "/home/moinak/Desktop/Histopathology/BreaKHis_v1"
print(f"  Dataset: {dataset_kind}")
print(f"  Data root: {data_root}")

try:
    if dataset_kind == "pcam":
        train_s, val_s, test_s = load_pcam_splits(data_root)
        print(f"  ok Loaded PCam splits: train={len(train_s)} val={len(val_s)} test={len(test_s)}")
    else:
        samples = index_dataset(data_root)
        print(f"  ok Found {len(samples)} samples")
        print(f"  ok Sample 0: {os.path.basename(samples[0].path)}")
        n_benign = sum(s.label == 0 for s in samples)
        n_malign = sum(s.label == 1 for s in samples)
        print(f"  ok Labels: {n_benign} benign, {n_malign} malignant")
except Exception as e:
    print(f"  FAIL Dataset loading: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/5] Checking patient-wise split...")
try:
    if dataset_kind == "pcam":
        print(f"  ok PCam uses pre-defined official splits (already loaded above); "
              f"skipping patient-wise split, which is BreaKHis-specific.")
    else:
        train_s, val_s, test_s = patientwise_split(samples, seed=42)
        print(f"  ok Train: {len(train_s)} images")
        print(f"  ok Val:   {len(val_s)} images")
        print(f"  ok Test:  {len(test_s)} images")
except Exception as e:
    print(f"  FAIL Split: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/5] Checking model creation...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = ResNet50Classifier(num_classes=2, pretrained=True).to(device)
    print("  ok ResNet-50 baseline created")
    model_bloch = BlochSphereClassifier(num_classes=2, pretrained=True,
                                         gate_config={"gate_mode": "soft"}).to(device)
    print("  ok Bloch-sphere model created")
    model_ricci = RicciSphereClassifier(num_classes=2, pretrained=True,
                                         gate_config={"gate_mode": "soft", "k_neighbors": 5}).to(device)
    print("  ok Ricci-curvature model created")
    model_smallloss = SmallLossClassifier(num_classes=2, pretrained=True,
                                           gate_config={"gate_mode": "soft"}).to(device)
    print("  ok Small-loss model created")
except Exception as e:
    print(f"  FAIL Model creation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[5/5] Checking forward pass...")
try:
    from torchvision import transforms
    from torch.utils.data import DataLoader

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    ds = PCamH5Dataset(train_s[:4], transform=tf) if dataset_kind == "pcam" else BreakHisDataset(train_s[:4], transform=tf)
    loader = DataLoader(ds, batch_size=2)

    images, labels, idx = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        out = model(images)
    print(f"  ok Baseline forward: {out.shape}")

    with torch.no_grad():
        out_bloch, gate_out = model_bloch(images, labels, return_gate=True)
    print(f"  ok Bloch forward: {out_bloch.shape}")
    ts = gate_out["trust_score"].mean().item()
    pu = gate_out["purity"].mean().item()
    print(f"  ok Bloch gate outputs: trust={ts:.4f}, purity={pu:.4f}")

    with torch.no_grad():
        out_ricci, gate_out_r = model_ricci(images, labels, return_gate=True)
    print(f"  ok Ricci forward: {out_ricci.shape}")
    ts_r = gate_out_r["trust_score"].mean().item()
    curv_r = gate_out_r["curvature"].mean().item()
    print(f"  ok Ricci gate outputs: trust={ts_r:.4f}, curvature={curv_r:.4f}")

    with torch.no_grad():
        out_sl, gate_out_sl = model_smallloss(images, labels, return_gate=True)
    print(f"  ok Small-loss forward: {out_sl.shape}")
    ts_sl = gate_out_sl["trust_score"].mean().item()
    print(f"  ok Small-loss gate outputs: trust={ts_sl:.4f}")

except Exception as e:
    print(f"  FAIL Forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)