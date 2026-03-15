"""
Model Calibration: Platt Scaling + Isotonic Regression

Transforms raw model outputs (compressed to 0-0.15 range) into
well-calibrated probabilities (0-1 range where 0.5 is the natural threshold).

Usage:
    python src/calibrate.py

Reads:  models/phase2_best.pt, processed/val_v2.pkl
Writes: models/calibrator.pkl
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, brier_score_loss,
    log_loss, classification_report
)
from tqdm import tqdm

from phase2.model_v2 import FakeJobDetectorV2


def get_model_predictions(model, dataloader, device):
    """Run model on a dataloader and return raw probabilities."""
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Predicting"):
            b_ids, b_mask, b_meta, b_graph = batch
            probs = model.predict_proba(
                b_ids.to(device), b_mask.to(device),
                b_meta.to(device), b_graph.to(device)
            )
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def make_dataloader(data, tokenizer, batch_size=32):
    """Create a DataLoader from a processed data dict."""
    print("  Tokenizing texts...")
    encodings = tokenizer(
        data['texts'], max_length=512, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(np.array(data['metadata']), dtype=torch.float32),
        torch.tensor(np.array(data['graph_features']), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    print("=" * 70)
    print("  MODEL CALIBRATION")
    print("  Platt Scaling + Isotonic Regression")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ---------------------------------------------------------------
    # 1. Load model
    # ---------------------------------------------------------------
    print("\n[1/5] Loading model...")
    with open('processed/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    checkpoint = torch.load('models/phase2_best.pt', map_location=device, weights_only=False)
    model = FakeJobDetectorV2(num_meta_features=len(feature_names), num_graph_features=14)

    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model'))
    mapped = {}
    for k, v in state_dict.items():
        nk = k.replace('bert.', 'distilbert.').replace('meta_enc.', 'metadata_encoder.').replace('graph_enc.', 'graph_encoder.encoder.')
        mapped[nk] = v
    model.load_state_dict(mapped, strict=False)
    model.to(device)
    model.eval()
    print("  Model loaded!")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # ---------------------------------------------------------------
    # 2. Get raw predictions on validation set
    # ---------------------------------------------------------------
    print("\n[2/5] Getting validation predictions...")
    with open('processed/val_v2.pkl', 'rb') as f:
        val_data = pickle.load(f)
    val_labels = np.array(val_data['labels'])

    val_loader = make_dataloader(val_data, tokenizer)
    val_raw_probs = get_model_predictions(model, val_loader, device)

    print(f"  Val samples:      {len(val_labels)}")
    print(f"  Val frauds:       {int(val_labels.sum())} ({100*val_labels.mean():.1f}%)")
    print(f"  Raw prob range:   [{val_raw_probs.min():.4f}, {val_raw_probs.max():.4f}]")
    print(f"  Raw prob mean:    {val_raw_probs.mean():.4f}")

    # ---------------------------------------------------------------
    # 3. Fit calibrators
    # ---------------------------------------------------------------
    print("\n[3/5] Fitting calibrators...")

    # Platt scaling (logistic regression on raw probs)
    platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    platt.fit(val_raw_probs.reshape(-1, 1), val_labels)
    platt_probs = platt.predict_proba(val_raw_probs.reshape(-1, 1))[:, 1]

    # Isotonic regression (non-parametric, more flexible)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(val_raw_probs, val_labels)
    iso_probs = iso.predict(val_raw_probs)

    print(f"  Platt calibrated range:    [{platt_probs.min():.4f}, {platt_probs.max():.4f}]")
    print(f"  Isotonic calibrated range: [{iso_probs.min():.4f}, {iso_probs.max():.4f}]")

    # ---------------------------------------------------------------
    # 4. Compare calibration quality
    # ---------------------------------------------------------------
    print("\n[4/5] Comparing calibration methods...")

    methods = {
        'Raw (uncalibrated)': val_raw_probs,
        'Platt scaling':      platt_probs,
        'Isotonic regression': iso_probs,
    }

    best_name = None
    best_f1 = 0
    best_calibrator = None

    for name, probs in methods.items():
        auc = roc_auc_score(val_labels, probs)
        brier = brier_score_loss(val_labels, probs)

        # Find best F1 threshold
        best_t, best_f1_val = 0.5, 0
        for t in np.arange(0.01, 0.95, 0.01):
            f1 = f1_score(val_labels, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1_val:
                best_f1_val = f1
                best_t = t

        preds = (probs >= best_t).astype(int)
        acc = accuracy_score(val_labels, preds)

        print(f"\n  --- {name} ---")
        print(f"    AUC:              {auc:.4f}")
        print(f"    Brier score:      {brier:.4f} (lower is better)")
        print(f"    Best threshold:   {best_t:.2f}")
        print(f"    F1 at threshold:  {best_f1_val:.4f}")
        print(f"    Accuracy:         {acc:.4f}")

        if best_f1_val > best_f1:
            best_f1 = best_f1_val
            best_name = name

    # Pick best calibrator. Isotonic is usually better for well-separated data.
    # But Platt is more stable. We'll save both and use isotonic as primary.
    # We also check: if isotonic brings threshold closer to 0.5, prefer it.
    platt_best_t = 0.5
    for t in np.arange(0.01, 0.95, 0.01):
        if f1_score(val_labels, (platt_probs >= t).astype(int), zero_division=0) > f1_score(val_labels, (platt_probs >= platt_best_t).astype(int), zero_division=0):
            platt_best_t = t

    iso_best_t = 0.5
    for t in np.arange(0.01, 0.95, 0.01):
        if f1_score(val_labels, (iso_probs >= t).astype(int), zero_division=0) > f1_score(val_labels, (iso_probs >= iso_best_t).astype(int), zero_division=0):
            iso_best_t = t

    print(f"\n  Platt best threshold:    {platt_best_t:.2f}")
    print(f"  Isotonic best threshold: {iso_best_t:.2f}")

    # ---------------------------------------------------------------
    # 5. Save calibrators
    # ---------------------------------------------------------------
    print("\n[5/5] Saving calibrators...")

    calibration_data = {
        'platt': platt,
        'isotonic': iso,
        'platt_threshold': platt_best_t,
        'isotonic_threshold': iso_best_t,
        'primary': 'isotonic',  # Use isotonic as primary
        'val_raw_range': (float(val_raw_probs.min()), float(val_raw_probs.max())),
        'val_samples': len(val_labels),
        'val_frauds': int(val_labels.sum()),
    }

    with open('models/calibrator.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)
    print("  Saved: models/calibrator.pkl")

    # Quick demo
    print("\n  --- Calibration Demo ---")
    demo_raws = [0.005, 0.01, 0.03, 0.05, 0.10, 0.15, 0.50, 0.90]
    for raw in demo_raws:
        p = platt.predict_proba([[raw]])[0][1]
        i = iso.predict([raw])[0]
        print(f"    Raw {raw:.3f}  -->  Platt: {p:.3f}  |  Isotonic: {i:.3f}")

    print("\n" + "=" * 70)
    print("  CALIBRATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
