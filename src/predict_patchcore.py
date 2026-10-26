"""Run PatchCore-style inference and save a localisation heatmap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.patchcore import (
    choose_device,
    extract_patch_embeddings,
    image_to_tensor,
    load_artifact,
    score_embeddings,
    upsample_anomaly_maps,
)


DEFAULT_MODEL_PATH = PROJECT_DIR / "outputs" / "patchcore" / "model.pt"
DEFAULT_OUTPUT_PATH = (
    PROJECT_DIR / "outputs" / "patchcore" / "prediction_heatmap.png"
)


def predict(image_path: Path, model_path: Path, output_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = choose_device()
    artifact, encoder = load_artifact(model_path, device)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        tensor = image_to_tensor(
            image,
            tuple(artifact["image_size"]),
        ).unsqueeze(0).to(device)
        embeddings = extract_patch_embeddings(
            encoder,
            tensor,
            tuple(artifact["patch_grid"]),
        ).cpu().numpy()

    memory_bank = artifact["memory_bank"].cpu().numpy()
    patch_scores = score_embeddings(embeddings, memory_bank)
    anomaly_score = float(patch_scores.max())
    threshold = float(artifact["threshold"])
    label = "Anomaly" if anomaly_score > threshold else "Good"
    anomaly_map = upsample_anomaly_maps(
        patch_scores,
        patch_grid=tuple(artifact["patch_grid"]),
        image_size=tuple(artifact["image_size"]),
    )[0]

    resized = np.asarray(
        image.resize(tuple(artifact["image_size"]), Image.Resampling.BILINEAR)
    )
    lower, upper = np.percentile(anomaly_map, [5, 99])
    normalised = np.clip(
        (anomaly_map - lower) / max(upper - lower, 1e-8),
        0,
        1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(resized)
    axes[0].set_title("Input")
    axes[1].imshow(normalised, cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title("Anomaly heatmap")
    axes[2].imshow(resized)
    axes[2].imshow(normalised, cmap="turbo", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title(f"Overlay — {label}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("\nPatchCore-style prediction")
    print(f"  Image: {image_path}")
    print(f"  Anomaly score: {anomaly_score:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Predicted label: {label}")
    print(f"  Heatmap saved to: {output_path}")
    return anomaly_score, threshold, label, anomaly_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict and localise anomalies with deep patch memory."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    predict(args.image_path, args.model, args.output)


if __name__ == "__main__":
    main()

