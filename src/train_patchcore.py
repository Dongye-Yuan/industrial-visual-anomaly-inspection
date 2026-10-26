"""Train and evaluate a compact PatchCore-style bottle anomaly detector."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.patchcore import (
    IMAGE_SIZE,
    NORMALISATION_MEAN,
    NORMALISATION_STD,
    PATCH_GRID,
    choose_device,
    create_encoder,
    extract_paths,
    score_embeddings,
    select_coreset,
    upsample_anomaly_maps,
)


DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "mvtec_ad" / "bottle"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "patchcore"
TEST_CATEGORIES = ["good", "broken_large", "broken_small", "contamination"]


def list_images(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"))


def load_test_data(data_dir: Path):
    paths: list[Path] = []
    categories: list[str] = []
    masks: list[np.ndarray] = []

    for category in TEST_CATEGORIES:
        category_paths = list_images(data_dir / "test" / category)
        paths.extend(category_paths)
        categories.extend([category] * len(category_paths))

        for path in category_paths:
            if category == "good":
                mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            else:
                mask_path = (
                    data_dir
                    / "ground_truth"
                    / category
                    / f"{path.stem}_mask.png"
                )
                with Image.open(mask_path) as mask_image:
                    resized = mask_image.convert("L").resize(
                        IMAGE_SIZE,
                        Image.Resampling.NEAREST,
                    )
                    mask = (np.asarray(resized) > 0).astype(np.uint8)
            masks.append(mask)

    labels = np.array(
        [0 if category == "good" else 1 for category in categories],
        dtype=np.uint8,
    )
    return paths, np.asarray(categories), labels, np.stack(masks)


def compute_aupro(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    *,
    maximum_fpr: float = 0.30,
    thresholds: int = 200,
) -> float:
    """Compute area under the per-region-overlap curve up to the target FPR."""
    normal_pixels = masks == 0
    score_thresholds = np.linspace(
        float(anomaly_maps.max()),
        float(anomaly_maps.min()),
        thresholds,
    )
    points: list[tuple[float, float]] = []

    labelled_regions = []
    for mask in masks:
        labels, region_count = ndimage.label(mask)
        labelled_regions.append((labels, region_count))

    for threshold in score_thresholds:
        prediction = anomaly_maps >= threshold
        false_positive_rate = float(
            prediction[normal_pixels].mean()
        )
        if false_positive_rate > maximum_fpr:
            continue

        overlaps = []
        for prediction_map, (labels, region_count) in zip(
            prediction,
            labelled_regions,
        ):
            for region_id in range(1, region_count + 1):
                region = labels == region_id
                overlaps.append(float(prediction_map[region].mean()))

        if overlaps:
            points.append((false_positive_rate, float(np.mean(overlaps))))

    if len(points) < 2:
        return 0.0

    points.sort(key=lambda point: point[0])
    unique: dict[float, float] = {}
    for false_positive_rate, overlap in points:
        unique[false_positive_rate] = max(
            overlap,
            unique.get(false_positive_rate, 0.0),
        )

    x = np.asarray(sorted(unique), dtype=np.float64)
    y = np.asarray([unique[value] for value in x], dtype=np.float64)
    if x[0] > 0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
    if x[-1] < maximum_fpr:
        x = np.append(x, maximum_fpr)
        y = np.append(y, y[-1])

    return float(np.trapezoid(y, x) / maximum_fpr)


def save_evaluation_plots(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Good", "Anomaly"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Deep Patch Memory Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(
        y_true,
        anomaly_scores,
        name="Deep patch memory",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Image-Level ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    PrecisionRecallDisplay.from_predictions(
        y_true,
        anomaly_scores,
        name="Deep patch memory",
    )
    plt.title("Image-Level Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.hist(anomaly_scores[y_true == 0], bins=12, alpha=0.70, label="Good")
    plt.hist(anomaly_scores[y_true == 1], bins=12, alpha=0.70, label="Anomaly")
    plt.axvline(threshold, color="black", linestyle="--", label="Threshold")
    plt.xlabel("Anomaly score")
    plt.ylabel("Number of images")
    plt.title("Deep Patch Memory Anomaly Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=200)
    plt.close()
    return matrix


def save_heatmap_examples(
    output_dir: Path,
    test_paths: list[Path],
    categories: np.ndarray,
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
) -> None:
    selected = [
        int(np.flatnonzero(categories == category)[0])
        for category in TEST_CATEGORIES[1:]
    ]
    figure, axes = plt.subplots(3, 4, figsize=(13, 10))

    for row, index in enumerate(selected):
        with Image.open(test_paths[index]) as image:
            rgb = np.asarray(
                image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            )
        anomaly_map = anomaly_maps[index]
        lower, upper = np.percentile(anomaly_map, [5, 99])
        normalised = np.clip(
            (anomaly_map - lower) / max(upper - lower, 1e-8),
            0,
            1,
        )

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(categories[index].replace("_", " ").title())
        axes[row, 1].imshow(masks[index], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground-truth mask")
        axes[row, 2].imshow(normalised, cmap="turbo", vmin=0, vmax=1)
        axes[row, 2].set_title("Anomaly heatmap")
        axes[row, 3].imshow(rgb)
        axes[row, 3].imshow(normalised, cmap="turbo", alpha=0.45, vmin=0, vmax=1)
        axes[row, 3].set_title("Heatmap overlay")

        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle("Deep Patch Memory — Defect Localisation", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "heatmap_examples.png", dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact PatchCore-style anomaly detector."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--memory-size", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_train_paths = list_images(data_dir / "train" / "good")
    model_train_paths, calibration_paths = train_test_split(
        all_train_paths,
        test_size=0.20,
        random_state=42,
    )
    test_paths, categories, y_test, masks = load_test_data(data_dir)
    device = choose_device()

    print(f"Device: {device}")
    print(f"Model-training images: {len(model_train_paths)}")
    print(f"Calibration images: {len(calibration_paths)}")
    print(f"Test images: {len(test_paths)}")
    print("Extracting pretrained ResNet-18 patch features...")

    training_start = time.perf_counter()
    encoder = create_encoder(pretrained=True).to(device)
    train_embeddings = extract_paths(
        model_train_paths,
        encoder,
        device,
        batch_size=args.batch_size,
    )
    memory_bank = select_coreset(
        train_embeddings,
        args.memory_size,
    )
    training_seconds = time.perf_counter() - training_start

    calibration_embeddings = extract_paths(
        calibration_paths,
        encoder,
        device,
        batch_size=args.batch_size,
    )
    calibration_patch_scores = score_embeddings(
        calibration_embeddings,
        memory_bank,
    )
    calibration_image_scores = calibration_patch_scores.max(axis=1)
    threshold = float(np.quantile(calibration_image_scores, 0.95))
    pixel_threshold = float(np.quantile(calibration_patch_scores, 0.995))

    print(f"Calibrated image threshold: {threshold:.4f}")
    print("Evaluating test images and localisation maps...")
    inference_start = time.perf_counter()
    test_embeddings = extract_paths(
        test_paths,
        encoder,
        device,
        batch_size=args.batch_size,
    )
    test_patch_scores = score_embeddings(test_embeddings, memory_bank)
    anomaly_scores = test_patch_scores.max(axis=1)
    anomaly_maps = upsample_anomaly_maps(test_patch_scores)
    inference_seconds = time.perf_counter() - inference_start

    y_pred = (anomaly_scores > threshold).astype(np.uint8)
    image_auroc = roc_auc_score(y_test, anomaly_scores)
    average_precision = average_precision_score(y_test, anomaly_scores)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pixel_auroc = roc_auc_score(masks.ravel(), anomaly_maps.ravel())
    pixel_average_precision = average_precision_score(
        masks.ravel(),
        anomaly_maps.ravel(),
    )
    aupro = compute_aupro(masks, anomaly_maps)

    matrix = save_evaluation_plots(
        output_dir,
        y_test,
        y_pred,
        anomaly_scores,
        threshold,
    )
    save_heatmap_examples(
        output_dir,
        test_paths,
        categories,
        masks,
        anomaly_maps,
    )

    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / (true_negative + false_positive)
    latency_ms = inference_seconds / len(test_paths) * 1_000
    per_category_accuracy = {
        category: float(
            np.mean(
                y_pred[categories == category]
                == (0 if category == "good" else 1)
            )
        )
        for category in TEST_CATEGORIES
    }

    metrics = {
        "model": "PatchCore-style ResNet-18 patch memory",
        "implementation_note": (
            "Compact educational implementation using a frozen ResNet-18 "
            "layer2 encoder and deterministic k-centre coreset."
        ),
        "model_training_images": len(model_train_paths),
        "calibration_images": len(calibration_paths),
        "test_images": len(test_paths),
        "image_size": list(IMAGE_SIZE),
        "patch_grid": list(PATCH_GRID),
        "embedding_dimension": int(memory_bank.shape[1]),
        "memory_bank_size": int(memory_bank.shape[0]),
        "threshold": threshold,
        "pixel_threshold": pixel_threshold,
        "threshold_strategy": (
            "95th percentile of held-out normal image scores"
        ),
        "image_auroc": float(image_auroc),
        "average_precision": float(average_precision),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "pixel_auroc": float(pixel_auroc),
        "pixel_average_precision": float(pixel_average_precision),
        "aupro_0_30": float(aupro),
        "training_seconds": float(training_seconds),
        "inference_ms_per_image": float(latency_ms),
        "confusion_matrix": matrix.tolist(),
        "per_category_accuracy": per_category_accuracy,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    predictions = pd.DataFrame(
        {
            "image_path": [str(path) for path in test_paths],
            "category": categories,
            "true_label": y_test,
            "anomaly_score": anomaly_scores,
            "predicted_label": y_pred,
        }
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    artifact = {
        "format_version": 1,
        "model_name": "PatchCore-style ResNet-18 patch memory",
        "encoder_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in encoder.state_dict().items()
        },
        "memory_bank": torch.from_numpy(memory_bank),
        "threshold": threshold,
        "pixel_threshold": pixel_threshold,
        "image_size": IMAGE_SIZE,
        "patch_grid": PATCH_GRID,
        "normalisation_mean": NORMALISATION_MEAN,
        "normalisation_std": NORMALISATION_STD,
    }
    torch.save(artifact, output_dir / "model.pt")

    print("\nEvaluation results:")
    print(f"  Image AUROC: {image_auroc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 score: {f1:.4f}")
    print(f"  Pixel AUROC: {pixel_auroc:.4f}")
    print(f"  Pixel AP: {pixel_average_precision:.4f}")
    print(f"  AU-PRO (FPR <= 0.30): {aupro:.4f}")
    print(f"  Inference latency: {latency_ms:.2f} ms/image")
    print(f"\nResults saved to: {output_dir}")
    print(f"Model saved to: {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()

