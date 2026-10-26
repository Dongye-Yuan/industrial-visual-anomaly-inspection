from pathlib import Path
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


DATA_DIR = Path("data/mvtec_ad/bottle")
OUTPUT_DIR = Path("outputs/hog_ocsvm")

IMAGE_SIZE = (128, 128)
TEST_CATEGORIES = [
    "good",
    "broken_large",
    "broken_small",
    "contamination",
]


def list_images(folder: Path):
    return sorted(folder.glob("*.png"))


def extract_hog_feature(image_path: Path):
    image = Image.open(image_path).convert("L")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(image)

    return hog(
        image_array,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def extract_features(image_paths):
    return np.vstack(
        [extract_hog_feature(image_path) for image_path in image_paths]
    )


def load_test_data():
    paths = []
    categories = []

    for category in TEST_CATEGORIES:
        category_paths = list_images(DATA_DIR / "test" / category)
        paths.extend(category_paths)
        categories.extend([category] * len(category_paths))

    labels = np.array(
        [0 if category == "good" else 1 for category in categories]
    )

    return paths, np.array(categories), labels


def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Good", "Anomaly"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("HOG + One-Class SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    return matrix


def save_roc_curve(y_true, anomaly_scores):
    RocCurveDisplay.from_predictions(
        y_true,
        anomaly_scores,
        name="HOG + One-Class SVM",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Image-Level ROC Curve")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
    plt.close()


def save_precision_recall_curve(y_true, anomaly_scores):
    PrecisionRecallDisplay.from_predictions(
        y_true,
        anomaly_scores,
        name="HOG + One-Class SVM",
    )
    plt.title("Image-Level Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=200)
    plt.close()


def save_score_distribution(categories, anomaly_scores):
    colors = {
        "good": "#4C78A8",
        "broken_large": "#E45756",
        "broken_small": "#F58518",
        "contamination": "#B279A2",
    }

    plt.figure(figsize=(10, 6))

    for category in TEST_CATEGORIES:
        category_scores = anomaly_scores[categories == category]
        plt.hist(
            category_scores,
            bins=12,
            alpha=0.65,
            label=category.replace("_", " ").title(),
            color=colors[category],
        )

    plt.axvline(
        0.0,
        color="black",
        linestyle="--",
        label="Decision threshold",
    )
    plt.xlabel("Anomaly score (higher means more anomalous)")
    plt.ylabel("Number of images")
    plt.title("HOG + One-Class SVM Anomaly Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distribution.png", dpi=200)
    plt.close()


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_paths = list_images(DATA_DIR / "train" / "good")
    test_paths, test_categories, y_test = load_test_data()

    print(f"Training images: {len(train_paths)}")
    print(f"Test images: {len(test_paths)}")
    print("Extracting HOG features...")

    train_start = time.perf_counter()
    x_train = extract_features(train_paths)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ocsvm",
                OneClassSVM(
                    kernel="rbf",
                    gamma="scale",
                    nu=0.05,
                ),
            ),
        ]
    )
    model.fit(x_train)
    training_seconds = time.perf_counter() - train_start

    print("Evaluating test images...")

    inference_start = time.perf_counter()
    x_test = extract_features(test_paths)

    # One-Class SVM returns positive values for inliers.
    # Negating the score makes larger values indicate stronger anomalies.
    anomaly_scores = -model.decision_function(x_test)
    inference_seconds = time.perf_counter() - inference_start

    threshold = 0.0
    y_pred = (anomaly_scores > threshold).astype(int)

    image_auroc = roc_auc_score(y_test, anomaly_scores)
    average_precision = average_precision_score(y_test, anomaly_scores)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    matrix = save_confusion_matrix(y_test, y_pred)
    save_roc_curve(y_test, anomaly_scores)
    save_precision_recall_curve(y_test, anomaly_scores)
    save_score_distribution(test_categories, anomaly_scores)

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )
    specificity = true_negative / (true_negative + false_positive)

    per_category_accuracy = {}

    for category in TEST_CATEGORIES:
        category_mask = test_categories == category
        expected_label = 0 if category == "good" else 1
        category_accuracy = np.mean(
            y_pred[category_mask] == expected_label
        )
        per_category_accuracy[category] = float(category_accuracy)

    latency_ms = inference_seconds / len(test_paths) * 1000

    metrics = {
        "model": "HOG + One-Class SVM",
        "training_images": len(train_paths),
        "test_images": len(test_paths),
        "image_size": list(IMAGE_SIZE),
        "hog_feature_dimension": int(x_train.shape[1]),
        "nu": 0.05,
        "threshold": threshold,
        "image_auroc": float(image_auroc),
        "average_precision": float(average_precision),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "training_seconds": float(training_seconds),
        "inference_ms_per_image": float(latency_ms),
        "confusion_matrix": matrix.tolist(),
        "per_category_accuracy": per_category_accuracy,
    }

    with (OUTPUT_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    predictions = pd.DataFrame(
        {
            "image_path": [str(path) for path in test_paths],
            "category": test_categories,
            "true_label": y_test,
            "anomaly_score": anomaly_scores,
            "predicted_label": y_pred,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    print("\nEvaluation results:")
    print(f"  Image AUROC: {image_auroc:.4f}")
    print(f"  Average Precision: {average_precision:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 score: {f1:.4f}")
    print(f"  Inference latency: {latency_ms:.2f} ms/image")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()