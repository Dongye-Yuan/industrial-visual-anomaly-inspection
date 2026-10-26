from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATA_DIR = Path("data/mvtec_ad/bottle")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"

CATEGORIES = [
    ("train", "good"),
    ("test", "good"),
    ("test", "broken_large"),
    ("test", "broken_small"),
    ("test", "contamination"),
]


def image_files(folder: Path):
    return sorted(folder.glob("*.png"))


def load_rgb(path: Path):
    return np.asarray(Image.open(path).convert("RGB"))


def count_images():
    rows = []

    for split, category in CATEGORIES:
        count = len(image_files(DATA_DIR / split / category))
        rows.append(
            {"split": split, "category": category, "count": count}
        )

    return rows


def save_summary(rows):
    summary_path = OUTPUT_DIR / "dataset_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["split", "category", "count"]
        )
        writer.writeheader()
        writer.writerows(rows)

    labels = [f"{row['split']}/{row['category']}" for row in rows]
    counts = [row["count"] for row in rows]
    colors = ["#2E8B57", "#4C78A8", "#E45756", "#F58518", "#B279A2"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, counts, color=colors)

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            count + 2,
            str(count),
            ha="center",
        )

    plt.title("MVTec AD Bottle Dataset Distribution")
    plt.ylabel("Number of images")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "dataset_distribution.png", dpi=200)
    plt.close()


def save_sample_overview():
    labels = [
        "Train: Good",
        "Test: Good",
        "Broken Large",
        "Broken Small",
        "Contamination",
    ]

    paths = [
        image_files(DATA_DIR / split / category)[0]
        for split, category in CATEGORIES
    ]

    figure, axes = plt.subplots(1, 5, figsize=(15, 4))

    for axis, label, path in zip(axes, labels, paths):
        axis.imshow(load_rgb(path))
        axis.set_title(label)
        axis.axis("off")

    figure.suptitle("Representative Bottle Samples")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "sample_overview.png", dpi=200)
    plt.close(figure)


def save_mask_examples():
    defect_types = ["broken_large", "broken_small", "contamination"]
    figure, axes = plt.subplots(3, 3, figsize=(10, 10))

    for row, defect_type in enumerate(defect_types):
        image_path = image_files(DATA_DIR / "test" / defect_type)[0]
        mask_path = (
            DATA_DIR
            / "ground_truth"
            / defect_type
            / f"{image_path.stem}_mask.png"
        )

        image = load_rgb(image_path)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0

        overlay = image.copy()
        overlay[mask] = (
            0.55 * image[mask] + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{defect_type}: Image")

        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("Ground Truth Mask")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Defect Overlay")

        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle("Pixel-Level Defect Annotations")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "defect_mask_examples.png", dpi=200)
    plt.close(figure)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = count_images()
    save_summary(rows)
    save_sample_overview()
    save_mask_examples()

    print("Dataset summary:")
    for row in rows:
        print(f"  {row['split']}/{row['category']}: {row['count']}")

    print("\nCreated:")
    print(f"  {OUTPUT_DIR / 'dataset_summary.csv'}")
    print(f"  {FIGURE_DIR / 'dataset_distribution.png'}")
    print(f"  {FIGURE_DIR / 'sample_overview.png'}")
    print(f"  {FIGURE_DIR / 'defect_mask_examples.png'}")


if __name__ == "__main__":
    main()
