from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_DIR / "outputs"

MODEL_FILES = {
    "OCSVM Baseline": OUTPUTS_DIR / "hog_ocsvm" / "metrics.json",
    "OCSVM Calibrated": OUTPUTS_DIR
    / "hog_ocsvm_calibrated"
    / "metrics.json",
    "Isolation Forest": OUTPUTS_DIR
    / "hog_isolation_forest"
    / "metrics.json",
}

# Colours sampled from the supplied blue-grey Canva PowerPoint template.
NAVY = "#1F3A5F"
STRONG_BLUE = "#2E5BBA"
LIGHT_BLUE = "#6EA8FE"
GREY_BLUE = "#8293A6"
SLIDE_GREY = "#E2E2E2"
PALE_BLUE = "#D7E4F7"
GRID_GREY = "#C4CAD1"
TEXT_GREY = "#4B5563"
WHITE = "#FFFFFF"

MODEL_COLOURS = [GREY_BLUE, STRONG_BLUE, LIGHT_BLUE]


def configure_plot_style() -> None:
    """Apply a clean style that visually matches the supplied presentation."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelcolor": TEXT_GREY,
            "xtick.color": TEXT_GREY,
            "ytick.color": TEXT_GREY,
            "legend.fontsize": 10.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def load_metrics(outputs_dir: Path) -> pd.DataFrame:
    """Read the evaluation results produced by the three training scripts."""
    model_files = {
        name: outputs_dir / path.relative_to(OUTPUTS_DIR)
        for name, path in MODEL_FILES.items()
    }
    rows = []

    for model_name, metrics_path in model_files.items():
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        rows.append(
            {
                "Model": model_name,
                "AUROC": metrics["image_auroc"],
                "Average Precision": metrics["average_precision"],
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "F1 Score": metrics["f1_score"],
                "Latency (ms/image)": metrics["inference_ms_per_image"],
            }
        )

    return pd.DataFrame(rows)


def format_results(results: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a display copy with consistent precision and shorter headers."""
    display = results[columns].copy()
    for column in columns:
        if column == "Model":
            continue
        if column == "Latency (ms/image)":
            display[column] = display[column].map(lambda value: f"{value:.2f}")
        else:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    return display.rename(
        columns={
            "Average Precision": "Avg. Precision",
            "F1 Score": "F1 Score",
            "Latency (ms/image)": "Latency\n(ms/image)",
        }
    )


def style_table(
    table,
    source_results: pd.DataFrame,
    columns: list[str],
    selected_row: int,
    font_size: float,
) -> None:
    """Style headers, selected row, borders, and best metric cells."""
    for column_index in range(len(columns)):
        cell = table[(0, column_index)]
        cell.set_facecolor(NAVY)
        cell.set_edgecolor(WHITE)
        cell.set_linewidth(1.2)
        cell.set_text_props(color=WHITE, weight="bold")

    best_rows: dict[str, int] = {}
    for column in columns:
        if column == "Model":
            continue
        if column == "Latency (ms/image)":
            best_rows[column] = int(source_results[column].idxmin())
        else:
            best_rows[column] = int(source_results[column].idxmax())

    for row_index in range(len(source_results)):
        for column_index, column in enumerate(columns):
            cell = table[(row_index + 1, column_index)]
            cell.set_edgecolor(GRID_GREY)
            cell.set_linewidth(0.8)
            cell.set_facecolor(PALE_BLUE if row_index == selected_row else WHITE)
            cell.set_text_props(color="#111827", fontsize=font_size)

            if column == "Model":
                cell.set_text_props(weight="bold")
            elif best_rows[column] == row_index:
                cell.set_text_props(weight="bold", color=NAVY)


def save_report_table(results: pd.DataFrame, output_dir: Path) -> Path:
    """Create a detailed report-ready comparison table."""
    columns = [
        "Model",
        "AUROC",
        "Average Precision",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1 Score",
        "Latency (ms/image)",
    ]
    display = format_results(results, columns)
    selected_row = int(results["F1 Score"].idxmax())

    fig, ax = plt.subplots(figsize=(15.2, 3.65), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.16, 0.085, 0.13, 0.09, 0.09, 0.085, 0.10, 0.09, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.95)
    style_table(table, results, columns, selected_row, 10)

    ax.set_title(
        "Industrial Visual Anomaly Detection — Model Comparison",
        color=NAVY,
        fontsize=19,
        fontweight="bold",
        pad=13,
    )
    ax.text(
        0.5,
        0.025,
        "Bold = best result for each metric  |  Lower is better for latency  |  "
        "Selected model is highlighted",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color=TEXT_GREY,
    )

    path = output_dir / "model_comparison_table_report.png"
    fig.savefig(path, facecolor=WHITE, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return path


def save_ppt_table(results: pd.DataFrame, output_dir: Path) -> Path:
    """Create a compact table sized for the presentation content area."""
    columns = [
        "Model",
        "AUROC",
        "Accuracy",
        "Recall",
        "Specificity",
        "F1 Score",
    ]
    display = format_results(results, columns)
    selected_row = int(results["F1 Score"].idxmax())

    fig, ax = plt.subplots(figsize=(13.33, 3.35), facecolor=SLIDE_GREY)
    ax.set_facecolor(SLIDE_GREY)
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.24, 0.145, 0.145, 0.145, 0.17, 0.145],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 2.30)
    style_table(table, results, columns, selected_row, 15)

    path = output_dir / "model_comparison_table_ppt.png"
    fig.savefig(path, facecolor=SLIDE_GREY, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def draw_grouped_bars(
    results: pd.DataFrame,
    output_path: Path,
    *,
    background: str,
    include_title: bool,
    presentation: bool,
) -> None:
    """Draw the same accurate comparison in report and presentation styles."""
    metrics = ["Accuracy", "Recall", "Specificity", "F1 Score"]
    labels = ["Accuracy", "Recall", "Specificity", "F1 score"]
    x = np.arange(len(metrics))
    width = 0.23
    offsets = (np.arange(len(results)) - (len(results) - 1) / 2) * width

    size = (13.33, 6.25) if presentation else (12.2, 6.7)
    fig, ax = plt.subplots(figsize=size, facecolor=background)
    ax.set_facecolor(background)

    for model_index, (_, row) in enumerate(results.iterrows()):
        values = [float(row[metric]) for metric in metrics]
        bars = ax.bar(
            x + offsets[model_index],
            values,
            width=width,
            label=row["Model"],
            color=MODEL_COLOURS[model_index],
            edgecolor=background,
            linewidth=0.9,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values],
            padding=4,
            fontsize=11.5 if presentation else 9.5,
            color="#111827",
            fontweight="bold" if model_index == 1 else "normal",
        )

    if include_title:
        ax.set_title(
            "Model Performance Comparison",
            color=NAVY,
            fontsize=20,
            fontweight="bold",
            pad=16,
        )

    ax.set_ylabel("Score", color=TEXT_GREY, fontsize=13)
    ax.set_ylim(0, 1.10)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=12.5 if presentation else 11)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", color=GRID_GREY, linestyle="-", linewidth=0.8, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_GREY)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="y", length=0)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01 if presentation else 1.0),
        ncol=3,
        frameon=False,
        columnspacing=2.1,
        handlelength=1.8,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_GREY)

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.14,
        top=0.87 if include_title else 0.91,
    )
    fig.savefig(output_path, facecolor=background, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def save_bar_charts(results: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    report_path = output_dir / "model_comparison_bar_chart_report.png"
    ppt_path = output_dir / "model_comparison_bar_chart_ppt.png"
    draw_grouped_bars(
        results,
        report_path,
        background=WHITE,
        include_title=True,
        presentation=False,
    )
    draw_grouped_bars(
        results,
        ppt_path,
        background=SLIDE_GREY,
        include_title=False,
        presentation=True,
    )
    return report_path, ppt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final model comparison visuals.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Directory containing model metric folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for comparison files (default: outputs/model_comparison).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else outputs_dir / "model_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    results = load_metrics(outputs_dir)
    csv_path = output_dir / "model_comparison.csv"
    results.to_csv(csv_path, index=False)

    created = [
        save_report_table(results, output_dir),
        save_ppt_table(results, output_dir),
        *save_bar_charts(results, output_dir),
    ]

    best_model = results.loc[results["F1 Score"].idxmax()]
    print("\nFinal model comparison visuals completed.")
    print(f"Selected model by F1 score: {best_model['Model']}")
    print(f"F1 score: {best_model['F1 Score']:.4f}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"Specificity: {best_model['Specificity']:.4f}")
    print("\nCreated:")
    print(f"  {csv_path}")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
