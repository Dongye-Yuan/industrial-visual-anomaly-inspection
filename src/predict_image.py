from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog


DEFAULT_MODEL_PATH = Path(
    "outputs/hog_ocsvm_calibrated/model.joblib"
)


def extract_hog_feature(
    image_path: Path,
    image_size,
    hog_parameters,
):
    with Image.open(image_path) as image:
        image = image.convert("L")
        image = image.resize(
            image_size,
            Image.Resampling.BILINEAR,
        )
        image_array = np.asarray(image)

    return hog(
        image_array,
        **hog_parameters,
        feature_vector=True,
    )


def predict_image(image_path: Path, model_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    artifact = joblib.load(model_path)

    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    image_size = artifact["image_size"]
    hog_parameters = artifact["hog_parameters"]

    feature = extract_hog_feature(
        image_path,
        image_size,
        hog_parameters,
    ).reshape(1, -1)

    # Larger values represent stronger anomalies.
    anomaly_score = float(
        -pipeline.decision_function(feature)[0]
    )

    predicted_label = (
        "Anomaly"
        if anomaly_score > threshold
        else "Good"
    )

    print("\nPrediction result")
    print(f"  Image: {image_path}")
    print(f"  Anomaly score: {anomaly_score:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Predicted label: {predicted_label}")


def main():
    parser = ArgumentParser(
        description=(
            "Predict whether a bottle image is "
            "Good or Anomaly."
        )
    )

    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the bottle image",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved model.joblib file",
    )

    args = parser.parse_args()

    predict_image(
        image_path=args.image_path,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()