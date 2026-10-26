import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from src.predict_image import extract_hog_feature, predict_image


HOG_PARAMETERS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


class FixedDecisionPipeline:
    """Small serializable test double for a fitted scikit-learn pipeline."""

    def __init__(self, decision_value):
        self.decision_value = decision_value

    def decision_function(self, features):
        return np.full(features.shape[0], self.decision_value)


class PredictImageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.image_path = self.root / "sample.png"

        gradient = np.tile(
            np.arange(128, dtype=np.uint8),
            (128, 1),
        )
        Image.fromarray(gradient).save(self.image_path)

    def save_artifact(self, decision_value, threshold=0.5):
        model_path = self.root / "model.joblib"
        joblib.dump(
            {
                "pipeline": FixedDecisionPipeline(decision_value),
                "threshold": threshold,
                "image_size": (128, 128),
                "hog_parameters": HOG_PARAMETERS,
            },
            model_path,
        )
        return model_path

    def test_hog_feature_contract(self):
        feature = extract_hog_feature(
            self.image_path,
            (128, 128),
            HOG_PARAMETERS,
        )

        self.assertEqual(feature.shape, (1764,))
        self.assertTrue(np.isfinite(feature).all())

    def test_prediction_uses_calibrated_threshold(self):
        cases = [
            (-0.4, "Good"),
            (-0.8, "Anomaly"),
        ]

        for decision_value, expected_label in cases:
            with self.subTest(expected_label=expected_label):
                model_path = self.save_artifact(decision_value)
                output = io.StringIO()

                with redirect_stdout(output):
                    predict_image(self.image_path, model_path)

                self.assertIn(
                    f"Predicted label: {expected_label}",
                    output.getvalue(),
                )

    def test_missing_image_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "Image not found"):
            predict_image(
                self.root / "missing.png",
                self.root / "model.joblib",
            )

    def test_missing_model_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "Model not found"):
            predict_image(
                self.image_path,
                self.root / "missing.joblib",
            )


class SavedArtifactTests(unittest.TestCase):
    def test_versioned_model_has_required_inference_metadata(self):
        artifact = joblib.load(
            "outputs/hog_ocsvm_calibrated/model.joblib"
        )

        self.assertEqual(
            set(artifact),
            {
                "pipeline",
                "threshold",
                "image_size",
                "hog_parameters",
            },
        )
        self.assertEqual(tuple(artifact["image_size"]), (128, 128))
        self.assertAlmostEqual(
            float(artifact["threshold"]),
            0.5603564758744974,
        )


if __name__ == "__main__":
    unittest.main()
