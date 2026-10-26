import unittest
from pathlib import Path

import numpy as np
import torch

from src.patchcore import score_embeddings, select_coreset, upsample_anomaly_maps


class PatchCoreUtilityTests(unittest.TestCase):
    def test_coreset_is_deterministic_and_has_requested_size(self):
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(5, 12, 8)).astype(np.float32)
        first = select_coreset(embeddings, 16, seed=42)
        second = select_coreset(embeddings, 16, seed=42)

        self.assertEqual(first.shape, (16, 8))
        np.testing.assert_allclose(first, second)

    def test_patch_scores_and_upsampled_map_have_expected_shapes(self):
        memory = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        embeddings = np.array(
            [[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]],
            dtype=np.float32,
        )
        scores = score_embeddings(embeddings, memory)
        maps = upsample_anomaly_maps(
            scores,
            patch_grid=(2, 2),
            image_size=(8, 8),
        )

        self.assertEqual(scores.shape, (1, 4))
        self.assertEqual(maps.shape, (1, 8, 8))
        self.assertAlmostEqual(float(scores[0, 0]), 0.0)
        self.assertGreater(float(scores[0, 3]), 1.0)


class SavedPatchArtifactTests(unittest.TestCase):
    def test_versioned_patch_artifact_has_required_metadata(self):
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "outputs" / "patchcore" / "model.pt"
        artifact = torch.load(model_path, map_location="cpu", weights_only=True)

        required_keys = {
            "format_version",
            "model_name",
            "encoder_state_dict",
            "memory_bank",
            "threshold",
            "pixel_threshold",
            "image_size",
            "patch_grid",
            "normalisation_mean",
            "normalisation_std",
        }
        self.assertTrue(required_keys.issubset(artifact))
        self.assertEqual(artifact["format_version"], 1)
        self.assertEqual(tuple(artifact["image_size"]), (224, 224))
        self.assertEqual(tuple(artifact["patch_grid"]), (14, 14))
        self.assertEqual(tuple(artifact["memory_bank"].shape), (1024, 128))


if __name__ == "__main__":
    unittest.main()

