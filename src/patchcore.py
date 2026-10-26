"""Shared utilities for the PatchCore-style deep patch memory model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as torch_functional
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2


IMAGE_SIZE = (224, 224)
PATCH_GRID = (14, 14)
NORMALISATION_MEAN = (0.485, 0.456, 0.406)
NORMALISATION_STD = (0.229, 0.224, 0.225)


def choose_device() -> torch.device:
    """Use Apple or NVIDIA acceleration when available, otherwise use CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_encoder(*, pretrained: bool) -> nn.Module:
    """Create a frozen ResNet-18 encoder ending at layer2."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)
    encoder = nn.Sequential(*list(backbone.children())[:6])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def build_transform(image_size: tuple[int, int] = IMAGE_SIZE):
    """Return the deterministic preprocessing used during training and inference."""
    return v2.Compose(
        [
            v2.Resize(image_size, antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(NORMALISATION_MEAN, NORMALISATION_STD),
        ]
    )


def image_to_tensor(
    image: Image.Image,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> torch.Tensor:
    """Convert one PIL image to a normalised RGB tensor."""
    return build_transform(image_size)(image.convert("RGB"))


def extract_patch_embeddings(
    encoder: nn.Module,
    images: torch.Tensor,
    patch_grid: tuple[int, int] = PATCH_GRID,
) -> torch.Tensor:
    """Return L2-normalised patch embeddings with shape B x P x D."""
    with torch.inference_mode():
        feature_map = encoder(images)
        feature_map = torch_functional.adaptive_avg_pool2d(
            feature_map,
            patch_grid,
        )
        embeddings = feature_map.flatten(2).transpose(1, 2)
        return torch_functional.normalize(embeddings, dim=-1)


def extract_paths(
    paths: Iterable[Path],
    encoder: nn.Module,
    device: torch.device,
    *,
    batch_size: int = 16,
    image_size: tuple[int, int] = IMAGE_SIZE,
    patch_grid: tuple[int, int] = PATCH_GRID,
) -> np.ndarray:
    """Extract patch embeddings for image paths into a NumPy array."""
    path_list = list(paths)
    batches = []

    for start in range(0, len(path_list), batch_size):
        tensors = []
        for path in path_list[start : start + batch_size]:
            with Image.open(path) as image:
                tensors.append(image_to_tensor(image, image_size))

        images = torch.stack(tensors).to(device)
        embeddings = extract_patch_embeddings(
            encoder,
            images,
            patch_grid,
        )
        batches.append(embeddings.cpu().numpy())

    if not batches:
        patch_count = patch_grid[0] * patch_grid[1]
        return np.empty((0, patch_count, 128), dtype=np.float32)

    return np.concatenate(batches, axis=0).astype(np.float32)


def select_coreset(
    patch_embeddings: np.ndarray,
    memory_size: int,
    *,
    seed: int = 42,
    candidate_limit: int = 10_000,
    projection_dimension: int = 32,
) -> np.ndarray:
    """Select a deterministic k-centre coreset of normal patch embeddings."""
    features = patch_embeddings.reshape(-1, patch_embeddings.shape[-1])
    if len(features) <= memory_size:
        return features.copy()

    rng = np.random.default_rng(seed)
    candidate_count = min(candidate_limit, len(features))
    candidate_indices = rng.choice(
        len(features),
        candidate_count,
        replace=False,
    )
    candidates = features[candidate_indices]

    projection = rng.normal(
        size=(candidates.shape[1], projection_dimension),
    ).astype(np.float32)
    projection /= np.sqrt(projection_dimension)
    projected = candidates @ projection

    selected = np.empty(memory_size, dtype=np.int64)
    selected[0] = int(rng.integers(candidate_count))
    minimum_distances = np.full(candidate_count, np.inf, dtype=np.float32)

    for index in range(memory_size):
        if index > 0:
            selected[index] = int(np.argmax(minimum_distances))
        centre = projected[selected[index]]
        squared_distances = np.sum((projected - centre) ** 2, axis=1)
        minimum_distances = np.minimum(
            minimum_distances,
            squared_distances,
        )

    return candidates[selected].astype(np.float32)


def score_embeddings(
    patch_embeddings: np.ndarray,
    memory_bank: np.ndarray,
    *,
    chunk_size: int = 2_048,
) -> np.ndarray:
    """Return nearest-memory distance for every patch."""
    original_shape = patch_embeddings.shape[:-1]
    flattened = torch.from_numpy(
        patch_embeddings.reshape(-1, patch_embeddings.shape[-1])
    )
    memory = torch.from_numpy(memory_bank)
    scores = []

    for start in range(0, len(flattened), chunk_size):
        distances = torch.cdist(
            flattened[start : start + chunk_size],
            memory,
        )
        scores.append(distances.min(dim=1).values)

    return torch.cat(scores).numpy().reshape(original_shape)


def upsample_anomaly_maps(
    patch_scores: np.ndarray,
    *,
    patch_grid: tuple[int, int] = PATCH_GRID,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Upsample flat patch scores to smooth image-sized anomaly maps."""
    score_tensor = torch.from_numpy(patch_scores).reshape(
        -1,
        1,
        patch_grid[0],
        patch_grid[1],
    )
    maps = torch_functional.interpolate(
        score_tensor,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    return maps[:, 0].numpy()


def load_artifact(path: Path, device: torch.device):
    """Load a safe tensor-only artifact and reconstruct its feature encoder."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    encoder = create_encoder(pretrained=False)
    encoder.load_state_dict(artifact["encoder_state_dict"])
    encoder.to(device).eval()
    return artifact, encoder

