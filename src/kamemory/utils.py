"""Small numerical helpers shared by models and data generation."""

from __future__ import annotations

import random
import hashlib

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int, deterministic: bool = False) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch and return a NumPy generator.

    The returned generator should be passed to data-generation functions. This
    avoids experiments depending on unrelated uses of NumPy's global RNG.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


def sparsemoid(z: torch.Tensor, k: int, beta: float) -> torch.Tensor:
    """Smoothly select approximately the top ``k`` units along the last axis."""

    if z.ndim == 1:
        z = z.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    width = z.shape[-1]
    if not 0 <= k <= width:
        raise ValueError(f"k must lie in [0, {width}], got {k}")

    if k == 0:
        threshold: torch.Tensor | float = 0.0
    else:
        ordered = torch.sort(z, descending=True, dim=-1).values
        threshold = ordered[..., k - 1 : min(k + 1, width)].mean(
            dim=-1, keepdim=True
        )
    result = torch.sigmoid(float(beta) * (z - threshold))
    return result.squeeze(0) if squeeze else result


def binary_cross_entropy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(x, y)


def cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between flattened tensors."""

    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    return torch.dot(x_flat, y_flat) / (
        torch.linalg.vector_norm(x_flat) * torch.linalg.vector_norm(y_flat) + eps
    )


def legacy_equal_tuning(n: int, n_connections: int, rng=None) -> np.ndarray:
    """Reproduce the original ``make_equal_tuning`` initialization.

    The function intentionally returns the historical index-valued matrix. It
    is isolated and explicitly named because changing this initialization is a
    scientific manipulation, not a code refactor.
    """

    if n <= 0 or n_connections <= 0:
        raise ValueError("n and n_connections must be positive")
    rng = np.random if rng is None else rng
    available = list(range(n))
    remaining = np.ones(n) * n_connections
    rows: list[list[int]] = []
    all_indices = np.arange(n)

    for _ in range(n):
        if len(available) >= n_connections:
            selected = rng.choice(available, replace=False, size=n_connections)
        else:
            selected = np.asarray(available, dtype=int)
            missing = n_connections - len(selected)
            if missing:
                fill = rng.choice(
                    all_indices,
                    replace=missing > len(all_indices),
                    size=missing,
                )
                selected = np.concatenate((selected, fill))
        rows.append(selected.tolist())

        for index in selected:
            if remaining[index] <= 0:
                continue
            remaining[index] -= 1
            if remaining[index] == 0 and index in available:
                available.remove(index)

    return np.asarray(rows, dtype=np.float32)


def swap_binary_activity(x: torch.Tensor, num_swaps: int) -> torch.Tensor:
    """Move active entries to inactive positions without changing sparsity."""

    if num_swaps <= 0:
        return x.clone()
    flat = x.reshape(-1).clone()
    active = torch.nonzero(flat != 0, as_tuple=False).flatten()
    inactive = torch.nonzero(flat == 0, as_tuple=False).flatten()
    count = min(int(num_swaps), len(active), len(inactive))
    if count == 0:
        return x.clone()
    turn_off = active[torch.randperm(len(active), device=x.device)[:count]]
    turn_on = inactive[torch.randperm(len(inactive), device=x.device)[:count]]
    flat[turn_off] = 0
    flat[turn_on] = 1
    return flat.reshape_as(x)


def as_float_tensor(data, *, device=None) -> torch.Tensor:
    """Convert arrays/tensors without needlessly copying existing tensors."""

    if isinstance(data, torch.Tensor):
        return data.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def array_digest(data) -> str:
    """Stable SHA-256 digest including an array's dtype, shape, and values."""

    array = np.ascontiguousarray(np.asarray(data))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()
