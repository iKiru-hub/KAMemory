"""Local CA3-to-CA1 plasticity rules used in controlled comparisons.

All functions are pure: they return an updated weight matrix without mutating
their inputs.  ``target_gated`` is the validated KAMemory update.  The other
rules are deliberately small baselines that receive the same presynaptic CA3
activity and the same content-bearing CA1 target.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


PlasticityRule = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]


def _validate_inputs(
    weights: torch.Tensor,
    presynaptic: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if weights.ndim != 2:
        raise ValueError("weights must be a two-dimensional matrix")
    presynaptic = torch.as_tensor(presynaptic, dtype=weights.dtype, device=weights.device)
    target = torch.as_tensor(target, dtype=weights.dtype, device=weights.device)
    if presynaptic.ndim == 1:
        presynaptic = presynaptic.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    if presynaptic.shape != (weights.shape[1], 1):
        raise ValueError(
            "presynaptic activity must have shape "
            f"({weights.shape[1]},) or ({weights.shape[1]}, 1)"
        )
    if target.shape != (weights.shape[0], 1):
        raise ValueError(
            f"target must have shape ({weights.shape[0]},) or ({weights.shape[0]}, 1)"
        )
    learning_rate = float(learning_rate)
    if not 0 <= learning_rate <= 1:
        raise ValueError("learning_rate must lie in [0, 1]")
    return presynaptic, target, learning_rate


def target_gated_update(
    weights: torch.Tensor,
    presynaptic: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
) -> torch.Tensor:
    """Validated IS-gated overwrite: active target rows approach the CA3 code."""

    presynaptic, target, learning_rate = _validate_inputs(
        weights, presynaptic, target, learning_rate
    )
    return (1 - learning_rate * target) * weights + learning_rate * (
        target @ presynaptic.T
    )


def hebbian_update(
    weights: torch.Tensor,
    presynaptic: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
) -> torch.Tensor:
    """Bounded potentiation-only Hebbian outer-product storage."""

    presynaptic, target, learning_rate = _validate_inputs(
        weights, presynaptic, target, learning_rate
    )
    return torch.clamp(
        weights + learning_rate * (target @ presynaptic.T), min=0.0, max=1.0
    )


def delta_update(
    weights: torch.Tensor,
    presynaptic: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
) -> torch.Tensor:
    """Bounded local least-mean-squares update with a CA1 prediction error.

    Dividing by presynaptic energy makes ``learning_rate`` the fraction of the
    single-pattern linear prediction error corrected by one storage event.
    """

    presynaptic, target, learning_rate = _validate_inputs(
        weights, presynaptic, target, learning_rate
    )
    energy = torch.sum(presynaptic.square()).clamp_min(
        torch.finfo(weights.dtype).eps
    )
    prediction_error = target - weights @ presynaptic
    return torch.clamp(
        weights
        + learning_rate * (prediction_error @ presynaptic.T) / energy,
        min=0.0,
        max=1.0,
    )


PLASTICITY_RULES: dict[str, PlasticityRule] = {
    "target_gated": target_gated_update,
    "hebbian": hebbian_update,
    "delta": delta_update,
}


def apply_plasticity(
    rule: str,
    weights: torch.Tensor,
    presynaptic: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
) -> torch.Tensor:
    """Apply a named rule, raising early for misspelled experiment settings."""

    try:
        update = PLASTICITY_RULES[rule]
    except KeyError as exc:
        choices = ", ".join(sorted(PLASTICITY_RULES))
        raise ValueError(f"unknown plasticity rule {rule!r}; choose from {choices}") from exc
    return update(weights, presynaptic, target, learning_rate)
