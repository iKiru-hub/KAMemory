"""
Core neural models used by experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .plasticity import PLASTICITY_RULES, apply_plasticity
from .utils import legacy_equal_tuning, sparsemoid, swap_binary_activity


class Autoencoder(nn.Module):
    """ single-layer sparse autoencoder """

    def __init__(self, input_dim: int = 10, encoding_dim: int = 10,
                 activation: str | None = None, K: int = 10,
                 beta: float = 20.0, use_bias: bool = True) -> None:
        super().__init__()
        del activation  # retained for old checkpoint/session constructors
        self._input_dim = input_dim
        self._encoding_dim = encoding_dim
        self._K = K
        self._beta = beta
        self._use_bias = use_bias

        # Sequential preserves existing checkpoint keys (encoder.0.weight, ...).
        self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim, bias=use_bias))
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, input_dim, bias=use_bias))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return sparsemoid(self.encoder(x), self._K, self._beta)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(10 * (self.decoder(code) - 0.1))

    def forward(self, x: torch.Tensor, ca1: bool = False):
        code = self.encode(x)
        reconstruction = self.decode(code)
        return (reconstruction, code) if ca1 else reconstruction

    def get_weights(self, bias: bool = False):
        encoder_weight = self.encoder[0].weight.detach()
        decoder_weight = self.decoder[0].weight.detach()
        if bias and self._use_bias:
            return (
                encoder_weight,
                decoder_weight,
                self.encoder[0].bias.detach().reshape(-1, 1),
                self.decoder[0].bias.detach().reshape(-1, 1),
            )
        return encoder_weight, decoder_weight, None, None


class BTSPMemory(nn.Module):
    """CA3-to-CA1 associative memory with an explicit BTSP storage operation.

    ``forward``/``retrieve`` never change synaptic weights. Use ``store`` to
    apply the plasticity rule. Fixed encoder/decoder weights and the plastic
    matrix are buffers because they are model state, not gradient parameters.
    """

    def __init__(self, W_ei_ca1: torch.Tensor, W_ca1_eo: torch.Tensor, K_lat: int,
                 K_out: int, dim_ca3: int, beta: float, alpha: float = 0.01,
                 K_ca3: int = 10, num_swaps: int = 0, identity_IS: bool = False,
                 random_IS: bool = False, B_ei_ca1: torch.Tensor | None = None,
                 B_ca1_eo: torch.Tensor | None = None,
                 instructive_permutation: torch.Tensor | None = None,
                 plasticity_rule: str = "target_gated", record: bool = False) -> None:
        super().__init__()
        encoder = torch.as_tensor(W_ei_ca1, dtype=torch.float32).detach().clone()
        decoder = torch.as_tensor(W_ca1_eo, dtype=torch.float32).detach().clone()
        self._dim_ei = encoder.shape[1]
        self._dim_ca1 = decoder.shape[1]
        self._dim_eo = decoder.shape[0]
        self._dim_ca3 = int(dim_ca3)
        if encoder.shape[0] != self._dim_ca1:
            raise ValueError("encoder and decoder disagree on CA1 dimension")
        if identity_IS and self._dim_ei != self._dim_ca1:
            raise ValueError("identity_IS requires equal EC-input and CA1 dimensions")

        self._K_lat = int(K_lat)
        self._K_out = int(K_out)
        self._K_ca3 = int(K_ca3)
        self._beta = float(beta)
        self._beta_ca3 = 100.0 * float(beta)
        self._alpha = float(alpha)
        self._num_swaps = int(num_swaps)
        self.identity_IS = bool(identity_IS)
        self.random_IS = bool(random_IS)
        self.set_plasticity_rule(plasticity_rule)
        self.learning_enabled = True
        self.record_enabled = bool(record)

        projection = torch.as_tensor(
            legacy_equal_tuning(self._dim_ca3, self._dim_ei), dtype=torch.float32
        ) / self._dim_ca3
        if projection.shape != (self._dim_ca3, self._dim_ei):
            raise ValueError(
                "legacy CA3 projection requires dim_ca3 rows and dim_ei columns"
            )

        self.register_buffer("W_ei_ca3", projection)
        self.register_buffer("W_ei_ca1", encoder)
        self.register_buffer("W_ca3_ca1", torch.zeros(self._dim_ca1, self._dim_ca3))
        self.register_buffer("W_ca1_eo", decoder)
        self.register_buffer(
            "B_ei_ca1",
            torch.zeros(self._dim_ca1, 1)
            if B_ei_ca1 is None
            else torch.as_tensor(B_ei_ca1, dtype=torch.float32).detach().clone(),
        )
        self.register_buffer(
            "B_ca1_eo",
            torch.zeros(self._dim_eo, 1)
            if B_ca1_eo is None
            else torch.as_tensor(B_ca1_eo, dtype=torch.float32).detach().clone(),
        )
        permutation = (
            torch.empty(0, dtype=torch.long)
            if instructive_permutation is None
            else torch.as_tensor(instructive_permutation, dtype=torch.long)
        )
        if permutation.numel() and sorted(permutation.tolist()) != list(range(self._dim_ca1)):
            raise ValueError("instructive_permutation must permute every CA1 unit once")
        self.register_buffer("instructive_permutation", permutation)
        self.recordings: dict[str, list[torch.Tensor]] = {}
        self.clear_recordings()

    @classmethod
    def from_autoencoder(cls, autoencoder: Autoencoder, **kwargs) -> "BTSPMemory":
        weights = autoencoder.get_weights(bias=True)
        return cls(
            W_ei_ca1=weights[0],
            W_ca1_eo=weights[1],
            B_ei_ca1=weights[2],
            B_ca1_eo=weights[3],
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"BTSPMemory(dim_ei={self._dim_ei}, dim_ca3={self._dim_ca3}, "
            f"dim_ca1={self._dim_ca1}, dim_eo={self._dim_eo}, "
            f"alpha={self._alpha}, beta={self._beta}, "
            f"plasticity_rule={self.plasticity_rule!r})"
        )

    @staticmethod
    def _column(x: torch.Tensor | np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim == 2 and x.shape[1] == 1:
            return x
        raise ValueError("BTSPMemory processes one pattern shaped (dim,) or (dim, 1)")

    def ca3_activity(self, x_ei: torch.Tensor) -> torch.Tensor:
        x_ei = self._column(x_ei)
        activity = sparsemoid(
            (self.W_ei_ca3 @ x_ei).T, self._K_ca3, self._beta_ca3
        ).T
        return swap_binary_activity(activity, self._num_swaps)

    def instructive_signal(self, x_ei: torch.Tensor) -> torch.Tensor:
        x_ei = self._column(x_ei)
        if self.identity_IS:
            signal = x_ei
        else:
            signal = sparsemoid(
                (self.W_ei_ca1 @ x_ei + self.B_ei_ca1).T,
                self._K_lat,
                self._beta,
            ).T
        if self.instructive_permutation.numel():
            signal = signal[self.instructive_permutation]
        elif self.random_IS:
            signal = signal[torch.randperm(signal.shape[0], device=signal.device)]
        return signal

    def ca1_activity(self, x_ca3: torch.Tensor) -> torch.Tensor:
        activity = sparsemoid(
            (self.W_ca3_ca1 @ x_ca3).T, self._K_lat, self._beta_ca3
        ).T
        return swap_binary_activity(activity, self._num_swaps)

    def decode(self, x_ca1: torch.Tensor) -> torch.Tensor:
        return sparsemoid(
            (self.W_ca1_eo @ x_ca1 + self.B_ca1_eo).T,
            self._K_out,
            self._beta,
        ).T

    @torch.no_grad()
    def store(
        self,
        x_ei: torch.Tensor,
        instructive_signal: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Apply one BTSP update and return the instructive signal used.

        Passing an explicit signal supports controlled experiments while the
        default derives the content-aware signal from ``x_ei``.
        """

        x_ei = self._column(x_ei).to(self.W_ca3_ca1)
        x_ca3 = self.ca3_activity(x_ei)
        if instructive_signal is None:
            signal = self.instructive_signal(x_ei)
        else:
            signal = self._column(instructive_signal).to(self.W_ca3_ca1)
            if signal.shape != (self._dim_ca1, 1):
                raise ValueError(
                    "instructive_signal must have shape "
                    f"({self._dim_ca1},) or ({self._dim_ca1}, 1)"
                )
        if self.learning_enabled:
            updated = apply_plasticity(
                self.plasticity_rule,
                self.W_ca3_ca1,
                x_ca3,
                signal,
                self._alpha,
            )
            self.W_ca3_ca1.copy_(updated)
        self._record(x_ei=x_ei, ca3=x_ca3, instructive_signal=signal)
        return signal

    def retrieve(self, x_ei: torch.Tensor, *, return_ca1: bool = False):
        x_ei = self._column(x_ei).to(self.W_ca3_ca1)
        x_ca3 = self.ca3_activity(x_ei)
        x_ca1 = self.ca1_activity(x_ca3)
        output = self.decode(x_ca1)
        self._record(x_ei=x_ei, ca3=x_ca3, ca1=x_ca1, output=output)
        return (output, x_ca1) if return_ca1 else output

    def forward(
        self,
        x_ei: torch.Tensor,
        ca1: bool = False,
        *,
        learn: bool = False,
        test: bool = False,
    ):
        if learn and not test:
            self.store(x_ei)
        return self.retrieve(x_ei, return_ca1=ca1)

    def reset_memory(self) -> None:
        self.W_ca3_ca1.zero_()
        self.clear_recordings()

    def reset(self) -> None:
        self.reset_memory()
        self.learning_enabled = True

    def pause_lr(self) -> None:
        self.learning_enabled = False

    def resume_lr(self) -> None:
        self.learning_enabled = True

    def set_alpha(self, alpha: float) -> None:
        self._alpha = float(alpha)

    def set_plasticity_rule(self, rule: str) -> None:
        if rule not in PLASTICITY_RULES:
            choices = ", ".join(sorted(PLASTICITY_RULES))
            raise ValueError(f"unknown plasticity rule {rule!r}; choose from {choices}")
        self.plasticity_rule = rule

    def clear_recordings(self) -> None:
        self.recordings = {
            "x_ei": [],
            "ca3": [],
            "IS": [],
            "ca1": [],
            "eo": [],
            "W_ca3_ca1": [],
        }

    def _record(self, **values: torch.Tensor) -> None:
        if not self.record_enabled:
            return
        names = {"instructive_signal": "IS", "output": "eo"}
        for name, value in values.items():
            self.recordings[names.get(name, name)].append(value.detach().clone())
        self.recordings["W_ca3_ca1"].append(self.W_ca3_ca1.detach().clone())


# Familiar name for rewritten experiments. Unlike the legacy MTL, forward does
# not learn unless explicitly called with learn=True.
MTL = BTSPMemory
