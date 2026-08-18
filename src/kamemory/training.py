""" Small, explicit training and evaluation loops for the main models """

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import BTSPMemory
from .utils import as_float_tensor


@dataclass
class AutoencoderTrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)

    @property
    def final_train_loss(self) -> float:
        return self.train_loss[-1]

    @property
    def final_validation_loss(self) -> float:
        return self.validation_loss[-1]


def evaluate_autoencoder(model: nn.Module, data, *, device=None) -> float:
    tensor = as_float_tensor(data, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = nn.functional.mse_loss(model(tensor), tensor).item()
    model.train(was_training)
    return loss


def fit_autoencoder(
    model: nn.Module,
    training_data,
    validation_data=None,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
    shuffle: bool = True,
    legacy_validation_order: bool = True,
) -> AutoencoderTrainingHistory:
    """Optimize an autoencoder and return per-epoch mean losses.

    ``legacy_validation_order=True`` reproduces the validated notebook loop:
    held-out data are evaluated through a batch-size-one ``DataLoader`` after
    every epoch. Besides matching the metric calculation, constructing each
    iterator preserves the historical PyTorch RNG consumption and therefore
    the exact shuffled minibatch order in later epochs.
    """

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    device = torch.device(device)
    model.to(device)
    train_tensor = as_float_tensor(training_data)
    validation_data = training_data if validation_data is None else validation_data
    validation_tensor = as_float_tensor(validation_data)
    loader = DataLoader(
        TensorDataset(train_tensor), batch_size=batch_size, shuffle=shuffle
    )
    validation_loader = DataLoader(
        TensorDataset(validation_tensor), batch_size=1, shuffle=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = AutoencoderTrainingHistory()

    for _ in range(epochs):
        model.train()
        total_batch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_batch_loss += loss.item()
        history.train_loss.append(total_batch_loss / len(loader))
        if legacy_validation_order:
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for (batch,) in validation_loader:
                    batch = batch.to(device)
                    validation_loss += nn.functional.mse_loss(model(batch), batch).item()
            history.validation_loss.append(validation_loss / len(validation_loader))
            model.train()
        else:
            history.validation_loss.append(
                evaluate_autoencoder(model, validation_data, device=device)
            )
    return history


def train_autoencoder(
    training_data,
    test_data,
    model,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
):
    """Compatibility wrapper returning ``(final_loss, model)``."""

    history = fit_autoencoder(
        model,
        training_data,
        test_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return history.final_train_loss, model


def reconstruct(model: nn.Module, data, *, return_latent: bool = True):
    tensor = as_float_tensor(data, device=next(model.parameters(), torch.empty(0)).device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if return_latent:
            output, latent = model(tensor, ca1=True)
        else:
            output, latent = model(tensor), None
    model.train(was_training)
    return output.cpu().numpy(), None if latent is None else latent.cpu().numpy()


@torch.no_grad()
def store_patterns(model: BTSPMemory, patterns, *, epochs: int = 1) -> BTSPMemory:
    """Store patterns sequentially using explicit BTSP updates."""

    tensor = as_float_tensor(patterns, device=model.W_ca3_ca1.device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    for _ in range(epochs):
        for pattern in tensor:
            model.store(pattern)
    return model


@torch.no_grad()
def evaluate_memory(model: BTSPMemory, patterns) -> dict[str, object]:
    """Retrieve patterns without learning and report reconstruction MSE."""

    tensor = as_float_tensor(patterns, device=model.W_ca3_ca1.device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    outputs = torch.stack([model.retrieve(pattern).flatten() for pattern in tensor])
    return {
        "mse": nn.functional.mse_loss(outputs, tensor).item(),
        "outputs": outputs.cpu().numpy(),
    }
