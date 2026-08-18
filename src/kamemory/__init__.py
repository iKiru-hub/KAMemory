"""Reusable backend for the KAMemory simulations.

Experiment orchestration and plotting deliberately live outside this package.
"""

from .data import (
    generate_factorial_track,
    generate_legacy_sparse_patterns,
    generate_sensory_patterns,
    generate_sparse_patterns,
    generate_track_input,
)
from .autoencoder_experiment import (
    make_factorial_autoencoder_datasets,
    run_autoencoder_experiment,
)
from .io import (
    PATHS,
    ProjectPaths,
    load_autoencoder_session,
    load_config,
    runtime_metadata,
)
from .models import Autoencoder, BTSPMemory, MTL
from .training import (
    AutoencoderTrainingHistory,
    evaluate_autoencoder,
    evaluate_memory,
    fit_autoencoder,
    reconstruct,
    store_patterns,
)

__all__ = [
    "Autoencoder",
    "AutoencoderTrainingHistory",
    "BTSPMemory",
    "MTL",
    "PATHS",
    "ProjectPaths",
    "evaluate_autoencoder",
    "evaluate_memory",
    "fit_autoencoder",
    "generate_factorial_track",
    "generate_legacy_sparse_patterns",
    "generate_sensory_patterns",
    "generate_sparse_patterns",
    "generate_track_input",
    "load_autoencoder_session",
    "load_config",
    "make_factorial_autoencoder_datasets",
    "reconstruct",
    "run_autoencoder_experiment",
    "runtime_metadata",
    "store_patterns",
]
