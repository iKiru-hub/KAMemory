"""Stimulus generation without plotting or filesystem side effects."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _generator(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def generate_sparse_patterns(
    num_patterns: int,
    num_active: int,
    size: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate binary patterns with exactly ``num_active`` active units."""

    if num_patterns < 0:
        raise ValueError("num_patterns must be non-negative")
    if not 0 <= num_active <= size:
        raise ValueError("num_active must lie between zero and size")
    rng = _generator(rng)
    samples = np.zeros((num_patterns, size), dtype=np.float32)
    for row in samples:
        row[rng.choice(size, replace=False, size=num_active)] = 1.0
    return samples


def generate_legacy_sparse_patterns(
    num_patterns: int,
    num_active: int,
    size: int,
) -> np.ndarray:
    """Exact sparse-pattern protocol used by the validated notebooks.

    This intentionally uses NumPy's seeded global RNG and preserves the
    original row-by-row ``np.random.choice`` call order. New experiments use
    it when numerical continuity with the saved notebook results is required.
    """

    if num_patterns < 0:
        raise ValueError("num_patterns must be non-negative")
    if not 0 <= num_active <= size:
        raise ValueError("num_active must lie between zero and size")
    samples = np.zeros((num_patterns, size), dtype=np.float32)
    for row in samples:
        indices = np.random.choice(range(size), replace=False, size=num_active)
        row[indices] = 1.0
    return samples


def circular_distance(x1: np.ndarray, x2: np.ndarray, period: int) -> np.ndarray:
    return np.minimum(np.abs(x1 - x2), period - np.abs(x1 - x2))


def place_field_activity(
    width: int,
    height: int,
    sigma: float,
    x_position: float,
    y_position: float,
) -> np.ndarray:
    """Gaussian place-field population on a track periodic in x."""

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    distance_squared = circular_distance(x, x_position, width) ** 2 + (
        y - y_position
    ) ** 2
    return np.exp(-distance_squared / (2 * sigma**2))


def generate_sensory_patterns(
    num_stimuli: int,
    num_active_sensory: int,
    spatial_size: int,
    sensory_size: int,
    track_width: int,
    track_height: int,
    place_field_sigma: float,
    *,
    positions: Sequence[tuple[int, int]] | None = None,
    lap_length: int | None = None,
    num_laps: int | None = None,
    num_cues: int | None = None,
    cue_positions: Sequence[int] | None = None,
    sensory_patterns: np.ndarray | None = None,
    binarize_spatial: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Generate concatenated MEC-like spatial and LEC-like sensory input.

    When positions and cue positions are supplied, cues alternate across laps
    and their expression probability follows place-field activity at the cue.
    """

    rng = _generator(rng)
    if spatial_size != track_width * track_height and spatial_size > 0:
        raise ValueError("spatial_size must equal track_width * track_height")
    if positions is not None and len(positions) != num_stimuli:
        raise ValueError("positions must contain one entry per stimulus")
    if sensory_patterns is not None and len(sensory_patterns) != num_stimuli:
        raise ValueError("sensory_patterns must contain one entry per stimulus")

    cue_mode = positions is not None and cue_positions is not None
    if cue_mode and (not lap_length or not num_laps or not num_cues):
        raise ValueError("cue mode requires lap_length, num_laps, and num_cues")
    if num_active_sensory > sensory_size:
        raise ValueError("num_active_sensory cannot exceed sensory_size")

    samples = np.zeros((num_stimuli, spatial_size + sensory_size), dtype=np.float32)
    cue_strength = np.zeros(num_stimuli, dtype=np.float32)
    lap_cues = np.zeros(num_laps, dtype=int) if cue_mode else None

    fixed_cues = None
    if cue_mode:
        if num_cues * num_active_sensory > sensory_size:
            raise ValueError("sensory_size is too small for non-overlapping fixed cues")
        fixed_cues = np.zeros((num_cues, sensory_size), dtype=np.float32)
        for cue in range(num_cues):
            start = cue * num_active_sensory
            fixed_cues[cue, start : start + num_active_sensory] = 1.0

    position_order = np.arange(track_width)
    for index in range(num_stimuli):
        if positions is None:
            if index % track_width == 0:
                rng.shuffle(position_order)
            x_position = int(position_order[index % track_width])
            y_position = int(rng.integers(track_height))
        else:
            x_position, y_position = positions[index]

        if spatial_size:
            spatial = place_field_activity(
                track_width, track_height, place_field_sigma, x_position, y_position
            ).reshape(-1)
            if binarize_spatial:
                spatial = (spatial > 0.5).astype(np.float32)
            samples[index, :spatial_size] = spatial

        if not sensory_size:
            continue
        if sensory_patterns is not None:
            sensory = sensory_patterns[index]
        elif cue_mode:
            lap = index // lap_length
            cue = lap % num_cues
            lap_cues[lap] = cue
            probability = samples[index, int(cue_positions[cue])]
            maximum = samples[index, :spatial_size].max()
            probability = float(probability / maximum) if maximum else 0.0
            cue_strength[index] = probability
            if rng.binomial(1, probability):
                sensory = fixed_cues[cue]
            else:
                sensory = generate_sparse_patterns(
                    1, num_active_sensory, sensory_size, rng=rng
                )[0]
        else:
            sensory = generate_sparse_patterns(
                1, num_active_sensory, sensory_size, rng=rng
            )[0]
        samples[index, spatial_size:] = sensory

    return samples, lap_cues, cue_strength


def generate_track_input(
    track_params: dict,
    network_params: dict,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Generate model input from the existing track/network config schema."""

    positions = [
        (position, 0)
        for _ in range(track_params["num_laps"])
        for position in range(track_params["length"])
    ]
    return generate_sensory_patterns(
        num_stimuli=len(positions),
        num_active_sensory=network_params["K_lec"],
        spatial_size=network_params["dim_mec"],
        sensory_size=network_params["dim_lec"],
        track_width=network_params["mec_N_x"],
        track_height=network_params["mec_N_y"],
        place_field_sigma=network_params["mec_sigma"],
        positions=positions,
        lap_length=track_params["length"],
        num_laps=track_params["num_laps"],
        num_cues=network_params["num_cues"],
        cue_positions=track_params["cue_position"],
        rng=rng,
    )


def generate_legacy_remapping_track(
    *,
    track_length: int,
    num_laps: int,
    cue_position: int,
    spatial_size: int,
    sensory_size: int,
    sensory_active: int,
    place_field_sigma: float,
    rng=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the one-cue track generator used in ``lab_fig_c1``.

    This is deliberately a narrow compatibility function. It preserves the
    notebook's RNG call order, fixed first-block cue, random K-hot sensory
    background, probabilistic cue expression, and position-derived alpha
    samples. New causal experiments should use :func:`generate_factorial_track`.
    """

    if track_length <= 0 or num_laps <= 0:
        raise ValueError("track_length and num_laps must be positive")
    if spatial_size != track_length:
        raise ValueError("the legacy one-dimensional track requires spatial_size == track_length")
    if not 0 <= cue_position < track_length:
        raise ValueError("cue_position must lie on the track")
    if not 0 < sensory_active <= sensory_size:
        raise ValueError("sensory_active must lie in [1, sensory_size]")

    rng = np.random if rng is None else rng
    num_samples = num_laps * track_length
    samples = np.zeros((num_samples, spatial_size + sensory_size), dtype=np.float32)
    alpha_samples = np.zeros(num_samples, dtype=np.float64)
    lap_cues = np.zeros(num_laps, dtype=np.float64)

    # The original function consumed this choice even though it then used the
    # deterministic first K entries. Keeping the call is required for parity.
    rng.choice(range(sensory_size), replace=False, size=sensory_active)
    fixed_cue = np.zeros(sensory_size, dtype=np.float64)
    fixed_cue[:sensory_active] = 1.0

    for index in range(num_samples):
        position = index % track_length
        spatial = place_field_activity(
            track_length, 1, place_field_sigma, position, 0
        ).reshape(-1)
        samples[index, :spatial_size] = spatial
        probability = float(spatial[cue_position] / spatial.max())
        alpha_samples[index] = probability
        if rng.binomial(1, probability):
            sensory = fixed_cue
        else:
            sensory = np.zeros(sensory_size, dtype=np.float64)
            sensory[
                rng.choice(range(sensory_size), replace=False, size=sensory_active)
            ] = 1.0
        samples[index, spatial_size:] = sensory

    return samples, lap_cues, alpha_samples


def generate_factorial_track(
    *,
    track_length: int,
    cue_positions: Sequence[int],
    repeats_per_combination: int,
    cue_free_laps: int,
    spatial_size: int,
    sensory_size: int,
    sensory_active: int,
    place_field_sigma: float,
    rng: np.random.Generator,
    cue_patterns: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Generate a balanced cue-identity × cue-position linear-track task.

    Every cue identity occurs at every nominated position. Cue-free laps are
    assigned balanced probe positions but contain only random sensory
    background. The cue identity and its presentation position are therefore
    statistically independent, unlike the original exploratory track task.

    Spatial and sensory components are returned separately as well as
    concatenated, making their projections and downstream metrics explicit.
    """

    if track_length <= 0 or repeats_per_combination <= 0 or cue_free_laps < 1:
        raise ValueError("track length/repeats must be positive and cue-free laps required")
    if spatial_size != track_length:
        raise ValueError("the current 1-D track requires spatial_size == track_length")
    if not 0 < sensory_active <= sensory_size:
        raise ValueError("sensory_active must lie in [1, sensory_size]")
    positions_array = np.asarray(cue_positions, dtype=int)
    if len(positions_array) < 2 or np.any((positions_array < 0) | (positions_array >= track_length)):
        raise ValueError("provide at least two valid cue positions")

    if cue_patterns is None:
        num_cues = 2
        if num_cues * sensory_active > sensory_size:
            raise ValueError("sensory population is too small for disjoint cues")
        cue_patterns = np.zeros((num_cues, sensory_size), dtype=np.float32)
        for cue in range(num_cues):
            start = cue * sensory_active
            cue_patterns[cue, start : start + sensory_active] = 1
    else:
        cue_patterns = np.asarray(cue_patterns, dtype=np.float32)
        if cue_patterns.ndim != 2 or cue_patterns.shape[1] != sensory_size:
            raise ValueError("cue_patterns must have shape (num_cues, sensory_size)")
        if not np.all(cue_patterns.sum(axis=1) == sensory_active):
            raise ValueError("each cue pattern must contain sensory_active active units")
        num_cues = len(cue_patterns)

    schedule = [
        (cue, int(position))
        for _ in range(repeats_per_combination)
        for cue in range(num_cues)
        for position in positions_array
    ]
    schedule.extend(
        (-1, int(positions_array[index % len(positions_array)]))
        for index in range(cue_free_laps)
    )
    rng.shuffle(schedule)
    num_laps = len(schedule)
    spatial = np.zeros((num_laps, track_length, spatial_size), dtype=np.float32)
    sensory = np.zeros((num_laps, track_length, sensory_size), dtype=np.float32)
    cue_ids = np.asarray([item[0] for item in schedule], dtype=np.int16)
    event_positions = np.asarray([item[1] for item in schedule], dtype=np.int16)
    cue_present = np.zeros((num_laps, track_length), dtype=bool)

    for lap, (cue, event_position) in enumerate(schedule):
        for position in range(track_length):
            spatial[lap, position] = place_field_activity(
                track_length, 1, place_field_sigma, position, 0
            ).reshape(-1)
            background = rng.choice(
                sensory_size, size=sensory_active, replace=False
            )
            sensory[lap, position, background] = 1
        if cue >= 0:
            sensory[lap, event_position] = cue_patterns[cue]
            cue_present[lap, event_position] = True

    inputs = np.concatenate((spatial, sensory), axis=-1)
    flat_positions = np.tile(np.arange(track_length, dtype=np.int16), num_laps)
    return {
        "inputs": inputs.reshape(num_laps * track_length, -1),
        "spatial": spatial.reshape(num_laps * track_length, spatial_size),
        "sensory": sensory.reshape(num_laps * track_length, sensory_size),
        "positions": flat_positions,
        "cue_ids": cue_ids,
        "event_positions": event_positions,
        "cue_present": cue_present.reshape(-1),
        "cue_patterns": cue_patterns.copy(),
        "num_laps": np.asarray(num_laps, dtype=np.int32),
        "track_length": np.asarray(track_length, dtype=np.int32),
    }


# Transitional names for code copied from the exploratory backend.
sparse_stimulus_generator = generate_sparse_patterns
sparse_stimulus_generator_sensory = generate_sensory_patterns
