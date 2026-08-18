"""Project paths, configurations, and checkpoint persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

import torch


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        return cls(Path(__file__).resolve().parents[2])

    @property
    def source(self) -> Path:
        return self.root / "src"

    @property
    def configs(self) -> Path:
        # Existing location remains supported until configs are intentionally moved.
        root_configs = self.root / "configs"
        return root_configs if root_configs.exists() else self.source / "configs"

    @property
    def data(self) -> Path:
        return self.source / "data"

    @property
    def autoencoders(self) -> Path:
        return self.data / "autoencoders"

    @property
    def experiments(self) -> Path:
        root_experiments = self.root / "experiments"
        return root_experiments if root_experiments.exists() else self.source / "experiments"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def results(self) -> Path:
        return self.root / "results"


PATHS = ProjectPaths.discover()


def runtime_metadata(paths: ProjectPaths = PATHS) -> dict[str, object]:
    """Capture enough environment information to identify a simulation run."""

    package_names = ("kamemory", "numpy", "torch", "matplotlib")
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None

    commit = None
    dirty = None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=paths.root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = commit_result.stdout.strip()
        dirty = bool(status_result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def check_project_paths(
    paths: ProjectPaths = PATHS, *, create_runtime: bool = False
) -> dict[str, dict[str, object]]:
    """Return existence/readability information for all important paths."""

    if create_runtime:
        paths.results.mkdir(parents=True, exist_ok=True)
    expected = {
        "root": paths.root,
        "source": paths.source,
        "configs": paths.configs,
        "data": paths.data,
        "autoencoders": paths.autoencoders,
        "experiments": paths.experiments,
        "media": paths.media,
        "results": paths.results,
    }
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
        }
        for name, path in expected.items()
    }


def load_config(name_or_path: str | Path, paths: ProjectPaths = PATHS) -> dict:
    """Load a JSON config by explicit path or by name from the config folder."""

    path = Path(name_or_path)
    if not path.is_absolute() and not path.exists():
        path = paths.configs / path
    if path.suffix == "":
        path = path.with_suffix(".json")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _natural_key(path: Path) -> list[tuple[int, int | str]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", path.name)
    ]


def list_autoencoder_sessions(directory: Path | None = None) -> list[Path]:
    directory = PATHS.autoencoders if directory is None else Path(directory)
    if not directory.exists():
        return []
    sessions = [
        path
        for path in directory.iterdir()
        if path.is_dir()
        and (path / "info.json").is_file()
        and (path / "autoencoder.pt").is_file()
    ]
    return sorted(sessions, key=_natural_key)


def _autoencoder_parameters(info: dict) -> dict[str, Any]:
    if "network_params" in info:
        params = info["network_params"]
        return {
            "input_dim": params["dim_ei"],
            "encoding_dim": params["dim_ca1"],
            "K": params["K_ca1"],
            "beta": params["beta_ca1"],
            "use_bias": params.get("bias", True),
        }
    hyper = info.get("hyperparameters", info)
    return {
        "input_dim": hyper["dim_ei"],
        "encoding_dim": hyper["dim_ca1"],
        "K": hyper.get("K_lat", hyper.get("K_ca1")),
        "beta": hyper.get("beta", hyper.get("beta_ca1")),
        "use_bias": hyper.get("bias", True),
    }


def load_autoencoder_session(
    session: int | str | Path,
    *,
    directory: Path | None = None,
    map_location: str | torch.device = "cpu",
):
    """Load a legacy or new autoencoder session deterministically."""

    from .models import Autoencoder

    sessions = list_autoencoder_sessions(directory)
    if isinstance(session, int):
        if not sessions:
            raise FileNotFoundError("no autoencoder sessions were found")
        try:
            session_path = sessions[session]
        except IndexError as exc:
            raise IndexError(f"session index {session} outside 0..{len(sessions)-1}") from exc
    else:
        session_path = Path(session)
        if not session_path.is_absolute() and not session_path.exists():
            base = PATHS.autoencoders if directory is None else Path(directory)
            session_path = base / session_path

    with (session_path / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    model = Autoencoder(**_autoencoder_parameters(info))
    state = torch.load(
        session_path / "autoencoder.pt",
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(map_location)
    model.eval()
    return info, model


def save_autoencoder_session(
    model,
    info: dict,
    *,
    name: str | None = None,
    directory: Path | None = None,
) -> Path:
    """Save a checkpoint and JSON metadata without relying on the cwd."""

    directory = PATHS.autoencoders if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = name or datetime.now().strftime("ae_%Y%m%d_%H%M%S")
    session_path = directory / name
    if session_path.exists():
        raise FileExistsError(f"session already exists: {session_path}")
    session_path.mkdir()
    torch.save(model.state_dict(), session_path / "autoencoder.pt")
    with (session_path / "info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return session_path
