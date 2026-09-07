"""Downloader utilities for Model Counting Competition test instances."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

MCCOMP_REPOSITORY_URL = "https://github.com/arijitsh/mccomp-test-instances"
MCCOMP_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/arijitsh/mccomp-test-instances/main"
)

MCCOMP_TRACKS = {
    "Track1_MC": ("random_mc", "mc", "exact model counting"),
    "Track3_PMC": ("random_pmc", "pmc", "projected model counting"),
    "Track4_PWMC": ("random_pwmc", "pwmc", "projected weighted model counting"),
    "Track5B_AMC": ("random_amc", "amc", "algebraic model counting"),
}

MCCOMP_SOURCE_PATHS = tuple(
    f"{track}/{prefix}_{index}.cnf"
    for track, (prefix, _problem_type, _description) in MCCOMP_TRACKS.items()
    for index in range(1, 11)
)

def parse_dimacs(text):
    lines = [line.strip() for line in text.strip().split("\n")]
    cleaned = [line for line in lines if not line.startswith("c") and line]

    num_vars = 0
    num_clauses = 0
    rest = []

    for i, line in enumerate(cleaned):
        if line.startswith("p cnf"):
            parts = line.split()
            num_vars = int(parts[2])
            num_clauses = int(parts[3])

            rest = " ".join(cleaned[i + 1 :]).split()
            break

    clauses = []
    current_clause = []
    idx = 0

    while len(clauses) < num_clauses and idx < len(rest):
        val = int(rest[idx])

        if val == 0:
            clauses.append(current_clause)
            current_clause = []
        else:
            current_clause.append(val)

        idx += 1

    return num_vars, clauses


def list_mccomp_instances(track: str | None = None) -> list[str]:
    """Return known source paths from the MC competition test-instance repo."""
    if track is None:
        return list(MCCOMP_SOURCE_PATHS)

    track_dir = normalize_mccomp_track(track)
    return [path for path in MCCOMP_SOURCE_PATHS if path.startswith(f"{track_dir}/")]


def download_mccomp_instance(
    source_name: str,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Download one MC competition CNF instance and return its local path."""
    source_path = normalize_mccomp_source_path(source_name)
    return _ensure_downloaded(source_path, data_dir)


def mccomp_source_url(source_path: str) -> str:
    return f"{MCCOMP_REPOSITORY_URL}/blob/main/{source_path}"


def mccomp_raw_url(source_path: str) -> str:
    return f"{MCCOMP_RAW_BASE_URL}/{source_path}"


def normalize_mccomp_track(track: str) -> str:
    lowered = track.lower()
    for track_dir, (_prefix, problem_type, _description) in MCCOMP_TRACKS.items():
        if lowered in {track_dir.lower(), problem_type.lower()}:
            return track_dir
    raise ValueError(f"Unknown MC competition track: {track!r}")


def normalize_mccomp_source_path(source_name: str) -> str:
    name = source_name.strip().removeprefix("mccomp://").removeprefix("mccomp:")
    name = name.removesuffix(".cnf") + ".cnf"
    if name in MCCOMP_SOURCE_PATHS:
        return name
    matches = [path for path in MCCOMP_SOURCE_PATHS if Path(path).name == name]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"Ambiguous MC competition instance name: {source_name!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.cnf", name):
        raise ValueError(f"Invalid MC competition source path: {source_name!r}")
    return name


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "mccomp"


def _ensure_downloaded(source_path: str, data_dir: str | Path | None) -> Path:
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dest_path = root / source_path
    if dest_path.exists():
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".tmp")
    try:
        urllib.request.urlretrieve(mccomp_raw_url(source_path), tmp_path)  # noqa: S310
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return dest_path
