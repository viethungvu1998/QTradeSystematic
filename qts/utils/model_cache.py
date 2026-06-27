"""Small pickle-based model cache helpers for research notebooks."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def json_ready(value: Any) -> Any:
    """Convert common scientific Python values into JSON-safe metadata."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return json_ready(value.tolist())
    return value


def frame_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    """Return a stable-enough fingerprint for cache invalidation."""

    available = [column for column in columns if column in frame.columns]
    if not available:
        return "0"
    subset = frame[available].reset_index(drop=True)
    values = pd.util.hash_pandas_object(subset, index=False).to_numpy(dtype="uint64")
    return f"{int(values.sum()) & ((1 << 64) - 1):016x}"


def metadata_matches(saved: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return True when saved metadata contains the expected values."""

    saved_ready = json_ready(saved)
    expected_ready = json_ready(expected)
    return all(saved_ready.get(key) == value for key, value in expected_ready.items())


def write_pickle_model_cache(
    artifact_dir: Path,
    stem: str,
    *,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = artifact_dir / f"{stem}.pkl"
    metadata_path = artifact_dir / f"{stem}.metadata.json"
    with pickle_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_payload = {
        **json_ready(metadata),
        "pickle_path": str(pickle_path),
    }
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return pickle_path


def read_pickle_model_cache(
    artifact_dir: Path,
    stem: str,
    *,
    expected_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    metadata_path = artifact_dir / f"{stem}.metadata.json"
    pickle_path = artifact_dir / f"{stem}.pkl"
    if not metadata_path.exists() or not pickle_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata_matches(metadata, expected_metadata):
        return None

    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload, metadata
