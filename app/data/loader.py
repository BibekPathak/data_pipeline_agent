"""Source loading: resolve a logical source key to a Polars DataFrame.

Supports ``fixtures:<basename>`` (reads from the fixture directory) and a bare
path. Scenarios in ``app/evaluation/scenarios.py`` generate their drifted data in
memory, so this loader is primarily for fixtures and the API path.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_source(source: str, fixture_dir: str | Path | None = None, **kwargs) -> pl.DataFrame:
    if source.startswith("fixtures:"):
        name = source.split(":", 1)[1]
        base = Path(fixture_dir) if fixture_dir else Path("./datasets/fixtures")
        path = base / name
    else:
        path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"source not found: {path}")

    if path.suffix.lower() == ".csv":
        return pl.read_csv(path, try_parse_dates=True, **kwargs)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pl.read_parquet(path, **kwargs)
    if path.suffix.lower() == ".json":
        return pl.read_json(path, **kwargs)
    raise ValueError(f"unsupported source extension: {path.suffix}")
