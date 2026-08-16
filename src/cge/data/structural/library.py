"""Load the vendored structural-trajectory artifact into a :class:`StructuralTrajectory` (7b.2).

``data/structural/trajectories_v1.json`` holds documented, sourced per-region annual growth
trajectories for the demographic + productivity drivers, with a per-entry citation and confidence.
This loader validates it through the contract (finite rates in a plausible band, known drivers,
per-entry provenance), so an unsourced or malformed trajectory fails loudly rather than entering a
run. See ``data/structural/NOTICE.md`` for the underlying sources and licences.
"""

from __future__ import annotations

import json
from pathlib import Path

from cge.contracts.data_objects import Provenance, StructuralTrajectory

# Repo-root-relative default artifact path, resolved from this file so it works regardless of the
# process's working directory (the same convention as the ENCORE/EXIOBASE vendored loaders). This
# file is src/cge/data/structural/library.py, so the repo root is parents[4].
_DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[4] / "data" / "structural" / "trajectories_v1.json"
)


def load_structural_trajectories(path: str | Path | None = None) -> StructuralTrajectory:
    """The vendored structural trajectories as a validated :class:`StructuralTrajectory`.

    ``path`` overrides the default vendored artifact (e.g. a build-specific trajectory). The JSON
    keys ``rates``/``sources``/``confidence``/``provenance`` map directly onto the contract; JSON
    object keys are strings, so per-year rate keys are coerced back to ``int`` here."""
    artifact = Path(path) if path is not None else _DEFAULT_ARTIFACT
    if not artifact.exists():
        raise FileNotFoundError(
            f"structural-trajectory artifact not found at {artifact}; expected the vendored "
            "data/structural/trajectories_v1.json (see data/structural/NOTICE.md)."
        )
    raw = json.loads(artifact.read_text())

    # JSON object keys are strings; the contract keys years by int. Coerce every path (both the
    # per-region ``rates`` and the per-sector ``sector_rates`` tables have the same nested shape).
    def _coerce(table: dict) -> dict:
        return {
            driver: {
                key: {int(year): float(rate) for year, rate in path.items()}
                for key, path in by_key.items()
            }
            for driver, by_key in table.items()
        }

    return StructuralTrajectory(
        provenance=Provenance(**raw["provenance"]),
        rates=_coerce(raw.get("rates", {})),
        sector_rates=_coerce(raw.get("sector_rates", {})),
        sources=raw.get("sources", {}),
        confidence=raw.get("confidence", {}),
    )


def default_structural_trajectories() -> StructuralTrajectory:
    """Alias for the vendored default trajectories — the set a recursive run uses when the scenario
    asks for sourced structural trends without naming a specific artifact."""
    return load_structural_trajectories()
