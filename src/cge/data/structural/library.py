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


_DEFAULT_CONCORDANCE = (
    Path(__file__).resolve().parents[4] / "data" / "structural" / "concordance_v1.json"
)


def load_structural_concordance(path: str | Path | None = None) -> dict:
    """The vendored region/sector → archetype concordance (Phase 7b.2, review P1c 2026-08-28).

    Maps each real EXIOBASE coarse-v3 build label onto one of the trajectory's archetype keys (N/S
    for regions, BRD/MIL for sectors), so a real build carries a documented, differentiated
    trajectory instead of every label silently falling through to the global ``__all__`` default.
    Validates the artifact carries provenance and non-empty archetype maps. See
    ``data/structural/NOTICE.md`` and ``concordance_v1.json`` for the sources."""
    artifact = Path(path) if path is not None else _DEFAULT_CONCORDANCE
    if not artifact.exists():
        raise FileNotFoundError(
            f"structural concordance not found at {artifact}; expected the vendored "
            "data/structural/concordance_v1.json (see data/structural/NOTICE.md)."
        )
    raw = json.loads(artifact.read_text())
    for key in ("provenance", "region_archetype", "sector_archetype"):
        if key not in raw or not raw[key]:
            raise ValueError(f"structural concordance is missing or empty '{key}'")
    return raw


class UnmappedStructuralLabels(ValueError):
    """Raised when a supposedly-real structural run has build labels that the concordance does not
    map (review P1c): every real region/sector would then silently take the global ``__all__``
    trajectory, so there would be NO country differentiation and NO sectoral composition drift — the
    exact overclaim the concordance is meant to prevent. Reject loudly instead."""


def structural_trajectories_for_build(
    regions: list[str],
    sectors: list[str],
    *,
    trajectory_path: str | Path | None = None,
    concordance_path: str | Path | None = None,
    require_full_coverage: bool = True,
) -> StructuralTrajectory:
    """A :class:`StructuralTrajectory` keyed by a REAL build's actual region/sector labels (Phase
    7b.2, review P1c 2026-08-28).

    Loads the archetype trajectory (``trajectories_v1.json``: paths keyed by the N/S and BRD/MIL
    archetypes) and the concordance (``concordance_v1.json``: real-label → archetype), then emits a
    new trajectory whose region/sector keys are the build's OWN labels, each carrying its
    archetype's rate path. On a real EXIOBASE build this is what gives actual country
    differentiation (advanced regions grow TFP slower than emerging ones) and sectoral composition
    drift (goods vs services), rather than every label collapsing to ``__all__``.

    ``require_full_coverage`` (default True): a build label not mapped by the concordance raises
    :class:`UnmappedStructuralLabels` — so a real run can never silently degrade to an
    all-``__all__`` trajectory. The ``__all__`` archetype is retained as a fallback for the
    piecewise-constant lookup but every named build label is bound explicitly."""
    archetype = load_structural_trajectories(trajectory_path)
    conc = load_structural_concordance(concordance_path)
    region_arch: dict[str, str] = conc["region_archetype"]
    sector_arch: dict[str, str] = conc["sector_archetype"]

    unmapped_r = [r for r in regions if r not in region_arch]
    unmapped_s = [s for s in sectors if s not in sector_arch]
    if require_full_coverage and (unmapped_r or unmapped_s):
        raise UnmappedStructuralLabels(
            "structural concordance does not map every build label; unmapped regions "
            f"{unmapped_r} and sectors {unmapped_s} would fall through to the global __all__ "
            "trajectory (no country/sector differentiation). Extend concordance_v1.json or pass "
            "require_full_coverage=False to accept the __all__ fallback explicitly."
        )

    def _remap_region_axis(table: dict) -> dict:
        # {driver: {archetype|__all__: path}} → {driver: {real_region: path, __all__: path}}
        out: dict = {}
        for driver, by_key in table.items():
            new: dict = {}
            for r in regions:
                arch = region_arch.get(r)
                path = by_key.get(arch) if arch else None
                if path is None:
                    path = by_key.get("__all__")
                if path is not None:
                    new[r] = dict(path)
            if "__all__" in by_key:
                new["__all__"] = dict(by_key["__all__"])
            out[driver] = new
        return out

    def _remap_sector_axis(table: dict) -> dict:
        out: dict = {}
        for driver, by_key in table.items():
            new: dict = {}
            for s in sectors:
                arch = sector_arch.get(s)
                path = by_key.get(arch) if arch else None
                if path is None:
                    path = by_key.get("__all__")
                if path is not None:
                    new[s] = dict(path)
            if "__all__" in by_key:
                new["__all__"] = dict(by_key["__all__"])
            out[driver] = new
        return out

    new_rates = _remap_region_axis(archetype.rates)
    new_sector_rates = _remap_sector_axis(archetype.sector_rates)

    # Rebuild sources/confidence so every emitted key carries provenance (the contract requires it):
    # each real label inherits its archetype's citation (falling back to the __all__ citation).
    sources: dict = {}
    confidence: dict = {}
    for table, arch_map in ((new_rates, region_arch), (new_sector_rates, sector_arch)):
        for driver, by_key in table.items():
            for k in by_key:
                arch = arch_map.get(k, "__all__")
                for meta, dst in (
                    (archetype.sources, sources),
                    (archetype.confidence, confidence),
                ):
                    val = meta.get(f"{driver}:{arch}") or meta.get(f"{driver}:__all__")
                    if val is not None:
                        dst[f"{driver}:{k}"] = val

    return StructuralTrajectory(
        provenance=archetype.provenance,
        rates=new_rates,
        sector_rates=new_sector_rates,
        sources=sources,
        confidence=confidence,
    )
