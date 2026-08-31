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
    Path(__file__).resolve().parents[4] / "data" / "structural" / "concordance_v2.json"
)


def load_structural_concordance(path: str | Path | None = None) -> dict:
    """The vendored region/sector concordance (Phase 7b.2; v2 review P1 2026-08-29).

    Maps each real EXIOBASE coarse-v3 build label onto the trajectory's archetypes. Two shapes are
    accepted:

    - **v2 (default)** — country-level ``country_archetype`` + GDP-weighted ``block_membership``, so
      a mixed coarse block (e.g. RoW_Asia) is the GDP-WEIGHTED blend of its members' archetype paths
      rather than one archetype (review P1 2026-08-29). The World Bank income vintage is pinned.
    - **v1** — a direct ``region_archetype`` label→archetype map (kept for back-compat).

    Validation (review P2 2026-08-29): the provenance parses through the :class:`Provenance`
    contract; ``sector_archetype`` targets are among the known BRD/MIL archetype keys; for v2 every
    ``block_membership`` country has a ``country_archetype``, its archetype is a known key, and each
    block's member weights are finite, non-negative and sum to 1. A malformed concordance fails
    loudly rather than silently substituting ``__all__``."""
    artifact = Path(path) if path is not None else _DEFAULT_CONCORDANCE
    if not artifact.exists():
        raise FileNotFoundError(
            f"structural concordance not found at {artifact}; expected the vendored "
            "data/structural/concordance_v2.json (see data/structural/NOTICE.md)."
        )
    raw = json.loads(artifact.read_text())
    if "provenance" not in raw or not raw["provenance"]:
        raise ValueError("structural concordance is missing 'provenance'")
    # Parse provenance through the contract so an unsourced/malformed block fails on load.
    Provenance(**raw["provenance"])
    if not raw.get("sector_archetype"):
        raise ValueError("structural concordance is missing or empty 'sector_archetype'")

    known_region_arch = set(load_structural_trajectories().rates.get("productivity", {}))
    known_sector_arch = set(
        load_structural_trajectories().sector_rates.get("sector_productivity", {})
    )
    bad_sec = {
        s: a
        for s, a in raw["sector_archetype"].items()
        if a not in known_sector_arch and a != "__all__"
    }
    if bad_sec:
        raise ValueError(
            f"sector_archetype targets not among the trajectory's sector paths "
            f"{sorted(known_sector_arch)}: {bad_sec}"
        )

    is_v2 = "country_archetype" in raw and "block_membership" in raw
    if is_v2:
        _validate_v2(raw, known_region_arch)
    elif raw.get("region_archetype"):
        bad_reg = {
            r: a
            for r, a in raw["region_archetype"].items()
            if a not in known_region_arch and a != "__all__"
        }
        if bad_reg:
            raise ValueError(
                f"region_archetype targets not among the trajectory's region paths "
                f"{sorted(known_region_arch)}: {bad_reg}"
            )
    else:
        raise ValueError(
            "structural concordance needs either 'country_archetype'+'block_membership' (v2) or "
            "'region_archetype' (v1)."
        )
    return raw


def _validate_v2(raw: dict, known_region_arch: set) -> None:
    """Validate the v2 country-level concordance (review P2 2026-08-29): every member country has
    a known archetype, and each block's weights are finite, non-negative and sum to 1."""
    import math

    country_arch: dict = raw["country_archetype"]
    bad_arch = {
        c: a for c, a in country_arch.items() if a not in known_region_arch and a != "__all__"
    }
    if bad_arch:
        raise ValueError(
            f"country_archetype targets not among the trajectory's region paths "
            f"{sorted(known_region_arch)}: {bad_arch}"
        )
    for block, members in raw["block_membership"].items():
        if not members:
            raise ValueError(f"block_membership[{block!r}] has no member countries")
        missing = [c for c in members if c not in country_arch]
        if missing:
            raise ValueError(
                f"block_membership[{block!r}] countries missing a country_archetype: {missing}"
            )
        w = list(members.values())
        if any(not math.isfinite(x) or x < 0 for x in w):
            raise ValueError(
                f"block_membership[{block!r}] weights must be finite and ≥ 0: {members}"
            )
        if abs(sum(w) - 1.0) > 1e-6:
            raise ValueError(
                f"block_membership[{block!r}] weights sum to {sum(w):.6f}, expected 1.0"
            )


class UnmappedStructuralLabels(ValueError):
    """Raised when a supposedly-real structural run has build labels that the concordance does not
    map (review P1c): every real region/sector would then silently take the global ``__all__``
    trajectory, so there would be NO country differentiation and NO sectoral composition drift — the
    exact overclaim the concordance is meant to prevent. Reject loudly instead."""


def _blend_region_path(
    by_arch: dict, block: str, conc: dict, region_arch_v1: dict | None
) -> dict | None:
    """The rate path for a coarse ``block`` on one region driver: the GDP-WEIGHTED blend of its
    member countries' archetype paths (v2), or the block's single archetype path (v1). Returns a
    ``{year: rate}`` dict, or None if no archetype path is available.

    v2 blend (review P1 2026-08-29): for each knot year present across the members' archetype paths,
    the blended rate is Σ_c weight[c] · archetype_rate(archetype[c], year) — so a mixed block gets a
    weighted average, honestly reflecting that (e.g.) RoW_Asia contains both advanced and emerging
    economies. The union of member knot years is used so no knot is dropped."""
    if region_arch_v1 is not None:  # v1: single archetype
        arch = region_arch_v1.get(block)
        path = by_arch.get(arch) if arch else None
        return dict(path) if path is not None else None
    members: dict = conc["block_membership"].get(block)
    if not members:
        return None
    country_arch: dict = conc["country_archetype"]
    # Union of knot years across the members' archetype paths.
    knot_years: set[int] = set()
    for c in members:
        p = by_arch.get(country_arch[c])
        if p:
            knot_years |= set(p)
    if not knot_years:
        return None

    def _rate_at(path: dict, year: int) -> float:
        # Piecewise-constant hold (same rule as StructuralTrajectory._lookup).
        applicable = [y for y in sorted(path) if y <= year]
        chosen = applicable[-1] if applicable else min(path)
        return float(path[chosen])

    blended: dict = {}
    for year in sorted(knot_years):
        acc = 0.0
        for c, w in members.items():
            p = by_arch.get(country_arch[c])
            if p:
                acc += float(w) * _rate_at(p, year)
        blended[int(year)] = acc
    return blended


def structural_trajectories_for_build(
    regions: list[str],
    sectors: list[str],
    *,
    trajectory_path: str | Path | None = None,
    concordance_path: str | Path | None = None,
    require_full_coverage: bool = True,
) -> StructuralTrajectory:
    """A :class:`StructuralTrajectory` keyed by a REAL build's actual region/sector labels (Phase
    7b.2, review P1c 2026-08-28; v2 GDP-weighted blend review P1 2026-08-29).

    Loads the archetype trajectory (``trajectories_v1.json``: N/S region paths, BRD/MIL sector
    paths) and the concordance (``concordance_v2.json``), then emits a new trajectory whose keys are
    the build's OWN labels. For each region label the path is the GDP-WEIGHTED blend of the block's
    member countries' archetype paths (so a MIXED block — RoW_Asia mixes advanced AU/KR/TW with
    emerging ID/WA — gets a blended path, not one archetype; review P1 2026-08-29). Sectors map to
    their goods/services archetype path.

    The emitted trajectory carries **composite provenance** (review P2 2026-08-29): the concordance
    source/version/licence, plus a stamp of the underlying archetype-trajectory identity and a
    content hash of both artifacts, so the mapped object is NOT mislabelled as the bare
    ``structural-trajectories-v1`` — it records that a concordance was applied.

    ``require_full_coverage`` (default True): a build label not mapped by the concordance raises
    :class:`UnmappedStructuralLabels` — so a real run can never silently degrade to an
    all-``__all__`` trajectory."""
    archetype = load_structural_trajectories(trajectory_path)
    conc = load_structural_concordance(concordance_path)
    region_arch_v1 = conc.get("region_archetype") if "block_membership" not in conc else None
    sector_arch: dict[str, str] = conc["sector_archetype"]
    mapped_regions = (
        set(conc["block_membership"]) if region_arch_v1 is None else set(region_arch_v1)
    )

    unmapped_r = [r for r in regions if r not in mapped_regions]
    unmapped_s = [s for s in sectors if s not in sector_arch]
    if require_full_coverage and (unmapped_r or unmapped_s):
        raise UnmappedStructuralLabels(
            "structural concordance does not map every build label; unmapped regions "
            f"{unmapped_r} and sectors {unmapped_s} would fall through to the global __all__ "
            "trajectory (no country/sector differentiation). Extend the concordance or pass "
            "require_full_coverage=False to accept the __all__ fallback explicitly."
        )

    # Region axis: blend per (driver, region) from the member countries' archetype paths.
    new_rates: dict = {}
    for driver, by_arch in archetype.rates.items():
        new: dict = {}
        for r in regions:
            path = _blend_region_path(by_arch, r, conc, region_arch_v1)
            if path is None:
                path = dict(by_arch["__all__"]) if "__all__" in by_arch else None
            if path is not None:
                new[r] = path
        if "__all__" in by_arch:
            new["__all__"] = dict(by_arch["__all__"])
        new_rates[driver] = new

    # Sector axis: each sector inherits its archetype path (unchanged shape).
    new_sector_rates: dict = {}
    for driver, by_arch in archetype.sector_rates.items():
        new = {}
        for s in sectors:
            arch = sector_arch.get(s)
            path = by_arch.get(arch) if arch else None
            if path is None:
                path = by_arch.get("__all__")
            if path is not None:
                new[s] = dict(path)
        if "__all__" in by_arch:
            new["__all__"] = dict(by_arch["__all__"])
        new_sector_rates[driver] = new

    # Provenance/confidence per emitted key: a blended region inherits the __all__ (or its dominant
    # archetype's) citation, tagged as concordance-derived; a sector inherits its archetype's.
    sources: dict = {}
    confidence: dict = {}
    for driver, by_key in new_rates.items():
        for k in by_key:
            src = archetype.sources.get(f"{driver}:__all__")
            cf = archetype.confidence.get(f"{driver}:__all__")
            # Fall back to any archetype citation for this driver if no __all__ entry exists.
            if src is None:
                src = next(
                    (v for kk, v in archetype.sources.items() if kk.startswith(f"{driver}:")), None
                )
            if cf is None:
                cf = next(
                    (v for kk, v in archetype.confidence.items() if kk.startswith(f"{driver}:")),
                    None,
                )
            if src is not None:
                conc_ver = conc["provenance"]["source_version"]
                sources[f"{driver}:{k}"] = (
                    src if k == "__all__" else f"{src} [concordance-blended: {conc_ver}]"
                )
            if cf is not None:
                confidence[f"{driver}:{k}"] = cf
    for driver, by_key in new_sector_rates.items():
        for k in by_key:
            arch = sector_arch.get(k, "__all__")
            src = archetype.sources.get(f"{driver}:{arch}") or archetype.sources.get(
                f"{driver}:__all__"
            )
            cf = archetype.confidence.get(f"{driver}:{arch}") or archetype.confidence.get(
                f"{driver}:__all__"
            )
            if src is not None:
                sources[f"{driver}:{k}"] = src
            if cf is not None:
                confidence[f"{driver}:{k}"] = cf

    # Composite provenance: the CONCORDANCE identity (source/version/licence) + a stamp of the
    # archetype-trajectory identity and a content hash of both, so the mapped object records that a
    # concordance was applied (review P2 2026-08-29: the old object kept only the v1 identity).
    cp = conc["provenance"]
    composite = Provenance(
        source=(f"{cp['source']} | archetype paths: {archetype.provenance.source}"),
        source_version=(f"{cp['source_version']}+{archetype.provenance.source_version}"),
        licence=cp["licence"],
        reference_year=cp.get("reference_year", archetype.provenance.reference_year),
        retrieved=cp.get("retrieved", archetype.provenance.retrieved),
    )
    # The mapped trajectory's numeric rate tables ARE the GDP-blended per-region paths, so the
    # recursive manifest's existing rate-table hash (in _trend_provenance) already moves when the
    # concordance weights or the archetype rates change — the mapping is captured in the numbers
    # themselves, and the composite provenance records WHICH concordance produced them.
    return StructuralTrajectory(
        provenance=composite,
        rates=new_rates,
        sector_rates=new_sector_rates,
        sources=sources,
        confidence=confidence,
    )
