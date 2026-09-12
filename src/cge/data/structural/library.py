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
        source_labour_shares={k: float(v) for k, v in raw.get("source_labour_shares", {}).items()},
    )


def default_structural_trajectories() -> StructuralTrajectory:
    """Alias for the vendored default trajectories — the set a recursive run uses when the scenario
    asks for sourced structural trends without naming a specific artifact."""
    return load_structural_trajectories()


_DEFAULT_CONCORDANCE = (
    Path(__file__).resolve().parents[4] / "data" / "structural" / "concordance_v3.json"
)


def load_structural_concordance(
    path: str | Path | None = None,
    *,
    trajectory: StructuralTrajectory | None = None,
) -> dict:
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
    loudly rather than silently substituting ``__all__``.

    ``trajectory`` (review P2 2026-08-31): validate the archetype targets against the trajectory the
    concordance will actually be applied to — NOT always the default artifact. Passing a custom
    trajectory here catches a concordance that names an archetype the custom trajectory lacks,
    rather than that member being silently dropped from the blend downstream."""
    artifact = Path(path) if path is not None else _DEFAULT_CONCORDANCE
    if not artifact.exists():
        raise FileNotFoundError(
            f"structural concordance not found at {artifact}; expected the vendored "
            "data/structural/concordance_v3.json (see data/structural/NOTICE.md)."
        )
    raw = json.loads(artifact.read_text())
    if "provenance" not in raw or not raw["provenance"]:
        raise ValueError("structural concordance is missing 'provenance'")
    # Parse provenance through the contract so an unsourced/malformed block fails on load.
    Provenance(**raw["provenance"])
    if not raw.get("sector_archetype"):
        raise ValueError("structural concordance is missing or empty 'sector_archetype'")

    # Validate against the trajectory this concordance will be applied to (default artifact if none
    # given), so a custom trajectory's missing archetype is caught here, not silently dropped later.
    #
    # PER-DRIVER validation (review P2 2026-09-05). A concordance archetype must be resolvable by
    # EVERY driver actually present on its axis — an explicit path for that archetype in that
    # driver, OR that driver's own ``__all__`` fallback. The earlier version validated against the
    # UNION of archetypes across the axis's drivers, which let a leak through: if ``population``
    # carried only archetype N and ``productivity`` only S while the concordance mapped a block to
    # both, {N,S} existed in the union so validation passed — but at blend time each driver silently
    # DROPS the members whose archetype it lacks and renormalises, so the SAME mixed block became N
    # 100% for population and 100% S for productivity. Requiring each present driver to resolve
    # every referenced archetype (via its explicit paths or its own ``__all__``) forbids that silent
    # divergence. A driver with an ``__all__`` fallback accepts any archetype (it degrades to the
    # global path for the missing ones), which is an explicit, non-silent fallback. We only
    # constrain an axis that the trajectory actually carries at least one driver on.
    traj = trajectory if trajectory is not None else load_structural_trajectories()

    def _check_axis(axis_label: str, drivers: dict, referenced: set) -> None:
        """Validate the concordance archetypes referenced on one axis against every present driver.

        TWO conditions (review P2 2026-09-05, refined 2026-09-06):
          1. Each referenced archetype must be REAL — an explicit path in AT LEAST ONE driver on the
             axis. This catches a typo/unknown archetype (e.g. US→'TYPO'). It is SKIPPED when EVERY
             driver on the axis supplies only ``__all__`` (a deliberately UNIFORM trajectory): then
             there are no explicit archetypes to be "unknown" relative to, and every referenced
             archetype resolves to the global path — rejecting it would forbid the legitimate
             ``__all__``-only fallback the contract defines (review P2 2026-09-06).
          2. EVERY present driver must RESOLVE each referenced archetype — an explicit path for it
             in that driver, OR that driver's own ``__all__`` fallback. This forbids the
             cross-driver leak where population (only N) and productivity (only S) would each
             silently drop the other's members and renormalise, so a mixed block becomes 100% N for
             100% S for productivity.
        """
        real = set().union(*(set(t) for t in drivers.values())) - {"__all__"}
        all_uniform = all("__all__" in t for t in drivers.values()) and not real
        unknown = referenced - real
        if unknown and not all_uniform:
            raise ValueError(
                f"{axis_label} archetype(s) {sorted(unknown)} are not among the trajectory's "
                f"{axis_label} paths {sorted(real)} (not defined by any driver) — a typo or an "
                "archetype the trajectory does not carry."
            )
        for driver, table in drivers.items():
            if "__all__" in table:
                continue  # explicit global fallback resolves every archetype (non-silent)
            missing = {a for a in referenced if a not in table}
            if missing:
                raise ValueError(
                    f"{axis_label} driver {driver!r} cannot resolve concordance archetype(s) "
                    f"{sorted(missing)} (no explicit path and no '__all__' fallback); it has "
                    f"{sorted(table)}. Every present driver must resolve every mapped archetype, "
                    "else the blend would silently drop members and renormalise (review P2)."
                )

    is_v2 = "country_archetype" in raw and "block_membership" in raw
    if traj.rates:
        if is_v2:
            referenced = {a for a in raw["country_archetype"].values() if a != "__all__"}
        elif raw.get("region_archetype"):
            referenced = {a for a in raw["region_archetype"].values() if a != "__all__"}
        else:
            referenced = set()
        _check_axis("region", traj.rates, referenced)

    if traj.sector_rates:
        referenced_sec = {a for a in raw["sector_archetype"].values() if a != "__all__"}
        _check_axis("sector", traj.sector_rates, referenced_sec)

    # Structural v2 checks (block weights sum to 1, members mapped) run regardless of the
    # trajectory.
    if is_v2:
        _validate_v2_structure(raw)
        if "block_weights" in raw:  # v3 (review-9 1c): validate the per-driver/year weight tables
            _validate_v3_block_weights(raw)
    elif not raw.get("region_archetype"):
        raise ValueError(
            "structural concordance needs either 'country_archetype'+'block_membership' (v2) or "
            "'region_archetype' (v1)."
        )
    return raw


def _validate_v3_block_weights(raw: dict) -> None:
    """Validate v3 ``block_weights`` (review-9 1c): every weight table (flat or year-indexed) covers
    only mapped countries and sums to 1; ``driver_weight_class`` names real classes. A malformed v3
    fails loudly rather than silently mis-blending."""
    import math

    country_arch = raw["country_archetype"]
    classes = raw["block_weights"]
    for cls, blocks in classes.items():
        for block, table in blocks.items():
            # A table is either flat {country: w} or year-indexed {year: {country: w}}.
            year_indexed = table and all(str(k).isdigit() for k in table)
            per_year = table.values() if year_indexed else [table]
            for w_map in per_year:
                missing = [c for c in w_map if c not in country_arch]
                if missing:
                    raise ValueError(
                        f"block_weights[{cls!r}][{block!r}] has unmapped countries {missing}"
                    )
                w = list(w_map.values())
                if any(not math.isfinite(x) or x < 0 for x in w):
                    raise ValueError(
                        f"block_weights[{cls!r}][{block!r}] weights must be finite and ≥ 0"
                    )
                if abs(sum(w) - 1.0) > 1e-3:
                    raise ValueError(
                        f"block_weights[{cls!r}][{block!r}] weights sum to {sum(w):.4f}, want 1.0"
                    )
    for driver, cls in raw.get("driver_weight_class", {}).items():
        if cls != "block_membership" and cls not in classes:
            raise ValueError(
                f"driver_weight_class[{driver!r}]={cls!r} not a block_weights class {list(classes)}"
            )


def _validate_v2_structure(raw: dict) -> None:
    """Validate the v2 country-level concordance's STRUCTURE (review P2 2026-08-29), independent of
    any trajectory: every ``block_membership`` country has a ``country_archetype``, and each block's
    weights are finite, non-negative and sum to 1. Archetype-vs-trajectory membership is now checked
    PER DRIVER by the caller (review P2 2026-09-05), so it is not repeated here."""
    import math

    country_arch: dict = raw["country_archetype"]
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


def _members_at(conc: dict, block: str, driver: str, year: int) -> dict | None:
    """The ``{country: weight}`` map for ``block`` under ``driver`` at ``year`` (review-9 1c). v3
    carries ``block_weights[class][block]`` where ``class`` is chosen by ``driver_weight_class``
    (population→population, productivity→output, participation→the static v2 GDP weights); the
    population class is year-indexed (``{year: {country: w}}``), output is flat ``{country: w}``. A
    v2 concordance (no ``block_weights``) falls back to the single static ``block_membership``."""
    bw = conc.get("block_weights")
    if bw:
        cls = conc.get("driver_weight_class", {}).get(driver, "block_membership")
        table = bw.get(cls, {}).get(block) if cls != "block_membership" else None
        if table is not None:
            if table and all(str(k).isdigit() for k in table):  # year-indexed
                yrs = sorted(int(k) for k in table)
                chosen = max([y for y in yrs if y <= year], default=yrs[0])
                return table[str(chosen)]
            return table  # flat (e.g. output)
    return conc.get("block_membership", {}).get(block)


def _blend_region_path(
    by_arch: dict, block: str, conc: dict, region_arch_v1: dict | None, driver: str = ""
) -> tuple[dict, dict] | None:
    """The rate path for a coarse ``block`` on one region ``driver``: the WEIGHTED blend of its
    member countries' archetype paths (v2/v3), or the block's single archetype path (v1). Returns
    ``(path, arch_weights)`` — the ``{year: rate}`` dict and the ``{archetype: weight}`` map of
    which archetypes actually contributed (for provenance) — or None if no archetype path is
    available.

    Blend (review P1 2026-08-29; PER-DRIVER + TIME-VARYING weights review-9 1c 2026-09-16): for each
    knot year the blended rate is Σ_c weight[c, driver, year] · archetype_rate(archetype[c], year),
    so a mixed block gets a weighted average with the driver-appropriate, year-appropriate weights
    (population growth by population shares that drift over the horizon, productivity by output
    shares). The union of member knot years is used so no knot is dropped.

    **Renormalisation (review P2 2026-08-31):** members whose archetype has NO path in *this*
    trajectory are dropped from BOTH the numerator and the weight denominator, so the blend stays a
    proper convex combination. If NO member has a usable path, returns None (caller falls back to
    ``__all__``)."""
    if region_arch_v1 is not None:  # v1: single archetype
        arch = region_arch_v1.get(block)
        path = by_arch.get(arch) if arch else None
        return (dict(path), {arch: 1.0}) if path is not None else None
    country_arch: dict = conc["country_archetype"]

    def _rate_at(path: dict, year: int) -> float:
        # Piecewise-constant hold (same rule as StructuralTrajectory._lookup).
        applicable = [y for y in sorted(path) if y <= year]
        chosen = applicable[-1] if applicable else min(path)
        return float(path[chosen])

    # Knot years = union of the members' archetype paths, using the FIRST-knot weights just to find
    # which archetypes have a usable path (membership doesn't change across the weight classes).
    seed = _members_at(conc, block, driver, year=-(10**9))  # earliest-knot weights
    if not seed:
        return None
    knot_years: set[int] = set()
    for c in seed:
        p = by_arch.get(country_arch[c])
        if p:
            knot_years |= set(p)
    if not knot_years:
        return None

    blended: dict = {}
    arch_weights_repr: dict = {}  # archetype weights at the earliest knot, for provenance
    for i, year in enumerate(sorted(knot_years)):
        members = _members_at(conc, block, driver, year) or seed
        # Surviving members (archetype path present), renormalised to a proper convex combination.
        surviving = {c: float(w) for c, w in members.items() if by_arch.get(country_arch[c])}
        total_w = sum(surviving.values())
        if total_w <= 0:
            continue
        aw: dict = {}
        for c, w in surviving.items():
            arch = country_arch[c]
            aw[arch] = aw.get(arch, 0.0) + w / total_w
        blended[int(year)] = sum(w * _rate_at(by_arch[arch], year) for arch, w in aw.items())
        if i == 0:
            arch_weights_repr = aw
    return (blended, arch_weights_repr) if blended else None


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

    The emitted trajectory carries **composite provenance** (review P2 2026-08-29/2026-08-31): the
    concordance source/version/licence + the archetype-trajectory identity + a CONTENT HASH of BOTH
    artifacts (so a concordance edit that leaves the blended numbers unchanged — e.g. reallocating
    weight among same-archetype countries — is still detectable). Each mapped region key records the
    ACTUAL contributing archetype sources and their blend weights (a mixed block shows both N and S,
    not a spurious world-average citation).

    **Weighting (review P1 2026-08-31; PER-DRIVER + TIME-VARYING review-9 1c 2026-09-16).** With the
    default v3 concordance, each region driver blends with its OWN weight class: population by WPP
    per-country population shares that DRIFT across the 2025/2035/2050 knots, productivity by PWT
    output shares; participation still uses the static v2 GDP weights (no per-country labour-force
    series available). The residual EXIOBASE W* aggregates (esp. ``RoW_MiddleEast`` = 100% S) remain
    coarse single-archetype proxies. The basis is stamped into the composite provenance notes and
    detailed in ``data/structural/NOTICE.md``. A v2 concordance (no ``block_weights``) still loads,
    falling back to the single static GDP weights for every driver.

    ``require_full_coverage`` (default True): a build label not mapped by the concordance raises
    :class:`UnmappedStructuralLabels` — so a real run can never silently degrade to an
    all-``__all__`` trajectory."""
    archetype = load_structural_trajectories(trajectory_path)
    # Validate the concordance against the trajectory it will actually be applied to (review P2
    # 2026-08-31): a custom trajectory missing an archetype the concordance names now fails loudly
    # here rather than that member being silently dropped from the blend.
    conc = load_structural_concordance(concordance_path, trajectory=archetype)
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

    # Region axis: blend per (driver, region) from the member countries' archetype paths. Record the
    # per-(driver, region) contributing archetypes + weights so the provenance can name the actual
    # sources rather than always the __all__ citation (review P2 2026-08-31).
    new_rates: dict = {}
    region_arch_weights: dict[tuple[str, str], dict] = {}  # (driver, region) -> {archetype: weight}
    for driver, by_arch in archetype.rates.items():
        new: dict = {}
        for r in regions:
            blend = _blend_region_path(by_arch, r, conc, region_arch_v1, driver=driver)
            if blend is not None:
                path, aw = blend
                new[r] = path
                region_arch_weights[(driver, r)] = aw
            elif "__all__" in by_arch:
                new[r] = dict(by_arch["__all__"])
                region_arch_weights[(driver, r)] = {"__all__": 1.0}
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

    # Provenance/confidence per emitted key (review P2 2026-08-31): a blended region records the
    # ACTUAL contributing archetype sources and their weights (not always the __all__ citation); a
    # pure block inherits its single archetype's source; a sector inherits its archetype's.
    conc_ver = conc["provenance"]["source_version"]
    sources: dict = {}
    confidence: dict = {}

    def _arch_src(driver: str, arch: str) -> str | None:
        return archetype.sources.get(f"{driver}:{arch}") or archetype.sources.get(
            f"{driver}:__all__"
        )

    def _arch_cf(driver: str, arch: str) -> str | None:
        return archetype.confidence.get(f"{driver}:{arch}") or archetype.confidence.get(
            f"{driver}:__all__"
        )

    for driver, by_key in new_rates.items():
        for k in by_key:
            if k == "__all__":
                src = _arch_src(driver, "__all__")
                if src is not None:
                    sources[f"{driver}:{k}"] = src
                cf = _arch_cf(driver, "__all__")
                if cf is not None:
                    confidence[f"{driver}:{k}"] = cf
                continue
            aw = region_arch_weights.get((driver, k), {})
            # Name each contributing archetype's source with its blend weight, so a mixed block has
            # both N and S sources (not a spurious world-average citation).
            parts = []
            for arch in sorted(aw):
                s = _arch_src(driver, arch)
                parts.append(f"{arch} ({aw[arch]:.2f}): {s}" if s else f"{arch} ({aw[arch]:.2f})")
            if parts:
                sources[f"{driver}:{k}"] = f"concordance-blended [{conc_ver}] — " + "; ".join(parts)
            # Confidence: the lowest (most conservative) among the contributing archetypes.
            cfs = [c for arch in aw if (c := _arch_cf(driver, arch)) is not None]
            order = {"low": 0, "medium": 1, "high": 2}
            if cfs:
                confidence[f"{driver}:{k}"] = min(cfs, key=lambda c: order.get(c, 1))
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

    # Composite provenance: the CONCORDANCE identity (source/version/licence) + the archetype-
    # trajectory identity + a CONTENT HASH of BOTH artifacts (review P2 2026-08-31). The concordance
    # hash matters because the emitted numeric rate tables can be UNCHANGED by a concordance edit
    # that reallocates weights among countries of the SAME archetype (the blend is unchanged), so a
    # rate-table hash alone cannot detect it — the concordance hash can.
    from cge.contracts.provenance import content_hash

    cp = conc["provenance"]
    conc_hash = content_hash(conc)
    # Include source_labour_shares in the archetype content hash (review P1 2026-09-06): they set
    # s_L^src in the source-side MFP identification and materially change the mapped trajectory's
    # emitted shocks, so an edit to them must move the mapped provenance hash.
    arch_hash = content_hash(
        {
            "rates": archetype.rates,
            "sector_rates": archetype.sector_rates,
            "source_labour_shares": archetype.source_labour_shares,
        }
    )
    composite = Provenance(
        source=(f"{cp['source']} | archetype paths: {archetype.provenance.source}"),
        source_version=(f"{cp['source_version']}+{archetype.provenance.source_version}"),
        licence=cp["licence"],
        reference_year=cp.get("reference_year", archetype.provenance.reference_year),
        retrieved=cp.get("retrieved", archetype.provenance.retrieved),
        notes=(
            f"concordance-mapped structural trajectory; concordance_content_hash={conc_hash}; "
            f"archetype_content_hash={arch_hash}. "
            + (
                (
                    "WEIGHTING (review-9 1c 2026-09-16): PER-DRIVER, TIME-VARYING block weights — "
                    "population blended by UN WPP per-country population shares that DRIFT across "
                    "the 2025/2035/2050 knots, productivity by PWT output shares; participation "
                    "uses the static v2 GDP weights (no per-country labour-force series). Residual "
                    "EXIOBASE W* aggregates (esp. RoW_MiddleEast=100% S) remain coarse "
                    "single-archetype proxies. See data/structural/NOTICE.md."
                )
                if "block_weights" in conc
                else (
                    "WEIGHTING CAVEAT (review P1 2026-08-31): the same static GDP-share block "
                    "weights are applied to ALL region drivers (population, participation, "
                    "productivity) — not per-driver weights — and are held fixed over the horizon; "
                    "residual W* blocks (esp. RoW_MiddleEast=100% S) are coarse single-archetype "
                    "aggregates. Documented simplification; see data/structural/NOTICE.md."
                )
            )
        ),
    )
    # Remap the archetype SOURCE labour shares onto the real build sectors the same way the sector
    # rates were remapped (pipeline step 1b): each real sector inherits its archetype's s_L^src (or
    # the archetype trajectory's __all__), so the mapped trajectory can still run source-share MFP
    # identification. Carries the archetype __all__ too, as the fallback for unmatched sectors.
    src_sl = archetype.source_labour_shares or {}
    mapped_src_sl: dict[str, float] = {}
    if src_sl:
        real_sectors = {k for by_key in new_sector_rates.values() for k in by_key}

        def _stamp_share(real_key: str, arch: str) -> None:
            # Inherit the archetype's per-share provenance/confidence (the contract requires a
            # source_labour_share:{key} entry for every share, review P2 2026-09-06).
            a_src = archetype.sources.get(f"source_labour_share:{arch}") or archetype.sources.get(
                "source_labour_share:__all__"
            )
            a_cf = archetype.confidence.get(
                f"source_labour_share:{arch}"
            ) or archetype.confidence.get("source_labour_share:__all__")
            if a_src is not None:
                sources[f"source_labour_share:{real_key}"] = a_src
            if a_cf is not None:
                confidence[f"source_labour_share:{real_key}"] = a_cf

        for k in real_sectors:
            arch = sector_arch.get(k, "__all__")
            val = src_sl.get(arch, src_sl.get("__all__"))
            if val is not None:
                mapped_src_sl[k] = float(val)
                _stamp_share(k, arch)
        if "__all__" in src_sl:
            mapped_src_sl["__all__"] = float(src_sl["__all__"])
            _stamp_share("__all__", "__all__")

    return StructuralTrajectory(
        provenance=composite,
        rates=new_rates,
        sector_rates=new_sector_rates,
        sources=sources,
        confidence=confidence,
        source_labour_shares=mapped_src_sl,
    )


def structural_trajectory_variants(
    regions: list[str] | None = None,
    sectors: list[str] | None = None,
    *,
    concordance_path: str | Path | None = None,
    require_full_coverage: bool = True,
) -> dict[str, StructuralTrajectory]:
    """The low / central / high structural trajectories, for an uncertainty SWEEP (review-9 1e).

    Returns ``{"low": ..., "central": ..., "high": ...}``. The central path is
    ``trajectories_v1.json``; the siblings ``trajectories_v1_low.json`` / ``_high.json`` are the
    sensitivity band (EU KLEMS sector drivers = empirical trend-window envelope; other drivers =
    confidence-tiered ±band), produced by ``scripts/build_uncertainty_sets.py`` — "low" is uniformly
    the weaker-growth / slower-decarbonisation world, "high" the stronger.

    If ``regions``/``sectors`` are given, each variant is mapped onto the build's real labels via
    :func:`structural_trajectories_for_build` (same concordance for all three); otherwise the raw
    N/S + BRD/MIL archetype variants are returned. Typical use::

        variants = structural_trajectory_variants(regions, sectors)
        results = {
            label: run_recursive(sc, config=DynamicConfig(structural=traj), data_source=build)
            for label, traj in variants.items()
        }
        # report the interval across results["low"] .. results["high"], not a single point.
    """
    siblings = {
        "low": _DEFAULT_ARTIFACT.with_name("trajectories_v1_low.json"),
        "central": _DEFAULT_ARTIFACT,
        "high": _DEFAULT_ARTIFACT.with_name("trajectories_v1_high.json"),
    }
    out: dict[str, StructuralTrajectory] = {}
    for label, path in siblings.items():
        if regions is not None and sectors is not None:
            out[label] = structural_trajectories_for_build(
                regions,
                sectors,
                trajectory_path=path,
                concordance_path=concordance_path,
                require_full_coverage=require_full_coverage,
            )
        else:
            out[label] = load_structural_trajectories(path)
    return out
