"""Load the **real** ENCORE knowledge base vendored under ``data/encore/`` (Phase 6, 2026-08-09).

This is the counterpart to ``fixture.py`` (which builds a small *synthetic* object for offline
tests). Here we ingest the actual ENCORE May-2026 ratings via ``load_encore_ratings_wide`` — real
ISIC-coded production processes, real ecosystem-service vocabulary, real ND/N-A cells. The data is
CC BY-SA 4.0; see ``data/encore/NOTICE.md`` for attribution.

**Scope.** This exposes the real *dependency* and *pressure* objects AND the real EXIOBASE↔ENCORE
concordance (``real_encore_concordance``, built in ``concordance_build``), so a real nature scenario
runs end-to-end against a real-EXIOBASE-labelled economy. Note that the synthetic *toy* concordance
does NOT map real ISIC codes — use ``real_encore_concordance`` (or your own) with the real objects.
The concordance is **equal-weighted (a documented v1 assumption)** and ENCORE ratings are indicators
of *potential* significance, not calibrated elasticities — so results stay illustrative of the
method, not calibrated risk.
"""

from __future__ import annotations

from pathlib import Path

from cge.contracts.data_objects import Provenance
from cge.nature.encore import EncoreDependencies, load_encore_ratings_wide

# Repo-root-relative default DATASET ROOT — the directory that contains BOTH ``ENCORE files/`` and
# ``Crosswalk tables/`` (data/encore/…). Resolved from this file so it works regardless of the
# process's working directory. Every public ``root`` argument uses this SAME convention (review P2
# round 5 2026-08-13: the dependency and concordance loaders previously disagreed on what ``root``
# meant).
_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "encore"
# The vendored EXIOBASE MRSUT-derived product→industry supply shares (review P1-methodology round 7
# 2026-08-14). CC BY-SA 4.0; see data/exiobase/NOTICE.md. Rebuilt by scripts/build_supply_shares.py.
# Artifacts are YEAR-BOUND: supply_shares_{year}.json. Only 2019 is vendored today; a build for a
# year without its own artifact falls back to this default year, but the mismatch is recorded in
# provenance rather than silently using the wrong year (review P2 round 8 2026-08-14).
_EXIOBASE_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "exiobase"
_DEFAULT_SUPPLY_SHARE_YEAR = 2019
# The 200-product count the artifact must match (industry membership is checked against the live ixi
# vocabulary, not a hard-coded count).
_N_EXIOBASE_PRODUCTS = 200


def _supply_shares_path(year: int | None = None) -> Path:
    yr = _DEFAULT_SUPPLY_SHARE_YEAR if year is None else year
    return _EXIOBASE_DATA_DIR / f"supply_shares_{yr}.json"


_ENCORE_SUBDIR = "ENCORE files"
_CROSSWALK_SUBDIR = "Crosswalk tables"
_DEPENDENCY_CSV = "06. Dependency mat ratings.csv"
_PRESSURE_CSV = "07. Pressure mat ratings.csv"
_CROSSWALK_CSV = "EXIOBASE - NACE Rev. 2 - ISIC Rev. 4 - ISIC Rev. 5.csv"


def _encore_dir(root: str | Path | None) -> Path:
    """The ``ENCORE files/`` directory under the dataset root."""
    return (Path(root) if root else _DEFAULT_ROOT) / _ENCORE_SUBDIR


def _crosswalk_path(root: str | Path | None) -> Path:
    """The EXIOBASE↔ISIC crosswalk CSV under the dataset root."""
    return (Path(root) if root else _DEFAULT_ROOT) / _CROSSWALK_SUBDIR / _CROSSWALK_CSV


# Attribution baked into every real-ENCORE run's provenance (CC BY-SA 4.0 requirement).
_ENCORE_SOURCE = "ENCORE knowledge base (ENCORE Partners: UNEP-WCMC, UNEP FI, Global Canopy)"
_ENCORE_VERSION = "Updated ENCORE knowledge base May 2026"
_ENCORE_LICENCE = "CC BY-SA 4.0"


def encore_data_available(root: str | Path | None = None) -> bool:
    """True if the vendored ENCORE dependency file is present (it may be absent in a checkout that
    excluded the data). Lets tests/GUI degrade gracefully rather than raise on import."""
    return (_encore_dir(root) / _DEPENDENCY_CSV).exists()


def _provenance() -> Provenance:
    return Provenance(
        source=_ENCORE_SOURCE,
        source_version=_ENCORE_VERSION,
        licence=_ENCORE_LICENCE,
        reference_year=2026,
        retrieved="2026-08-09",
        notes=(
            "Real ENCORE ratings vendored under data/encore/ (CC BY-SA 4.0; see NOTICE.md). "
            "Materiality ratings are indicators of POTENTIAL significance, not calibrated "
            "output/TFP elasticities — the severity→productivity mapping remains a scenario "
            "assumption."
        ),
    )


def real_encore_dependencies(root: str | Path | None = None) -> EncoreDependencies:
    """The real ENCORE **dependency** ratings (production process × ecosystem service) as an
    ``EncoreDependencies`` object, with attribution provenance. ND cells are kept distinct.

    ``root`` is the DATASET ROOT (the directory containing ``ENCORE files/`` and ``Crosswalk
    tables/``), the same convention as every other loader here."""
    path = _encore_dir(root) / _DEPENDENCY_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"ENCORE dependency file not found at {path}. It is vendored under data/encore/ "
            "(CC BY-SA 4.0); ensure the data directory is present in this checkout."
        )
    return load_encore_ratings_wide(str(path), provenance=_provenance(), kind="dependency")


def real_encore_pressures(root: str | Path | None = None) -> EncoreDependencies:
    """The real ENCORE **pressure / impact-driver** ratings, typed ``kind='impact'``. Ingested and
    provenance-stamped, but NOT yet consumed by an engine (dependencies drive the productivity
    channel; a pressure/impact channel is a documented follow-up). ``root`` is the dataset root."""
    path = _encore_dir(root) / _PRESSURE_CSV
    if not path.exists():
        raise FileNotFoundError(f"ENCORE pressure file not found at {path}.")
    return load_encore_ratings_wide(str(path), provenance=_provenance(), kind="impact")


def real_encore_concordance(root: str | Path | None = None):
    """The real **EXIOBASE → ENCORE** concordance, derived from the vendored crosswalk against the
    real dependency processes. Returns ``(ConcordanceMap, ConcordanceAudit)``. Equal-weighted v1
    (a documented assumption — see ``concordance_build`` and the audit). This is what lets a real
    nature scenario run against a real EXIOBASE-shaped economy. ``root`` is the dataset root (same
    convention as ``real_encore_dependencies``)."""
    from cge.nature.concordance_build import build_exiobase_encore_concordance

    dep = real_encore_dependencies(root)
    return build_exiobase_encore_concordance(dep, crosswalk_path=_crosswalk_path(root))


def real_encore_concordance_industries(root: str | Path | None = None):
    """The real EXIOBASE **industry** (ixi) → ENCORE concordance, COMPLETED over the full
    163-industry classification via the shared NACE-sibling fallback (review P2 round 7 2026-08-14).

    The vendored crosswalk covers 162 industries; the one it omits (``Production of electricity
    nec``) is filled from its covered NACE siblings, so a direct ``system="ixi"`` build attaches
    nature over the whole classification — the same completed object the product bridge consumes.
    Returns ``(ConcordanceMap, filled)`` where ``filled`` records the fallback for the audit.
    ``root`` is the dataset root."""
    from cge.nature.concordance_build import complete_industry_concordance

    industry_conc, _audit = real_encore_concordance(root)
    return complete_industry_concordance(industry_conc)


def supply_shares_available(year: int | None = None, path: str | Path | None = None) -> bool:
    """True if a vendored EXIOBASE supply-share artifact is present for ``year`` (or the default
    year when ``year`` is None / has no artifact of its own)."""
    if path is not None:
        return Path(path).exists()
    if _supply_shares_path(year).exists():
        return True
    return _supply_shares_path(_DEFAULT_SUPPLY_SHARE_YEAR).exists()


class SupplyShareValidationError(ValueError):
    """A vendored supply-share artifact is malformed / incomplete — better to fail than to silently
    degrade a build back to the code-prefix methodology (review P2 round 8 2026-08-14)."""


def _valid_industry_vocab() -> set[str]:
    import pymrio

    return {str(x).strip() for x in pymrio.get_classification("exio3_ixi").sectors["ExioName"]}


def _validate_supply_share_artifact(art: dict, source: str) -> None:
    """Reject a malformed artifact instead of letting it silently degrade the bridge (P2 round 8).

    Checks: (1) shares and zero_supply_products partition EXACTLY the 200 EXIOBASE products,
    disjointly; (2) every producing industry is in the 163-industry vocabulary; (3) each product's
    weights are finite, non-negative, and sum to 1. A product merely MISSING from both sets is the
    dangerous case (it would be treated as zero-supply and reported as 'no market supply' though it
    was really dropped), so a wrong product count fails here."""
    prov = art.get("provenance", {})
    shares = art.get("shares")
    if not isinstance(shares, dict):
        raise SupplyShareValidationError(f"{source}: 'shares' missing or not an object")
    zero = prov.get("zero_supply_products", [])
    if not isinstance(zero, list):
        raise SupplyShareValidationError(f"{source}: provenance.zero_supply_products not a list")
    share_keys, zero_keys = set(shares), set(zero)
    overlap = share_keys & zero_keys
    if overlap:
        raise SupplyShareValidationError(
            f"{source}: {len(overlap)} product(s) are in BOTH shares and zero_supply_products "
            f"(e.g. {sorted(overlap)[:3]}); the two sets must be disjoint."
        )
    total = len(share_keys) + len(zero_keys)
    if total != _N_EXIOBASE_PRODUCTS:
        raise SupplyShareValidationError(
            f"{source}: shares ∪ zero_supply_products cover {total} products, expected exactly "
            f"{_N_EXIOBASE_PRODUCTS}. A product missing from both would be silently mistaken for "
            "'no market supply'. Regenerate the artifact from the MRSUT."
        )
    vocab = _valid_industry_vocab()
    for product, wmap in shares.items():
        if not isinstance(wmap, dict) or not wmap:
            raise SupplyShareValidationError(f"{source}: product {product!r} has no weights")
        bad_ind = [i for i in wmap if i not in vocab]
        if bad_ind:
            raise SupplyShareValidationError(
                f"{source}: product {product!r} maps to non-EXIOBASE industry(ies) {bad_ind[:3]}"
            )
        vals = list(wmap.values())
        if any((not isinstance(v, int | float)) or v != v or v < 0 for v in vals):  # NaN/neg
            raise SupplyShareValidationError(
                f"{source}: product {product!r} has a non-finite/negative weight"
            )
        if abs(sum(vals) - 1.0) > 1e-6:
            raise SupplyShareValidationError(
                f"{source}: product {product!r} weights sum to {sum(vals):.6f}, not 1"
            )


def load_supply_shares(
    year: int | None = None, path: str | Path | None = None
) -> tuple[dict[str, dict[str, float]], dict]:
    """Load and VALIDATE the vendored EXIOBASE product→industry **supply shares** for ``year``.

    Returns ``(shares, provenance)`` where ``shares[product][industry]`` is the observed fraction of
    the product's total monetary supply produced by that industry (summed over regions, sums to 1).
    Products with no market supply (recycling/treatment residuals) are ABSENT from ``shares`` and
    listed in ``provenance.zero_supply_products`` — the caller falls back to the code-prefix method
    for exactly those. The provenance carries the EXIOBASE DOI, SUT year+version, threshold, and —
    when the requested ``year`` has no artifact — a ``year_fallback`` note recording that the
    default year's shares were substituted (review P2 round 8 2026-08-14; the mismatch is
    provenance-visible, never silent). The artifact is validated on load; a malformed one raises
    ``SupplyShareValidationError`` rather than silently degrading the bridge. CC BY-SA 4.0."""
    import json

    fell_back_year: int | None = None
    if path is not None:
        p = Path(path)
    else:
        p = _supply_shares_path(year)
        if not p.exists() and year is not None:
            fallback = _supply_shares_path(_DEFAULT_SUPPLY_SHARE_YEAR)
            if fallback.exists():
                fell_back_year = year
                p = fallback
    if not p.exists():
        raise FileNotFoundError(
            f"EXIOBASE supply-share artifact not found at {p}. Regenerate it with "
            "`python scripts/build_supply_shares.py` from the MRSUT download (see NOTICE.md)."
        )
    with open(p) as fh:
        art = json.load(fh)
    _validate_supply_share_artifact(art, source=str(p))
    prov = dict(art.get("provenance", {}))
    if fell_back_year is not None:
        prov["year_fallback"] = (
            f"build year {fell_back_year} has no supply-share artifact; used the "
            f"{_DEFAULT_SUPPLY_SHARE_YEAR} shares instead (producing-industry mix assumed stable)."
        )
    return art["shares"], prov


def real_encore_concordance_products(
    root: str | Path | None = None,
    *,
    year: int | None = None,
    with_audit: bool = False,
):
    """The real EXIOBASE **product** (pxp) → ENCORE concordance.

    The crosswalk is keyed by industry-style labels, but the default live build is ``system="pxp"``
    (product labels). This bridges each product to its producing ixi industry(ies) and averages the
    industry concordance onto the products. The producing-industry weights come from the **observed
    EXIOBASE MRSUT supply shares** when the vendored artifact is present (review P1-method round 7
    2026-08-14) — so, e.g., the biofuels no longer receive byte-identical prefix-inferred weights;
    products with no market supply (and any product not in the artifact) fall back to the
    classification-prefix method. When the artifact is absent, the whole bridge uses the prefix
    method (the round-6 behaviour), so nature still runs without the multi-GB download.

    Returns ``(ConcordanceMap, uncovered_products)`` (or, with ``with_audit=True``,
    ``(ConcordanceMap, uncovered_products, ProductBridgeAudit)``) — a product whose producing
    industries are all uncovered is reported, not silently dropped, so the build's complete-coverage
    gate can act on it. ``root`` is the dataset root."""
    from cge.nature.concordance_build import bridge_to_products, pxp_to_ixi_industries

    industry_conc, _audit = real_encore_concordance(root)
    shares, version = (None, "")
    if supply_shares_available(year):
        shares, prov = load_supply_shares(year)
        version = prov.get("source_version", "")
        if prov.get("year_fallback"):  # make the year mismatch visible on the bridge provenance too
            version = f"{version} [{prov['year_fallback']}]"
    cmap, uncovered, bridge_audit = bridge_to_products(
        industry_conc,
        pxp_to_ixi_industries(),
        supply_shares=shares,
        supply_shares_version=version,
    )
    if with_audit:
        return cmap, uncovered, bridge_audit
    return cmap, uncovered
