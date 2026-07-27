"""Raw SAM construction from an EXIOBASE build (roadmap Phase 5.1a).

Maps an ``IOSystem`` (multi-regional, from an aggregated EXIOBASE build) into the accounts of a
single-region **closed-economy** SAM — the calibration target for the CGE pilot. The pilot model
is closed (no Armington trade yet), so this collapses the MRIO's regions into one economy by
summing flows; inter-regional trade is folded into the domestic block until the open-economy
sub-phase adds a rest-of-world account (documented, not hidden).

Steps:
1. Gross output ``x = (I − A)⁻¹ · fd`` (Leontief), then intermediate flows ``Z = A · diag(x)``.
2. Aggregate Z, final demand and value added over regions → sector×sector, sector-vectors.
3. Value added per sector ``VA_i = x_i − Σ_j Z[j,i]`` (output minus intermediate purchases),
   split into capital/labour by a documented share (EXIOBASE's factor split is thin, so the
   split is an explicit assumption recorded in the SAM quality report).
4. Assemble the SAM: activities/commodities collapsed per sector, factors CAP/LAB, one household.

The result is passed to ``balance.py`` (RAS) and ``quality.py`` before the CGE calibrates on it.
Every fabricated cell / assumed share is recorded so a reviewer can see how much was "helped".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cge.contracts.data_objects import SAM, IOSystem, Provenance
from cge.contracts.quality import QualityReport
from cge.data.sam.balance import is_balanced, ras_balance
from cge.data.sam.quality import sam_quality_report
# Canonical trade-materiality threshold — shared with the calibrator so the builder's route drop
# and the calibrator's ``active_routes`` test use ONE threshold and ONE (global) denominator
# (review P1, 2026-07-26): after GDP-normalisation a share-of-global-output drop coincides with the
# calibrator's share-of-global-GDP threshold, so no route survives the builder yet is declared
# inactive (retained-but-unclearing) at calibration.
from cge.engines.cge_static.calibrate_multi import ROUTE_MATERIALITY_THRESHOLD as _ROUTE_MATERIALITY

# Default capital share of value added when no factor split is available in the build. EXIOBASE's
# value-added detail is thin; 0.4 capital / 0.6 labour is a common macro default (documented, and
# recorded in the SAM quality report as an assumption, not silently applied).
DEFAULT_CAPITAL_SHARE = 0.4

FACTORS = ["CAP", "LAB"]
HOUSEHOLD = "HOH"


@dataclass(frozen=True)
class RawSAM:
    """A raw (pre-balancing) SAM plus the audit trail of how it was built."""

    sam: SAM
    sectors: list[str]
    capital_share: float
    # audit: source aggregates the SAM must preserve (checked by quality.py)
    source_gross_output: float
    source_final_demand: float
    source_value_added: float
    value_added_clipped: float  # total negative value added clipped to zero (audit)
    # How home final demand was attributed on the OPEN build (review P1): "measured" when the
    # build carries final demand by consuming region, "imputed_import_share" when only an
    # aggregate column exists and the imported share was imputed from the intermediate-use ratio.
    # None on closed builds (no attribution needed).
    fd_attribution: str | None = None


def _gross_output(io: IOSystem) -> tuple[np.ndarray, list[str]]:
    """x = (I − A)⁻¹ · final_demand, per (region:sector) label."""
    labels = list(io.A.columns)
    A = io.A.to_numpy(dtype=float)
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
    x = np.linalg.solve(np.eye(A.shape[0]) - A, fd)
    return x, labels


def _sector_of(label: str) -> str:
    return label.split(":", 1)[1]


def build_raw_sam(io: IOSystem, *, capital_share: float = DEFAULT_CAPITAL_SHARE) -> RawSAM:
    """Build a single-region closed-economy raw SAM from a (multi-regional) ``io``.

    Regions are summed into one economy. Sectors are the build's distinct sector labels. Value
    added is derived from the IO identity and split into capital/labour by ``capital_share``.
    """
    if not 0.0 < capital_share < 1.0:
        raise ValueError(f"capital_share must be in (0,1), got {capital_share}")

    x, labels = _gross_output(io)
    A = io.A.to_numpy(dtype=float)
    # Intermediate flows Z[(r,i),(s,j)] = A · diag(x); aggregate to sector×sector.
    Z = A * x[None, :]  # column j scaled by output x_j
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)

    sectors = sorted({_sector_of(lb) for lb in labels})
    s_index = {s: k for k, s in enumerate(sectors)}
    ns = len(sectors)

    # Aggregate intermediates and final demand over regions.
    Zagg = np.zeros((ns, ns))
    FDagg = np.zeros(ns)
    Xagg = np.zeros(ns)
    for a, lb_a in enumerate(labels):
        i = s_index[_sector_of(lb_a)]
        FDagg[i] += fd[a]
        Xagg[i] += x[a]
        for b, lb_b in enumerate(labels):
            j = s_index[_sector_of(lb_b)]
            Zagg[i, j] += Z[a, b]  # supply of sector i to sector j

    # Value added per sector = output − intermediate purchases (column sum of Z into j).
    VAagg_raw = Xagg - Zagg.sum(axis=0)
    VAagg = np.clip(VAagg_raw, 0.0, None)  # guard negatives (recorded below for the audit)
    va_clip = float(np.sum(np.abs(np.minimum(VAagg_raw, 0.0))))  # total negative VA clipped

    # Assemble the SAM (row = receipts, col = payments).
    accounts = sectors + FACTORS + [HOUSEHOLD]
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)
    # Intermediates: sector i supplies sector j.
    for i, si in enumerate(sectors):
        for j, sj in enumerate(sectors):
            m.loc[si, sj] = Zagg[i, j]
    # Value added: factors paid by sectors (split by capital_share).
    for i, si in enumerate(sectors):
        m.loc["CAP", si] = capital_share * VAagg[i]
        m.loc["LAB", si] = (1.0 - capital_share) * VAagg[i]
    # Final demand: household buys commodities.
    for i, si in enumerate(sectors):
        m.loc[si, HOUSEHOLD] = FDagg[i]
    # Factor income to the household (closes the loop): all factor income flows to HOH.
    m.loc[HOUSEHOLD, "CAP"] = capital_share * VAagg.sum()
    m.loc[HOUSEHOLD, "LAB"] = (1.0 - capital_share) * VAagg.sum()

    prov = Provenance(
        source=io.provenance.source,
        source_version=io.provenance.source_version,
        licence=io.provenance.licence,
        reference_year=io.provenance.reference_year,
        retrieved=io.provenance.retrieved,
        build_id=io.provenance.build_id,
        generation=io.provenance.generation,
        notes=(
            f"single-region closed SAM from {io.provenance.build_id}; VA split "
            f"cap={capital_share} (assumption); regions summed (trade folded into domestic block)."
        ),
    )
    sam = SAM(provenance=prov, accounts=accounts, matrix=m)
    return RawSAM(
        sam=sam,
        sectors=sectors,
        capital_share=capital_share,
        source_gross_output=float(Xagg.sum()),
        source_final_demand=float(FDagg.sum()),
        # Record the *pre-clip* value added as the source aggregate, so the quality audit sees the
        # true transformation size rather than the already-clipped total (review robustness note).
        source_value_added=float(VAagg_raw.sum()),
        value_added_clipped=va_clip,
    )


def build_sam(
    io: IOSystem, *, capital_share: float = DEFAULT_CAPITAL_SHARE, balance_tol: float = 1e-6
) -> tuple[SAM, QualityReport, list[str]]:
    """Build, balance (if needed) and quality-report a SAM from ``io``.

    Returns ``(sam, quality_report, sectors)``. The closed IO construction is balanced by
    construction; if a residual imbalance exceeds ``balance_tol`` (e.g. after fabricating cells on
    thinner data) it is RAS-balanced to a common row/column total per account, and the adjustment
    magnitude is recorded in the quality report. A build whose SAM cannot be balanced, or that
    fails aggregate preservation, produces a FAIL report (the caller must not calibrate on it)."""
    raw = build_raw_sam(io, capital_share=capital_share)
    m = raw.sam.matrix
    adjustment = None
    if not is_balanced(m, tol=balance_tol):
        # Target each account's total as the mean of its row and column sums (standard RAS target).
        target = (m.sum(axis=1) + m.sum(axis=0)) / 2.0
        balanced = ras_balance(m, target, tol=balance_tol)
        adjustment = m - balanced
        m = balanced
        raw.sam.matrix.loc[:, :] = m  # keep the SAM object's matrix in sync

    report = sam_quality_report(
        io.provenance.build_id or "sam",
        m,
        source_gross_output=raw.source_gross_output,
        source_final_demand=raw.source_final_demand,
        source_value_added=raw.source_value_added,
        sectors=raw.sectors,
        factors=FACTORS,
        household=HOUSEHOLD,
        capital_share=capital_share,
        adjustment=adjustment,
        value_added_clipped=raw.value_added_clipped,
    )
    return raw.sam, report, raw.sectors


# ---------------------------------------------------------------------------
# Open-economy SAM (Phase 5 — Armington/CET on real data)
# ---------------------------------------------------------------------------


def _split_home_rest(io: IOSystem, home_region: str) -> tuple[list[str], list[str]]:
    """Partition the build's region labels into the home economy and the rest of world."""
    regions = list(io.regions.labels)
    if home_region not in regions:
        raise ValueError(f"home_region {home_region!r} not in build regions {regions}")
    if len(regions) < 2:
        raise ValueError(
            f"an open SAM needs ≥2 regions (one home + rest-of-world); build has {regions}"
        )
    return [home_region], [r for r in regions if r != home_region]


def build_open_raw_sam(
    io: IOSystem, *, home_region: str, capital_share: float = DEFAULT_CAPITAL_SHARE
) -> RawSAM:
    """Build a raw **single-region-open** SAM (activity/commodity + rest-of-world accounts) from a
    multi-regional ``io`` by treating ``home_region`` as the economy and all other regions as the
    rest of world (ROW), in the standard Armington/CET structure [Hosoe2010, ch. 7].

    Flows (per sector ``i``), from the MRIO's inter-regional blocks:
    - **domestic intermediate** ``INT[i,j]`` = home ``i`` used by home activity ``j``;
    - **exports** ``E[i]`` = home ``i`` sold to ROW (as ROW intermediates + ROW final demand);
    - **imports** ``M[i]`` = ROW ``i`` used by the home economy (home intermediates + home final
      demand);
    - **domestic sales** ``D[i]`` = home output − exports; **home final demand** ``FD[i]``.

    The aggregate trade account need not balance per sector; net foreign savings
    ``Sf = ΣM − ΣE`` is closed by a ROW→household capital transfer (the open CGE's ROW closure).
    """
    if not 0.0 < capital_share < 1.0:
        raise ValueError(f"capital_share must be in (0,1), got {capital_share}")
    home, rest = _split_home_rest(io, home_region)
    home_set, rest_set = set(home), set(rest)

    x, labels = _gross_output(io)
    A = io.A.to_numpy(dtype=float)
    Z = A * x[None, :]  # Z[a,b] = supply of label a to label b
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)

    sectors = sorted({_sector_of(lb) for lb in labels})
    s_index = {s: k for k, s in enumerate(sectors)}
    ns = len(sectors)

    def region_of(lb: str) -> str:
        return lb.split(":", 1)[0]

    # We aggregate the MRIO into a single home economy + ROW, keeping the accounts consistent so the
    # commodity balance holds **by construction** (D + M = intermediate use + final demand per
    # commodity). The composite ``c_i`` is what home activities and the home household buy;
    # it is supplied by domestic activities (``D``) and imports (``M``). Home activity output ``Z``
    # is sold domestically (``D``) or exported (``E``).
    INT = np.zeros((ns, ns))  # composite i used by home activity j (all home use of commodity i)
    Muse = np.zeros(ns)  # imports of commodity i used by home activities (intermediate only)
    FD = np.zeros(ns)  # home household final demand on composite i
    Mfd = np.zeros(ns)  # imports of commodity i for home FINAL use (measured path only)
    Xhome = np.zeros(ns)  # home activity gross output per sector
    Xrow = np.zeros(ns)  # ROW gross output per sector (for the export attribution)

    # Home final demand attribution (review P1). When the build retains final demand BY CONSUMING
    # REGION, home final demand is MEASURED: the home column of Y gives what home consumers buy
    # from every producing label — home purchases from ROW producers are imports for final use,
    # and ROW purchases of home products flow into exports via the residual. When only an
    # aggregate column exists (legacy builds), fall back to the documented imputation: the
    # region-summed final demand of home producers proxies home consumption, and the imported
    # share of final use is imputed from the intermediate-use import ratio.
    fd_region = io.fd_by_region()
    fd_measured = fd_region is not None
    if fd_measured:
        yh = fd_region[home_region].reindex(labels).fillna(0.0).to_numpy(dtype=float)

    for a, lb_a in enumerate(labels):
        ra, ia = region_of(lb_a), s_index[_sector_of(lb_a)]
        if ra in home_set:
            Xhome[ia] += x[a]
            if not fd_measured:
                # Imputed path: home producers' region-summed FD proxies home consumption.
                FD[ia] += fd[a]
        else:
            Xrow[ia] += x[a]
        if fd_measured:
            # Measured path: home consumers' purchases from label a — ANY producing region.
            FD[ia] += yh[a]
            if ra in rest_set:
                Mfd[ia] += yh[a]  # bought from a ROW producer → import for final use
        for b, lb_b in enumerate(labels):
            rb, jb = region_of(lb_b), s_index[_sector_of(lb_b)]
            if rb in home_set:
                # Every input into a home activity j goes through the composite commodity market:
                # domestic (ra home) and imported (ra ROW) alike. Track imports separately.
                INT[ia, jb] += Z[a, b]
                if ra in rest_set:
                    Muse[ia] += Z[a, b]

    #   composite supply of i = home use of i = Σ_j INT[i,j] + FD[i]
    #   imports M[i]          = imported share of that use
    #   domestic sales D[i]   = composite supply − imports
    home_use = INT.sum(axis=1) + FD  # total composite use of commodity i
    if fd_measured:
        M = Muse + Mfd  # imports measured exactly: intermediate + final use
    else:
        # Imputed path: attribute the imported share of each commodity's home use by the same
        # domestic/import ratio as intermediates (a documented, standard reduction assumption —
        # the region-summed final demand does not separate supplier region).
        with np.errstate(divide="ignore", invalid="ignore"):
            import_frac = np.where(INT.sum(axis=1) > 0, Muse / INT.sum(axis=1), 0.0)
        M = import_frac * home_use  # imports of commodity i (intermediate + final, same ratio)
    D = home_use - M  # domestic supply of the composite
    E = Xhome - D  # exports = home output not sold domestically
    Z0 = Xhome

    if float(np.min(Xhome)) <= 0:
        raise ValueError("open SAM: some home sector has non-positive gross output")
    if float(np.min(D)) <= 0:
        raise ValueError("open SAM: some home sector has non-positive domestic sales")
    if float(np.min(E)) < -1e-6 * float(Xhome.max()):
        raise ValueError(
            "open SAM: some home sector's domestic use exceeds its output (negative exports); the "
            "single-region-open reduction needs each home sector to be a net domestic supplier."
        )
    E = np.clip(E, 0.0, None)

    # Value added per home activity = output − intermediate composite purchases (column sum of INT).
    VA_raw = Z0 - INT.sum(axis=0)
    VA = np.clip(VA_raw, 0.0, None)
    va_clip = float(np.sum(np.abs(np.minimum(VA_raw, 0.0))))

    fd_attribution = "measured" if fd_measured else "imputed_import_share"
    sam = _assemble_open_sam(
        sectors, INT, E, M, D, FD, VA, capital_share, io, home_region, fd_attribution
    )
    return RawSAM(
        sam=sam,
        sectors=sectors,
        capital_share=capital_share,
        source_gross_output=float(Xhome.sum()),
        source_final_demand=float(FD.sum()),
        source_value_added=float(VA_raw.sum()),
        value_added_clipped=va_clip,
        fd_attribution=fd_attribution,
    )


def _assemble_open_sam(
    sectors, INT, E, M, D, FD, VA, capital_share, io, home_region, fd_attribution
):
    """Assemble the balanced open SAM matrix (a_<s>/c_<s>/CAP/LAB/HOH/ROW)."""
    ns = len(sectors)
    act = [f"a_{s}" for s in sectors]
    com = [f"c_{s}" for s in sectors]
    accounts = act + com + FACTORS + [HOUSEHOLD, "ROW"]
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)
    for i in range(ns):
        m.loc[act[i], com[i]] = D[i]  # activity → domestic commodity market
        m.loc[act[i], "ROW"] = E[i]  # exports
        m.loc["ROW", com[i]] = M[i]  # imports into the commodity composite
        m.loc[com[i], HOUSEHOLD] = FD[i]  # household final demand on the composite
        m.loc["CAP", act[i]] = capital_share * VA[i]
        m.loc["LAB", act[i]] = (1.0 - capital_share) * VA[i]
        for j in range(ns):
            m.loc[com[i], act[j]] = INT[i, j]  # composite i used by activity j
    m.loc[HOUSEHOLD, "CAP"] = capital_share * VA.sum()
    m.loc[HOUSEHOLD, "LAB"] = (1.0 - capital_share) * VA.sum()
    # Net foreign savings Sf = ΣM − ΣE closed by a capital transfer written in the direction of the
    # flow: a trade DEFICIT (Sf > 0) is financed by a ROW → household inflow; a trade SURPLUS
    # (Sf < 0) is home lending abroad, a household → ROW outflow. Writing the signed value into one
    # cell put a NEGATIVE entry in the SAM on the surplus side (review P1) — a valid exporter
    # economy then failed the non-negativity quality gate.
    sf = float(M.sum() - E.sum())
    if sf >= 0.0:
        m.loc[HOUSEHOLD, "ROW"] = sf
    else:
        m.loc["ROW", HOUSEHOLD] = -sf

    prov = Provenance(
        source=io.provenance.source,
        source_version=io.provenance.source_version,
        licence=io.provenance.licence,
        reference_year=io.provenance.reference_year,
        retrieved=io.provenance.retrieved,
        build_id=io.provenance.build_id,
        generation=io.provenance.generation,
        notes=(
            f"single-region-open SAM from {io.provenance.build_id}; home={home_region}, "
            f"rest-of-world = other regions; VA split cap={capital_share} (assumption); "
            f"home final demand {fd_attribution.replace('_', ' ')}"
            + (
                ""
                if fd_attribution == "measured"
                else " (SYNTHETIC: aggregate FD column only — imported share of final use "
                "imputed from the intermediate-use import ratio)"
            )
            + "."
        ),
    )
    return SAM(provenance=prov, accounts=accounts, matrix=m)


def build_open_sam(
    io: IOSystem,
    *,
    home_region: str,
    capital_share: float = DEFAULT_CAPITAL_SHARE,
    balance_tol: float = 1e-6,
) -> tuple[SAM, QualityReport, list[str]]:
    """Build, balance and quality-report an **open** SAM from ``io`` (home region + rest-of-world).

    Returns ``(sam, quality_report, sectors)``. The open reduction is not balanced by construction
    (per-sector trade is unbalanced and the regional final-demand split is approximate), so a
    residual imbalance beyond ``balance_tol`` is RAS-balanced and the adjustment is recorded. A SAM
    that cannot be balanced or fails aggregate preservation yields a FAIL report."""
    raw = build_open_raw_sam(io, home_region=home_region, capital_share=capital_share)
    m = raw.sam.matrix
    adjustment = None
    if not is_balanced(m, tol=balance_tol):
        target = (m.sum(axis=1) + m.sum(axis=0)) / 2.0
        balanced = ras_balance(m, target, tol=balance_tol)
        adjustment = m - balanced
        m = balanced
        raw.sam.matrix.loc[:, :] = m

    report = sam_quality_report(
        io.provenance.build_id or "open_sam",
        m,
        source_gross_output=raw.source_gross_output,
        source_final_demand=raw.source_final_demand,
        source_value_added=raw.source_value_added,
        sectors=raw.sectors,
        factors=FACTORS,
        household=HOUSEHOLD,
        capital_share=capital_share,
        adjustment=adjustment,
        value_added_clipped=raw.value_added_clipped,
        open_economy=True,
        fd_attribution=raw.fd_attribution,
    )
    return raw.sam, report, raw.sectors


# ---------------------------------------------------------------------------
# Multi-region SAM (Phase 5.1b — true bilateral trade among the build's regions)
# ---------------------------------------------------------------------------


class TopologyError(ValueError):
    """A multi-region build whose region-trade graph is disconnected — two regions (or blocks) with
    no active bilateral trade route between them, directly or via a chain. The multi-region CGE's
    single global numéraire + single dropped factor-market equation is only a valid closure on a
    CONNECTED region graph (a disconnected component's overall price level is genuinely
    underdetermined — see MultiCalibratedModel.connected_components). Rejected here, before
    calibration, with the offending partition, rather than silently solved to a non-unique
    equilibrium."""


@dataclass(frozen=True)
class RawMultiSAM:
    """A raw (pre-balancing) multi-region SAM plus the audit trail of how it was built."""

    sam: SAM
    regions: list[str]
    sectors: list[str]
    capital_share: float
    source_gross_output: float
    source_final_demand: float
    source_value_added: float
    value_added_clipped: float
    fd_attribution: str  # "measured" (final demand by consuming region) — required for multi


def build_multi_raw_sam(
    io: IOSystem,
    *,
    regions: list[str] | None = None,
    capital_share: float = DEFAULT_CAPITAL_SHARE,
    materiality: float = _ROUTE_MATERIALITY,
) -> RawMultiSAM:
    """Build a raw **R-region** SAM (``a_<r>_<s>``/``c_<r>_<s>`` activity/commodity, per-region
    factors ``<f>_<r>`` and households ``HOH_<r>``) from a multi-regional ``io``, in the bilateral
    Armington/CET structure [Hosoe2010, ch. 7 generalised] that ``calibrate_multi`` consumes.

    Generalises ``build_open_raw_sam`` from *home + rest-of-world* to *R genuine regions each with
    its own household and bilateral trade to every other region*. The MRIO already carries the
    inter-regional blocks, so — unlike the single-region-open reduction, where exports are a
    residual (``E = Xhome − D``) — bilateral trade is read **directly**:

    - **domestic sales** ``D[r,s]`` = region ``r``'s output of ``s`` used within region ``r`` (its
      own intermediates + own final demand of its own product);
    - **bilateral exports** ``EX[o,s,d]`` = region ``o``'s output of ``s`` used by region ``d`` (d's
      intermediates + d's final demand of o's product) — equivalently region ``d``'s **imports**
      ``M[d,s,o]``;
    - **intermediates** ``INT[r,i,j]`` = region ``r``'s composite ``i`` used by region ``r``'s
      activity ``j`` (composite = domestic variety + imports, so an imported input enters here and
      the import appears once, on the trade block);
    - **household final demand** ``FD[r,s]`` = region ``r``'s consumption of composite ``s``.

    Final demand MUST be retained by consuming region (``io.fd_by_region()``): a multi-region SAM
    attributes each region's own final demand, so an aggregate-only build (no by-region column)
    cannot be reduced without inventing the regional split — rejected, not imputed.

    ``materiality`` drops bilateral flows below this share of GLOBAL output as numerical dust
    (aggregation/RAS noise), FOLDING each dropped flow into the origin region's domestic sales so
    the SAM stays balanced. It defaults to (and must not exceed) the calibrator's
    ``ROUTE_MATERIALITY_THRESHOLD`` so every route the builder keeps is one the calibrator will
    clear — there is no retained-but-inactive gap (review P1, 2026-07-26).
    """
    if not 0.0 < capital_share < 1.0:
        raise ValueError(f"capital_share must be in (0,1), got {capital_share}")
    all_regions = list(io.regions.labels)
    regions = list(regions) if regions is not None else all_regions
    unknown = [r for r in regions if r not in all_regions]
    if unknown:
        raise ValueError(f"requested regions {unknown} not in build regions {all_regions}")
    if len(regions) < 2:
        raise ValueError(f"a multi-region SAM needs ≥2 regions; got {regions}")

    fd_region = io.fd_by_region()
    if fd_region is None:
        raise ValueError(
            "a multi-region SAM needs final demand BY CONSUMING REGION (io.fd_by_region()); this "
            "build carries only an aggregate final-demand column, so each region's own final demand "
            "cannot be attributed without inventing the split — rebuild with by-region final demand."
        )

    x, labels = _gross_output(io)
    A = io.A.to_numpy(dtype=float)
    Z = A * x[None, :]  # Z[a,b] = supply of label a to label b

    sectors = sorted({_sector_of(lb) for lb in labels})
    s_index = {s: k for k, s in enumerate(sectors)}
    r_index = {r: k for k, r in enumerate(regions)}
    region_set = set(regions)
    nr, ns = len(regions), len(sectors)

    def region_of(lb: str) -> str:
        return lb.split(":", 1)[0]

    # Trade block T[o, s, d] = region o's output of s used by region d (intermediate + final). The
    # own-region slot (o==d) is domestic sales D[r,s]. Off-diagonal is a bilateral export/import.
    T = np.zeros((nr, ns, nr))  # [origin, sector, dest]
    INT = np.zeros((nr, ns, ns))  # [r, i, j] composite i used by r's activity j (any source region)
    FD = np.zeros((nr, ns))  # [r, s] region r final demand on composite s (any source region)
    Xreg = np.zeros((nr, ns))  # [r, s] region r gross output of s

    # Only the requested regions form the modelled economy; a build may carry more regions than we
    # model, in which case flows to/from unmodelled regions are folded into the nearest modelled
    # aggregate is NOT attempted — we require regions to be the full build (checked below) so the
    # global economy is closed. (Subsetting to a strict subset would leak trade off-model.)
    if region_set != set(all_regions):
        raise ValueError(
            "multi-region SAM must model EVERY build region (the global economy is closed with no "
            f"external rest-of-world); build has {all_regions}, requested {regions}. Aggregate the "
            "build to the desired regions first (build a coarser IOSystem), then model all of them."
        )

    for a, lb_a in enumerate(labels):
        ra, ia = region_of(lb_a), s_index[_sector_of(lb_a)]
        if ra not in region_set:
            continue
        Xreg[r_index[ra], ia] += x[a]
        # Final demand of label a (an o-region product) by each consuming region d.
        for d in regions:
            yad = float(fd_region[d].get(lb_a, 0.0))
            T[r_index[ra], ia, r_index[d]] += yad  # o's product consumed by d
        for b, lb_b in enumerate(labels):
            rb, jb = region_of(lb_b), s_index[_sector_of(lb_b)]
            if rb not in region_set:
                continue
            # o=ra supplies its product a into activity b (in region rb): a trade flow o→rb, and it
            # is an intermediate input into rb's composite ia used by rb's activity jb.
            T[r_index[ra], ia, r_index[rb]] += Z[a, b]
            INT[r_index[rb], ia, jb] += Z[a, b]

    # Domestic sales / bilateral trade split off the diagonal of T. Trade materiality (review P1,
    # 2026-07-26): a route below the threshold is dropped AND its value is FOLDED INTO DOMESTIC
    # SALES of the origin region, so (a) the SAM stays balanced by construction (nothing vanishes),
    # and (b) the builder and the calibrator use ONE threshold and ONE denominator — a global scale
    # so the drop matches ``calibrate_multi.ROUTE_MATERIALITY_THRESHOLD`` (a share of global output,
    # which after GDP-normalisation is the calibrator's ``active_routes`` test). This closes the gap
    # the review found: previously the builder dropped below ``1e-9 × regional output`` while the
    # calibrator declared inactive below ``1e-6 × global GDP``, so a route between the two was kept
    # with positive Armington/CET shares (used by demand/supply) but had no clearing residual.
    global_scale = max(float(Xreg.sum()), 1.0)
    threshold = materiality * global_scale
    D = np.zeros((nr, ns))
    EX = np.zeros((nr, ns, nr))  # [o, s, d] exports (o≠d)
    for ri in range(nr):
        for si in range(ns):
            D[ri, si] = T[ri, si, ri]
            for di in range(nr):
                if di == ri:
                    continue
                v = T[ri, si, di]
                if v > threshold:
                    EX[ri, si, di] = v
                else:
                    D[ri, si] += v  # fold a dropped dust route into domestic sales (balance-safe)

    # Household final demand per region-sector: consumption of composite s by region r, from ANY
    # producing region (measured, by consuming region), read directly off fd_region over the
    # composite (sector) axis (T folds the final-demand part into the trade block for D/EX above).
    FD = np.zeros((nr, ns))
    for a, lb_a in enumerate(labels):
        ia = s_index[_sector_of(lb_a)]
        if region_of(lb_a) not in region_set:
            continue
        for d in regions:
            FD[r_index[d], ia] += float(fd_region[d].get(lb_a, 0.0))

    Z0 = D + EX.sum(axis=2)  # activity output = domestic sales + all exports, per region-sector
    # Value added per region-activity = output − intermediate composite purchases.
    VA_raw = Z0 - INT.sum(axis=1)  # INT[r,i,j] summed over i (composite) → per activity j
    VA = np.clip(VA_raw, 0.0, None)
    va_clip = float(np.sum(np.abs(np.minimum(VA_raw, 0.0))))

    if float(np.min(Z0)) <= 0:
        raise ValueError("multi SAM: some region-sector has non-positive gross output")

    sam = _assemble_multi_sam(regions, sectors, D, EX, INT, FD, VA, capital_share, io)
    return RawMultiSAM(
        sam=sam,
        regions=regions,
        sectors=sectors,
        capital_share=capital_share,
        source_gross_output=float(Z0.sum()),
        source_final_demand=float(FD.sum()),
        source_value_added=float(VA_raw.sum()),
        value_added_clipped=va_clip,
        fd_attribution="measured",
    )


def _assemble_multi_sam(regions, sectors, D, EX, INT, FD, VA, capital_share, io):
    """Assemble the multi-region SAM matrix (a_<r>_<s>/c_<r>_<s>/<f>_<r>/HOH_<r>) with the bilateral
    trade block and the household↔household current-account closure."""
    nr, ns = len(regions), len(sectors)
    accounts: list[str] = []
    for r in regions:
        accounts += [f"a_{r}_{s}" for s in sectors] + [f"c_{r}_{s}" for s in sectors]
    for r in regions:
        accounts += [f"{f}_{r}" for f in FACTORS] + [f"HOH_{r}"]
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)

    for ri, r in enumerate(regions):
        for si, s in enumerate(sectors):
            m.loc[f"a_{r}_{s}", f"c_{r}_{s}"] = D[ri, si]  # domestic sales into own composite
            m.loc[f"c_{r}_{s}", f"HOH_{r}"] = FD[ri, si]  # household final demand
            m.loc[f"CAP_{r}", f"a_{r}_{s}"] = capital_share * VA[ri, si]
            m.loc[f"LAB_{r}", f"a_{r}_{s}"] = (1.0 - capital_share) * VA[ri, si]
            for di, d in enumerate(regions):
                if di == ri:
                    continue
                # r's activity s sells into d's composite s: payment c_<d>_<s> → a_<r>_<s>.
                if EX[ri, si, di] > 0:
                    m.loc[f"a_{r}_{s}", f"c_{d}_{s}"] = EX[ri, si, di]
            for ji, j in enumerate(sectors):
                m.loc[f"c_{r}_{s}", f"a_{r}_{j}"] = INT[ri, si, ji]  # composite s into activity j
    # Factor income to each region's household.
    for r in regions:
        m.loc[f"HOH_{r}", f"CAP_{r}"] = m.loc[f"CAP_{r}", :].sum()
        m.loc[f"HOH_{r}", f"LAB_{r}"] = m.loc[f"LAB_{r}", :].sum()
    # Current-account closure: region r's net foreign savings Sf_r = imports − exports (globally
    # zero-sum) financed by a bilateral HOH↔HOH capital transfer, exactly as the toy multi SAM does.
    _add_multi_capital_transfers(m, regions, sectors)

    prov = Provenance(
        source=io.provenance.source,
        source_version=io.provenance.source_version,
        licence=io.provenance.licence,
        reference_year=io.provenance.reference_year,
        retrieved=io.provenance.retrieved,
        build_id=io.provenance.build_id,
        generation=io.provenance.generation,
        notes=(
            f"multi-region SAM from {io.provenance.build_id}; regions={regions}; bilateral "
            f"Armington/CET; VA split cap={capital_share} (assumption); final demand measured by "
            "consuming region; current accounts closed by HOH↔HOH capital transfers (globally "
            "zero-sum)."
        ),
    )
    return SAM(provenance=prov, accounts=accounts, matrix=m)


def _add_multi_capital_transfers(m: pd.DataFrame, regions: list[str], sectors: list[str]) -> None:
    """Close each region's current account with an HOH↔HOH capital transfer (surplus regions finance
    deficit regions, proportional to surplus) — the same zero-sum settlement as the toy multi SAM,
    so the per-region and global accounts balance by construction."""
    ca = {}
    for r in regions:
        exports = sum(m.loc[f"a_{r}_{s}", f"c_{d}_{s}"] for d in regions if d != r for s in sectors)
        imports = sum(m.loc[f"a_{o}_{s}", f"c_{r}_{s}"] for o in regions if o != r for s in sectors)
        ca[r] = float(exports - imports)  # >0 ⇒ surplus (net lender)
    total_surplus = sum(v for v in ca.values() if v > 0)
    if total_surplus <= 0:
        return
    for lender in regions:
        if ca[lender] <= 0:
            continue
        for borrower in regions:
            if ca[borrower] >= 0:
                continue
            amount = (-ca[borrower]) * (ca[lender] / total_surplus)
            m.loc[f"HOH_{borrower}", f"HOH_{lender}"] += amount


def build_multi_sam(
    io: IOSystem,
    *,
    regions: list[str] | None = None,
    capital_share: float = DEFAULT_CAPITAL_SHARE,
    materiality: float = _ROUTE_MATERIALITY,
    balance_tol: float = 1e-6,
) -> tuple[SAM, QualityReport, list[str], list[str]]:
    """Build, balance, quality-report and **topology-validate** a multi-region SAM from ``io``.

    Returns ``(sam, quality_report, regions, sectors)``. The bilateral reduction is not balanced by
    construction (the regional final-demand split and VA derivation are approximate), so a residual
    imbalance beyond ``balance_tol`` is RAS-balanced and the adjustment recorded. The region-trade
    graph is checked for **connectivity** after balancing — a disconnected build raises
    ``TopologyError`` (the multi-region closure is only valid on a connected graph), which is far
    more likely on a live aggregated build than on a hand-built toy."""
    raw = build_multi_raw_sam(
        io, regions=regions, capital_share=capital_share, materiality=materiality
    )
    m = raw.sam.matrix
    adjustment = None
    if not is_balanced(m, tol=balance_tol):
        target = (m.sum(axis=1) + m.sum(axis=0)) / 2.0
        balanced = ras_balance(m, target, tol=balance_tol)
        adjustment = m - balanced
        m = balanced
        raw.sam.matrix.loc[:, :] = m

    _assert_multi_connected(m, raw.regions, raw.sectors, materiality)

    report = sam_quality_report(
        io.provenance.build_id or "multi_sam",
        m,
        source_gross_output=raw.source_gross_output,
        source_final_demand=raw.source_final_demand,
        source_value_added=raw.source_value_added,
        sectors=raw.sectors,
        factors=FACTORS,
        household=HOUSEHOLD,
        capital_share=capital_share,
        adjustment=adjustment,
        value_added_clipped=raw.value_added_clipped,
        fd_attribution=raw.fd_attribution,
        regions=raw.regions,
    )
    return raw.sam, report, raw.regions, raw.sectors


def _assert_multi_connected(
    m: pd.DataFrame, regions: list[str], sectors: list[str], materiality: float
) -> None:
    """Reject a disconnected region-trade graph (Phase 5.1b topology validation). Two regions are
    linked if they trade ANY good in EITHER direction above the materiality threshold; the graph
    must be a single connected component or the multi-region closure (one numéraire, one dropped
    factor equation) is under-determined. Mirrors MultiCalibratedModel.connected_components so the
    data-side gate matches the calibration-side invariant, but rejects here with a clear message."""
    nr = len(regions)
    parent = list(range(nr))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # ONE denominator (global output) and ONE threshold, matching the route-drop step above and the
    # calibrator's ``active_routes`` (review P1): a route counts as a trade link iff it clears at
    # calibration, so the connectivity graph is exactly the calibrator's ``active_routes`` graph.
    threshold = materiality * max(
        float(sum(m.loc[f"a_{o}_{s}", :].sum() for o in regions for s in sectors)), 1.0
    )
    for oi, o in enumerate(regions):
        for di, d in enumerate(regions):
            if oi == di:
                continue
            traded = any(float(m.loc[f"a_{o}_{s}", f"c_{d}_{s}"]) > threshold for s in sectors)
            if traded:
                ri, rj = find(oi), find(di)
                if ri != rj:
                    parent[ri] = rj
    comps: dict[int, list[str]] = {}
    for i, r in enumerate(regions):
        comps.setdefault(find(i), []).append(r)
    if len(comps) > 1:
        raise TopologyError(
            "multi-region build has a DISCONNECTED region-trade graph: components "
            f"{[sorted(c) for c in comps.values()]} have no active bilateral trade route between "
            "them, so the single-numéraire multi-region closure is under-determined. This is not a "
            "solvable equilibrium — aggregate the disconnected regions together or add the missing "
            "trade link."
        )
