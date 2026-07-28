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


def build_raw_sam(
    io: IOSystem, *, capital_share: float = DEFAULT_CAPITAL_SHARE, institutions: bool = True
) -> RawSAM:
    """Build a single-region closed-economy raw SAM from a (multi-regional) ``io``.

    Regions are summed into one economy. Sectors are the build's distinct sector labels. Value
    added is derived from the IO identity and split into capital/labour by ``capital_share``.

    ``institutions`` (default True, Phase 5.1b+ review P1 round 13): when the build carries the
    final-demand INSTITUTION split (``io.fd_by_institution()`` — EXIOBASE Y categories classified
    into household/government/investment), route government consumption to a ``GOV`` account and
    gross capital formation to a ``SAVINV`` account, instead of lumping ALL final demand into the
    household. Government spending is financed by an IMPUTED household→gov direct tax equal to
    that spending (EXIOBASE Y carries no tax detail — the tax is an explicit, provenance-flagged
    imputation, not sourced), and investment by imputed household savings; the SAM stays balanced by
    construction. This lets the macro closures (gov/investment) run on a REAL built SAM. With
    no institution split (or ``institutions=False``) all final demand goes to the household, exactly
    as before."""
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

    # Institution split (national totals per sector), if the build carries it.
    fbi = io.fd_by_institution() if institutions else None
    gov_fd = np.zeros(ns)
    inv_fd = np.zeros(ns)
    hh_fd = np.zeros(ns)
    if fbi is not None:
        sep = IOSystem.INSTITUTION_SEP
        for col in fbi.columns:
            inst = str(col).split(sep)[1]
            vec = fbi[col].reindex(labels).fillna(0.0).to_numpy(dtype=float)
            target = {"government": gov_fd, "investment": inv_fd, "household": hh_fd}[inst]
            for a, lb_a in enumerate(labels):
                target[s_index[_sector_of(lb_a)]] += vec[a]

    # Aggregate intermediates and (total) final demand over regions.
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

    # Reconcile the institution split with total final demand FDagg per sector. In this CLOSED
    # (region-summed) build the institution split (household+gov+investment) should already sum to
    # FDagg — the only legitimate gap is small aggregation/rounding DRIFT. A LARGE gap means the
    # split does not decompose this build's final demand: REJECT it rather than silently rescale
    # (review P1 round 14 — a 0.4 split against 200 FD was amplified 500× into a plausible SAM).
    # a small drift is normalised, and its magnitude is recorded for the quality audit.
    institution_rescale = 0.0
    if fbi is not None:
        split_total = gov_fd + inv_fd + hh_fd
        agg_split = float(split_total.sum())
        agg_fd = float(FDagg.sum())
        rel_gap = abs(agg_split - agg_fd) / max(abs(agg_fd), 1.0)
        if rel_gap > 0.01:  # >1% aggregate gap is corruption, not drift
            raise ValueError(
                f"final-demand institution split totals {agg_split:.6g} but total final demand is "
                f"{agg_fd:.6g} (rel gap {rel_gap:.2%} > 1%): the split does not decompose this "
                "build's final demand. Rejecting rather than rescaling it into a plausible SAM "
                "(supply a split consistent with final_demand, or disable institutions)."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            scale_i = np.where(split_total > 0, FDagg / split_total, 0.0)
        pos = split_total > 0
        institution_rescale = float(np.abs(scale_i[pos] - 1.0).max()) if pos.any() else 0.0
        gov_fd, inv_fd = gov_fd * scale_i, inv_fd * scale_i
        hh_fd = FDagg - gov_fd - inv_fd  # household is the residual; the three sum to FDagg exactly

    has_institutions = fbi is not None and (gov_fd.sum() > 0 or inv_fd.sum() > 0)
    # When the institution split is present, household FD is its own column; else it is all of FD.
    hh_col = hh_fd if has_institutions else FDagg

    # Assemble the SAM (row = receipts, col = payments).
    accounts = sectors + FACTORS + [HOUSEHOLD]
    if has_institutions:
        accounts = accounts + ["GOV", "SAVINV"]
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)
    # Intermediates: sector i supplies sector j.
    for i, si in enumerate(sectors):
        for j, sj in enumerate(sectors):
            m.loc[si, sj] = Zagg[i, j]
    # Value added: factors paid by sectors (split by capital_share).
    for i, si in enumerate(sectors):
        m.loc["CAP", si] = capital_share * VAagg[i]
        m.loc["LAB", si] = (1.0 - capital_share) * VAagg[i]
    # Final demand: household buys its own consumption column.
    for i, si in enumerate(sectors):
        m.loc[si, HOUSEHOLD] = hh_col[i]
    # Factor income to the household (closes the loop): all factor income flows to HOH.
    m.loc[HOUSEHOLD, "CAP"] = capital_share * VAagg.sum()
    m.loc[HOUSEHOLD, "LAB"] = (1.0 - capital_share) * VAagg.sum()
    if has_institutions:
        # Government buys its consumption column, financed by an IMPUTED HOH→GOV direct tax = its
        # spending (EXIOBASE Y has no tax detail — flagged in provenance). Investment (SAVINV) buys
        # gross capital formation, financed by imputed household savings. Both balance by
        # construction: GOV row = GOV col = gov spend; SAVINV row = SAVINV col = investment.
        for i, si in enumerate(sectors):
            m.loc[si, "GOV"] = gov_fd[i]
            m.loc[si, "SAVINV"] = inv_fd[i]
        m.loc["GOV", HOUSEHOLD] = float(gov_fd.sum())  # imputed direct tax
        m.loc["SAVINV", HOUSEHOLD] = float(inv_fd.sum())  # imputed household savings

    inst_note = (
        f" government (imputed tax {gov_fd.sum():.4g}) + investment (imputed savings "
        f"{inv_fd.sum():.4g}) routed to GOV/SAVINV from the EXIOBASE FD institution split "
        f"(tax/savings IMPUTED — no source tax detail; split↔FD reconciliation rescaled by at most "
        f"{institution_rescale:.2%})."
        if has_institutions
        else " all final demand to the household (no institution split available)."
    )
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
            f"cap={capital_share} (assumption); regions summed (trade folded into domestic block);"
            + inst_note
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

    This raw builder keeps **every** bilateral route; dust removal (dropping tiny routes and
    RAS-rebalancing so the SAM stays balanced) is done in ``build_multi_sam`` on the assembled
    square matrix — the balance-safe place to do it (review P1, 2026-07-27).
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
            "build carries only an aggregate final-demand column, so each region's own final "
            "demand cannot be attributed without inventing the split — rebuild with by-region FD."
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

    # Domestic sales / bilateral trade split off the diagonal of T. ALL off-diagonal flows are kept
    # here (dust removal + rebalancing is done in ``build_multi_sam`` on the assembled SAM, review
    # P1 2026-07-27): folding a dropped export into the ORIGIN's domestic sales at this point leaves
    # the DESTINATION's intermediate/final uses of that import unchanged, so the commodity balance
    # breaks (the earlier "balance-safe" fold was not). The right place to drop dust is on the whole
    # square SAM, followed by a RAS re-balance that restores every account's row=col sum.
    D = np.zeros((nr, ns))
    EX = np.zeros((nr, ns, nr))  # [o, s, d] exports (o≠d)
    for ri in range(nr):
        for si in range(ns):
            D[ri, si] = T[ri, si, ri]
            for di in range(nr):
                if di != ri:
                    EX[ri, si, di] = T[ri, si, di]

    # Household final demand per region-sector: consumption of composite s by region r, from ANY
    # producing region (measured, by consuming region), read directly off fd_region over the
    # composite (sector) axis (T folds the final-demand part into the trade block for D/EX above).
    FD = np.zeros((nr, ns))
    for lb_a in labels:
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


def _bilateral_trade_by_route(
    io: IOSystem, regions: list[str], sectors: list[str]
) -> dict[tuple[str, str, str], float]:
    """Bilateral trade value on each PER-SECTOR route ``(o, d, s)`` (o≠d) — o's product ``s``
    consumed by d (intermediates + final demand). This is exactly the granularity
    ``build_multi_sam``
    checks for dust (a route is one ``a_<o>_<s> → c_<d>_<s>`` cell), so ``aggregate_dust_regions``
    must merge on the same granularity to eliminate it."""
    labels = list(io.A.columns)
    x, _ = _gross_output(io)
    A = io.A.to_numpy(dtype=float)
    Z = A * x[None, :]
    fd_region = io.fd_by_region()
    out: dict[tuple[str, str, str], float] = {}

    def region_of(lb):
        return lb.split(":", 1)[0]

    def sector_of(lb):
        return lb.split(":", 1)[1]

    for a, lb_a in enumerate(labels):
        ra, sa = region_of(lb_a), sector_of(lb_a)
        if fd_region is not None:
            for d in regions:
                if d != ra:
                    v = float(fd_region[d].get(lb_a, 0.0))
                    if v:
                        out[(ra, d, sa)] = out.get((ra, d, sa), 0.0) + v
        for b, lb_b in enumerate(labels):
            rb = region_of(lb_b)
            if rb != ra and Z[a, b]:
                out[(ra, rb, sa)] = out.get((ra, rb, sa), 0.0) + float(Z[a, b])
    return out


def aggregate_dust_regions(
    io: IOSystem,
    satellites: list | None = None,
    *,
    materiality: float = _ROUTE_MATERIALITY,
    build_id: str | None = None,
):
    """Fold low-trade region pairs into coarser region GROUPS so that no surviving bilateral route
    is dust (review P1 round 14, 2026-07-28) — the explicit upstream workflow the dust-rejection
    error points to. Greedily merges the region with the most sub-threshold outgoing trade into its
    largest trading partner, repeating until every remaining cross-group bilateral flow is
    ≥ ``materiality × global GDP`` (so ``build_multi_sam`` accepts the result). Trade *within* a
    merged group becomes intra-regional (domestic) and drops out of the bilateral block, which is
    exactly where negligible inter-region trade belongs.

    Returns ``(coarse_io, coarse_satellites, grouping)`` where ``grouping`` maps each ORIGINAL
    region to its GROUP label (the recorded transformation). A build that already has no dust
    returns the identity grouping and the input unchanged (aggregated trivially)."""
    from datetime import date

    from cge.data.aggregate import aggregate_io
    from cge.data.concordance.concordance import one_to_one
    from cge.data.metadata import BuildMeta

    satellites = satellites or []
    regions = list(io.regions.labels)
    gdp = float(io.final_demand.sum(axis=1).sum())  # total absorption ≈ GDP scale for the threshold
    threshold = materiality * max(gdp, 1.0)

    # Group assignment (union-find), seeded with each region its own group.
    group = {r: r for r in regions}

    def _find(r):
        while group[r] != r:
            group[r] = group[group[r]]
            r = group[r]
        return r

    def _merge(a, b):
        group[_find(a)] = _find(b)

    routes = _bilateral_trade_by_route(io, regions, list(io.sectors.labels))
    for _ in range(len(regions)):  # at most nr merges reach a single group
        # A group PAIR is dust if ANY per-sector route between the groups is a nonzero sub-threshold
        # flow (the builder rejects at that granularity). Track, per group pair, the total trade
        # (to merge sensibly) and whether it has a dust route.
        grp: dict[tuple[str, str], list[float]] = {}  # (go,gd) -> [total, has_dust(0/1)]
        for (o, d, _s), v in routes.items():
            go, gd = _find(o), _find(d)
            if go == gd:
                continue
            entry = grp.setdefault((go, gd), [0.0, 0.0])
            entry[0] += v
            if 0.0 < v < threshold:
                entry[1] = 1.0
        dusty = {pair: tot for pair, (tot, hasdust) in grp.items() if hasdust}
        if not dusty:
            break
        # Merge the dusty group pair carrying the MOST total trade (minimal, sensible transformation
        # — fold the pair that is most connected, so the merged group's internal trade is genuine).
        (o, d), _tot = max(dusty.items(), key=lambda kv: kv[1])
        _merge(o, d)

    grouping = {r: _find(r) for r in regions}
    # Rename groups to a compact, stable label = the sorted members joined (so it is auditable).
    members: dict[str, list[str]] = {}
    for r, g in grouping.items():
        members.setdefault(g, []).append(r)
    label_of = {g: "+".join(sorted(ms)) if len(ms) > 1 else ms[0] for g, ms in members.items()}
    grouping = {r: label_of[g] for r, g in grouping.items()}

    n_groups = len(set(grouping.values()))
    if n_groups < 2:
        raise ValueError(
            "aggregate_dust_regions collapsed to a single region: this build's inter-region trade "
            "is entirely dust at the sector granularity (every region pair has a sub-threshold "
            "sector route), so no genuine multi-region structure survives. Aggregate SECTORS too "
            "(a coarser sector map raises per-route flows above the threshold), lower "
            "`materiality` if the flows are real-but-small trade, or model this build with the "
            "single-region closed CGE. (This is the offline pymrio test MRIO's situation — its "
            "cross-region trade is genuinely negligible; the coarse 2×3 custom build in the "
            "validation suite has real "
            "trade and does build.)"
        )

    region_cmap = one_to_one(
        grouping,
        from_classification=io.regions.name,
        to_classification="dust-aggregated-regions",
        provenance=io.provenance,
    )
    sector_cmap = one_to_one(
        {s: s for s in io.sectors.labels},
        from_classification=io.sectors.name,
        to_classification=io.sectors.name,
        provenance=io.provenance,
    )
    meta = BuildMeta(
        build_id=io.provenance.build_id or "io",
        source=io.provenance.source,
        source_version=io.provenance.source_version,
        licence=io.provenance.licence,
        reference_year=io.provenance.reference_year,
        currency=io.currency,
        monetary_unit=io.unit,
        final_demand_kind=io.final_demand_kind,
        retrieved=io.provenance.retrieved or date.today().isoformat(),
    )
    new_id = build_id or f"{meta.build_id}-dustagg"
    coarse_io, coarse_sats, _m = aggregate_io(
        io,
        satellites,
        sector_cmap=sector_cmap,
        region_cmap=region_cmap,
        meta=meta,
        new_build_id=new_id,
        aggregation_name="dust-region-aggregation",
    )
    return coarse_io, coarse_sats, grouping


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
    more likely on a live aggregated build than on a hand-built toy.

    ``materiality`` is the dust threshold as a share of global GDP (bounded contract, review P1
    2026-07-27): it must lie in ``[ROUTE_MATERIALITY_THRESHOLD, 0.1)`` — at least the calibrator's
    threshold, and below 10% of GDP (a larger value would erase real trade, not dust).

    **Dust routes are REJECTED, not repaired (review P1, round 13, 2026-07-28).** An earlier version
    zeroed a sub-threshold bilateral cell and RAS-rebalanced; that transformation is **not**
    balance-preserving for an *asymmetric* dust route (zeroing an off-diagonal cell can leave the
    row and column targets incompatible with the imposed zero pattern, so RAS fails to converge —
    reproduced). A dropped flow also has to go *somewhere*, and there is no local exact place to put
    it (§8a: the domestic diagonal is shared between the Armington and CET identities). So the safe
    contract is: a route is either genuine trade (≥ threshold) or exactly zero; a nonzero
    sub-threshold route is a **domain error** the caller must fix upstream (aggregate the regions
    coarser, or clean the source), NOT something the builder silently rewrites."""
    if not (_ROUTE_MATERIALITY <= materiality < 0.1):
        raise ValueError(
            f"materiality must be in [{_ROUTE_MATERIALITY:g}, 0.1) (a share of global GDP): at "
            "least the calibrator's ROUTE_MATERIALITY_THRESHOLD so the built SAM passes the "
            f"calibrator's dust gate, and below 0.1 so real trade is not erased; got {materiality}."
        )
    raw = build_multi_raw_sam(io, regions=regions, capital_share=capital_share)
    m = raw.sam.matrix
    adjustment = None
    if not is_balanced(m, tol=balance_tol):
        target = (m.sum(axis=1) + m.sum(axis=0)) / 2.0
        balanced = ras_balance(m, target, tol=balance_tol)
        adjustment = m - balanced
        m = balanced
        raw.sam.matrix.loc[:, :] = m

    # Trade-materiality: REJECT dust routes (see the docstring). The threshold is a share of ACTUAL
    # global GDP (total value added) — the same normalisation the calibrator uses (it divides by
    # GDP), so a route the builder accepts is exactly one the calibrator will accept, with no
    # ``max(GDP, 1)`` scale break (review P1 r13). ``>= threshold`` keeps; ``0 < v < threshold``
    # rejects; exactly 0 is a structural non-route.
    gdp = float(sum(m.loc[f"{f}_{r}", :].sum() for r in raw.regions for f in FACTORS))
    dust_threshold = materiality * gdp
    _reject_dust_cells(m, raw.regions, raw.sectors, dust_threshold)

    _assert_multi_connected(m, raw.regions, raw.sectors, dust_threshold)

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


def _reject_dust_cells(
    m: pd.DataFrame, regions: list[str], sectors: list[str], dust_threshold: float
) -> None:
    """Reject a built SAM carrying a **dust** bilateral trade cell ``a_<o>_<s> → c_<d>_<s>`` (o≠d)
    that is positive but below ``dust_threshold`` (review P1 round 13, 2026-07-28). A route is
    either genuine trade (≥ threshold, which gets a clearing equation at calibration) or exactly
    zero; a sub-threshold nonzero cell would get a near-singular price column, so — since there is
    no exact, balance-preserving way to remove it locally — it is a domain error the caller must fix
    upstream (aggregate to coarser regions, or clean the source data), not something the builder
    silently rewrites."""
    dust = []
    for o in regions:
        for d in regions:
            if o == d:
                continue
            for s in sectors:
                v = float(m.loc[f"a_{o}_{s}", f"c_{d}_{s}"])
                if 0.0 < v < dust_threshold:
                    dust.append((o, d, s, v))
    if dust:
        o, d, s, v = dust[0]
        raise ValueError(
            f"built multi-region SAM has {len(dust)} dust trade route(s) below the materiality "
            f"threshold {dust_threshold:.3e} (materiality × global GDP) — e.g. {o}→{d} sector {s} "
            f"= {v:.3e}. A bilateral route must be genuine trade (≥ threshold) or exactly zero; a "
            "tiny route gets a near-singular price column the solver cannot pin, and there is no "
            "exact balance-preserving way to remove it. Fix upstream: call "
            "``aggregate_dust_regions(io, ...)`` to fold low-trade region pairs into coarser "
            "groups (recording the transformation) so no dust remains, OR — if these flows really "
            "are negligible non-trade — LOWER ``materiality`` so they fall below it and are "
            "treated as structural zeros. (Raising ``materiality`` does the OPPOSITE: it "
            "classifies MORE "
            "routes as dust.)"
        )


def _assert_multi_connected(
    m: pd.DataFrame, regions: list[str], sectors: list[str], threshold: float
) -> None:
    """Reject a disconnected region-trade graph (Phase 5.1b topology validation). Two regions are
    linked if they trade ANY good in EITHER direction at or above ``threshold`` — the SAME
    ``>= threshold`` test the dust rejection uses to KEEP a route (review P1 round 13), so a link is
    exactly a route that survives to be cleared at calibration (no exactly-at-threshold route that
    is kept but counted as no link). The graph must be a single connected component or the
    multi-region closure (one numéraire, one dropped factor equation) is under-determined. Mirrors
    MultiCalibratedModel.connected_components but rejects here with a clear message."""
    nr = len(regions)
    parent = list(range(nr))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for oi, o in enumerate(regions):
        for di, d in enumerate(regions):
            if oi == di:
                continue
            traded = any(float(m.loc[f"a_{o}_{s}", f"c_{d}_{s}"]) >= threshold for s in sectors)
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
