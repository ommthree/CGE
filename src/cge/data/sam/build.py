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
    Muse = np.zeros(ns)  # imports of commodity i used by the home economy (intermediate + final)
    FD = np.zeros(ns)  # home household final demand on composite i
    Xhome = np.zeros(ns)  # home activity gross output per sector
    Xrow = np.zeros(ns)  # ROW gross output per sector (for the export attribution)

    for a, lb_a in enumerate(labels):
        ra, ia = region_of(lb_a), s_index[_sector_of(lb_a)]
        if ra in home_set:
            Xhome[ia] += x[a]
            FD[ia] += fd[
                a
            ]  # home final demand for good ia (home-consumed; supplier resolved below)
        else:
            Xrow[ia] += x[a]
        for b, lb_b in enumerate(labels):
            rb, jb = region_of(lb_b), s_index[_sector_of(lb_b)]
            if rb in home_set:
                # Every input into a home activity j goes through the composite commodity market:
                # domestic (ra home) and imported (ra ROW) alike. Track imports separately.
                INT[ia, jb] += Z[a, b]
                if ra in rest_set:
                    Muse[ia] += Z[a, b]

    # Home final demand is met partly by imports; attribute the imported share of each commodity's
    # home use by the same domestic/import ratio as intermediates (a documented, standard reduction
    # assumption — the region-summed final demand does not separate supplier region). Then:
    #   composite supply of i = home use of i = Σ_j INT[i,j] + FD[i]
    #   imports M[i]          = imported share of that use
    #   domestic sales D[i]   = composite supply − imports
    home_use = INT.sum(axis=1) + FD  # total composite use of commodity i
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

    sam = _assemble_open_sam(sectors, INT, E, M, D, FD, VA, capital_share, io, home_region)
    return RawSAM(
        sam=sam,
        sectors=sectors,
        capital_share=capital_share,
        source_gross_output=float(Xhome.sum()),
        source_final_demand=float(FD.sum()),
        source_value_added=float(VA_raw.sum()),
        value_added_clipped=va_clip,
    )


def _assemble_open_sam(sectors, INT, E, M, D, FD, VA, capital_share, io, home_region):
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
    # Net foreign savings Sf = ΣM − ΣE closed by a ROW → household capital transfer.
    m.loc[HOUSEHOLD, "ROW"] = float(M.sum() - E.sum())

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
            f"rest-of-world = other regions; VA split cap={capital_share} (assumption)."
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
    )
    return raw.sam, report, raw.sectors
