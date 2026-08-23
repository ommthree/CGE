"""A hand-checkable MULTI-REGION toy SAM (Phase 5.4 — true bilateral trade).

Extends the single-region-open toy to a **closed global economy** of ``R`` regions that trade only
with each other (no external rest-of-world), in the bilateral Armington/CET structure
[Hosoe2010, ch. 7 generalised to many regions]. Each region has the same sectors, its own factors
and household, and:

- **activities** ``a_<r>_<s>`` produce regional output, sold domestically or **exported to each
  other region**;
- **commodities** ``c_<r>_<s>`` are the Armington composite bought by that region's intermediates
  and household — a CES over the region's own domestic variety **and imports from every other
  region**;
- **factors** ``CAP_<r>`` / ``LAB_<r>`` (region-specific, immobile across regions);
- **household** ``HOH_<r>``.

**Bilateral trade.** ``T[o→d, s]`` is region ``o``'s export of commodity ``s`` to region ``d`` (=
region ``d``'s import of ``s`` from ``o``). In the SAM this is a payment from ``c_<d>_<s>`` (the
importing composite) to ``a_<o>_<s>`` (the exporting activity). The **global** trade account
balances (Σ exports = Σ imports across all regions); each region's current account may be non-zero,
closed by a bilateral capital transfer to its household (the ROW closure, generalised).

This 2-region × 2-sector instance is small enough to hand-check and is the exact replication target
for the multi-region calibration + model.

**Convention** (as in ``toy_open.py``): ``M[row, col]`` is a payment from account ``col`` to
account ``row``; the matrix is balanced (row sum = column sum per account).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from cge.contracts.data_objects import SAM, Provenance

REGIONS = ["N", "S"]  # North, South
SECTORS = ["BRD", "MIL"]


def _acc(regions: list[str], sectors: list[str]) -> list[str]:
    accounts: list[str] = []
    for r in regions:
        accounts += [f"a_{r}_{s}" for s in sectors]
        accounts += [f"c_{r}_{s}" for s in sectors]
    for r in regions:
        accounts += [f"CAP_{r}", f"LAB_{r}", f"HOH_{r}"]
    return accounts


# Benchmark flows (money units) for the 2-region × 2-sector instance. Chosen so both regions and
# the global economy balance, with genuine two-way trade in both goods.
#
# Per region: domestic sales of each good, and bilateral exports o→d per good.
_DOMESTIC_SALES = {
    ("N", "BRD"): 70.0,
    ("N", "MIL"): 100.0,
    ("S", "BRD"): 90.0,
    ("S", "MIL"): 60.0,
}
# Bilateral exports: (origin, dest, sector) → value. Two-way trade in both goods.
_EXPORTS = {
    ("N", "S", "BRD"): 18.0,
    ("N", "S", "MIL"): 12.0,
    ("S", "N", "BRD"): 14.0,
    ("S", "N", "MIL"): 16.0,
}
# Intermediate use within a region: (region, commodity, activity) → value.
_INTERMEDIATE = {
    ("N", "MIL", "BRD"): 20.0,
    ("N", "BRD", "MIL"): 12.0,
    ("S", "MIL", "BRD"): 22.0,
    ("S", "BRD", "MIL"): 10.0,
}


def toy_multi_sam() -> SAM:
    """The hand-checkable 2-region × 2-sector multi-region benchmark SAM (globally balanced)."""
    regions, sectors = REGIONS, SECTORS
    accounts = _acc(regions, sectors)
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)

    # Activity → own commodity (domestic sales) and → other regions' commodities (exports).
    for r in regions:
        for s in sectors:
            m.loc[f"a_{r}_{s}", f"c_{r}_{s}"] = _DOMESTIC_SALES[(r, s)]
    for (o, d, s), v in _EXPORTS.items():
        # o's activity sells commodity s to d's composite: payment c_<d>_<s> → a_<o>_<s>.
        m.loc[f"a_{o}_{s}", f"c_{d}_{s}"] = v

    # Intermediates: composite commodity (row) bought by activity (col), within a region.
    for (r, com, act), v in _INTERMEDIATE.items():
        m.loc[f"c_{r}_{com}", f"a_{r}_{act}"] = v

    # Value added = activity output (domestic + all exports) − intermediate purchases, split 50/50.
    for r in regions:
        for s in sectors:
            output = _DOMESTIC_SALES[(r, s)] + sum(
                _EXPORTS.get((r, d, s), 0.0) for d in regions if d != r
            )
            intermediates = sum(m.loc[f"c_{r}_{c}", f"a_{r}_{s}"] for c in sectors)
            va = output - intermediates
            m.loc[f"CAP_{r}", f"a_{r}_{s}"] = va / 2.0
            m.loc[f"LAB_{r}", f"a_{r}_{s}"] = va / 2.0

    # Household final demand = commodity supply (col total) − intermediate uses (row total), per
    # region-commodity. (Commodity supply = domestic sales + imports into that composite.)
    for r in regions:
        for s in sectors:
            com = f"c_{r}_{s}"
            supply = m[com].sum()  # payments INTO the composite = domestic sales + imports
            uses = m.loc[com].sum()  # payments the composite makes so far = intermediates
            m.loc[com, f"HOH_{r}"] = supply - uses

    # Factor income to each region's household.
    for r in regions:
        m.loc[f"HOH_{r}", f"CAP_{r}"] = m.loc[f"CAP_{r}", :].sum()
        m.loc[f"HOH_{r}", f"LAB_{r}"] = m.loc[f"LAB_{r}", :].sum()

    # Current-account closure: each region's net imports (imports − exports, valued at benchmark
    # prices) are financed by a bilateral capital transfer. In a closed global economy Σ current
    # accounts = 0, so region d's deficit is a transfer from the surplus region's household.
    _add_capital_transfers(m, regions, sectors)

    prov = Provenance(
        source="toy (hand-built)",
        source_version="multi-2region-2sector-v1",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes="Globally-balanced 2-region × 2-sector multi-region SAM (bilateral Armington/CET).",
    )
    return SAM(provenance=prov, accounts=accounts, matrix=m)


# Per-region government tax + savings, mirroring the closed/open toy_gov SAMs (a direct tax the
# government spends on that region's composites, and household savings the SAVINV account spends on
# gross capital formation). Kept modest so both households still have positive consumption of both
# goods after the tax + savings deduction.
_GOV_TAX = {"N": 16.0, "S": 12.0}  # HOH_r → GOV_r
_GOV_SPEND = {  # GOV_r → c_r_s
    ("N", "BRD"): 9.0,
    ("N", "MIL"): 7.0,
    ("S", "BRD"): 7.0,
    ("S", "MIL"): 5.0,
}
_SAVINGS = {"N": 14.0, "S": 10.0}  # HOH_r → SAVINV_r
_INV_SPEND = {  # SAVINV_r → c_r_s (gross capital formation)
    ("N", "BRD"): 8.0,
    ("N", "MIL"): 6.0,
    ("S", "BRD"): 6.0,
    ("S", "MIL"): 4.0,
}


def toy_multi_gov_sam() -> SAM:
    """The multi-region toy SAM extended with a per-region **government** (``GOV_<r>``) and
    **savings-investment** (``SAVINV_<r>``) account, globally balanced — the dynamic-capable
    multi-region variant (Phase 7.1).

    Structure: the same bilateral Armington/CET economy as :func:`toy_multi_sam`, plus per region an
    imputed direct tax the government spends on that region's composites, and household savings the
    savings-investment account spends on gross capital formation. Two differences from the plain
    multi SAM matter:

    - the current-account (foreign-savings) settlement routes between the **SAVINV accounts**, not
      the households — with SAVINV accounts present, ``calibrate_multi`` requires foreign savings to
      finance investment, not consumption, and rejects a cross-region HOH transfer;
    - each household's consumption falls by exactly (its tax + its savings + any net foreign lending
      it now routes through SAVINV) so every account still balances.

    This gives the multi-region CGE a per-region savings-investment account, so it reports a
    per-region benchmark stock–flow bridge (``capital_dynamics`` with one K per region) — the input
    the recursive-dynamic wrapper's per-region capital path needs.
    """
    regions, sectors = REGIONS, SECTORS
    base_accounts = _acc(regions, sectors)
    accounts = base_accounts + [f"GOV_{r}" for r in regions] + [f"SAVINV_{r}" for r in regions]
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)

    # Rebuild the base economy WITHOUT the HOH↔HOH current-account transfer (that routing is
    # forbidden once SAVINV accounts exist); the SAVINV-routed transfer is added below.
    for r in regions:
        for s in sectors:
            m.loc[f"a_{r}_{s}", f"c_{r}_{s}"] = _DOMESTIC_SALES[(r, s)]
    for (o, d, s), v in _EXPORTS.items():
        m.loc[f"a_{o}_{s}", f"c_{d}_{s}"] = v
    for (r, com, act), v in _INTERMEDIATE.items():
        m.loc[f"c_{r}_{com}", f"a_{r}_{act}"] = v
    for r in regions:
        for s in sectors:
            output = _DOMESTIC_SALES[(r, s)] + sum(
                _EXPORTS.get((r, d, s), 0.0) for d in regions if d != r
            )
            intermediates = sum(m.loc[f"c_{r}_{c}", f"a_{r}_{s}"] for c in sectors)
            va = output - intermediates
            m.loc[f"CAP_{r}", f"a_{r}_{s}"] = va / 2.0
            m.loc[f"LAB_{r}", f"a_{r}_{s}"] = va / 2.0
    for r in regions:
        m.loc[f"HOH_{r}", f"CAP_{r}"] = m.loc[f"CAP_{r}", :].sum()
        m.loc[f"HOH_{r}", f"LAB_{r}"] = m.loc[f"LAB_{r}", :].sum()

    # Government: per-region direct tax spent on that region's composites.
    for r in regions:
        m.loc[f"GOV_{r}", f"HOH_{r}"] = _GOV_TAX[r]
    for (r, s), v in _GOV_SPEND.items():
        m.loc[f"c_{r}_{s}", f"GOV_{r}"] = v
    # Savings-investment: per-region household savings spent on gross capital formation.
    for r in regions:
        m.loc[f"SAVINV_{r}", f"HOH_{r}"] = _SAVINGS[r]
    for (r, s), v in _INV_SPEND.items():
        m.loc[f"c_{r}_{s}", f"SAVINV_{r}"] = v
    # Current-account settlement routed between the SAVINV accounts (foreign savings finance
    # investment, not consumption).
    _add_capital_transfers(m, regions, sectors, via="SAVINV")

    # Close each region's household: consumption of its composites = income − (tax + savings). Split
    # across the two goods in the same proportion the plain multi SAM used (so both stay positive).
    for r in regions:
        income = m.loc[f"HOH_{r}", :].sum()
        outlays_so_far = m[f"HOH_{r}"].sum()  # tax + savings already booked as HOH payments
        to_consume = income - outlays_so_far
        # Proportional split from the no-gov benchmark final demand (recomputed on this matrix).
        weights = {}
        for s in sectors:
            com = f"c_{r}_{s}"
            supply = m[com].sum()
            uses = m.loc[com].sum()  # intermediates + gov + investment demand booked so far
            weights[s] = supply - uses  # residual capacity for household demand
        wsum = sum(weights.values())
        for s in sectors:
            m.loc[f"c_{r}_{s}", f"HOH_{r}"] = to_consume * (weights[s] / wsum)

    prov = Provenance(
        source="toy (hand-built)",
        source_version="multi-gov-savinv-2region-v1",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes=(
            "Globally-balanced 2-region × 2-sector multi-region SAM with per-region government "
            "(GOV_<r>) and savings-investment (SAVINV_<r>) accounts — the dynamic-capable "
            "multi-region variant for Phase 7.1. Current-account settlement routes between the "
            "SAVINV accounts."
        ),
    )
    return SAM(provenance=prov, accounts=accounts, matrix=m)


SPARSE_REGIONS = ["N", "S", "E"]  # North, South, East
SPARSE_SECTORS = ["BRD", "MIL"]

# Domestic sales, per (region, sector).
_SPARSE_DOMESTIC = {(r, s): 80.0 for r in SPARSE_REGIONS for s in SPARSE_SECTORS}
# Bilateral exports: (origin, dest, sector) → value. BRD trades only between N and S — the N-E and
# S-E routes for BRD are STRUCTURALLY ZERO (no such flow in the SAM at all), while MIL trades on
# every route. This is the sparse-trade topology the review (2026-07) found rank-deficient: a route
# with zero benchmark trade still got a live price unknown with no equation to pin it.
_SPARSE_EXPORTS = {
    ("N", "S", "BRD"): 15.0,
    ("S", "N", "BRD"): 12.0,
    ("N", "S", "MIL"): 10.0,
    ("S", "N", "MIL"): 8.0,
    ("N", "E", "MIL"): 9.0,
    ("E", "N", "MIL"): 7.0,
    ("S", "E", "MIL"): 6.0,
    ("E", "S", "MIL"): 5.0,
}
_SPARSE_INTERMEDIATE = {(r, "MIL", "BRD"): 15.0 for r in SPARSE_REGIONS} | {
    (r, "BRD", "MIL"): 10.0 for r in SPARSE_REGIONS
}


def toy_multi_sparse_sam() -> SAM:
    """A 3-region × 2-sector multi-region SAM with a STRUCTURALLY ZERO trade route (BRD does not
    trade on the N↔E or S↔E routes at all), used to pin the fix for the sparse-topology rank
    deficiency the 2026-07 review found: an inactive route must get no price unknown and no
    clearing residual (`MultiCalibratedModel.active_routes`), not a live-but-unpinned one."""
    regions, sectors = SPARSE_REGIONS, SPARSE_SECTORS
    accounts = _acc(regions, sectors)
    m = pd.DataFrame(0.0, index=accounts, columns=accounts)

    for r in regions:
        for s in sectors:
            m.loc[f"a_{r}_{s}", f"c_{r}_{s}"] = _SPARSE_DOMESTIC[(r, s)]
    for (o, d, s), v in _SPARSE_EXPORTS.items():
        m.loc[f"a_{o}_{s}", f"c_{d}_{s}"] = v
    for (r, com, act), v in _SPARSE_INTERMEDIATE.items():
        m.loc[f"c_{r}_{com}", f"a_{r}_{act}"] = v

    for r in regions:
        for s in sectors:
            output = _SPARSE_DOMESTIC[(r, s)] + sum(
                _SPARSE_EXPORTS.get((r, d, s), 0.0) for d in regions if d != r
            )
            intermediates = sum(m.loc[f"c_{r}_{c}", f"a_{r}_{s}"] for c in sectors)
            va = output - intermediates
            m.loc[f"CAP_{r}", f"a_{r}_{s}"] = va / 2.0
            m.loc[f"LAB_{r}", f"a_{r}_{s}"] = va / 2.0

    for r in regions:
        for s in sectors:
            com = f"c_{r}_{s}"
            supply = m[com].sum()
            uses = m.loc[com].sum()
            m.loc[com, f"HOH_{r}"] = supply - uses

    for r in regions:
        m.loc[f"HOH_{r}", f"CAP_{r}"] = m.loc[f"CAP_{r}", :].sum()
        m.loc[f"HOH_{r}", f"LAB_{r}"] = m.loc[f"LAB_{r}", :].sum()

    _add_capital_transfers(m, regions, sectors)

    prov = Provenance(
        source="toy (hand-built)",
        source_version="multi-3region-2sector-sparse-v1",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes=(
            "Globally-balanced 3-region × 2-sector multi-region SAM with a structurally zero "
            "trade route (BRD does not trade between N-E or S-E) — pins the sparse-topology "
            "rank-deficiency fix."
        ),
    )
    return SAM(provenance=prov, accounts=accounts, matrix=m)


def _add_capital_transfers(
    m: pd.DataFrame, regions: list[str], sectors: list[str], *, via: str = "HOH"
) -> None:
    """Close each region's current account with a bilateral capital transfer so both the per-region
    and global accounts balance. Region r's current account (exports − imports) is its net foreign
    savings; the sum over regions is zero, so deficits are financed by surpluses.

    ``via`` selects the settling account family: ``"HOH"`` (household↔household — the pre-5d.2
    routing, when there is no savings-investment account) or ``"SAVINV"`` (savings-investment
    account↔savings-investment account — required once ``SAVINV_<r>`` accounts exist, since with
    them foreign savings must finance investment, not consumption: ``calibrate_multi`` rejects a
    cross-region HOH transfer in that case)."""
    ca = {}
    for r in regions:
        exports = sum(m.loc[f"a_{r}_{s}", f"c_{d}_{s}"] for d in regions if d != r for s in sectors)
        imports = sum(m.loc[f"a_{o}_{s}", f"c_{r}_{s}"] for o in regions if o != r for s in sectors)
        ca[r] = exports - imports  # >0 ⇒ surplus (net lender)
    # For 2 regions the transfer is unambiguous: the surplus region lends its surplus to the deficit
    # region (deficit ← surplus). Generalises via any settlement of a zero-sum vector; a
    # proportional split is used for R>2. The settling account is HOH_<r> or SAVINV_<r> per ``via``.
    total_surplus = sum(v for v in ca.values() if v > 0)
    if total_surplus <= 0:
        return  # balanced trade — no transfer needed
    for lender in regions:
        if ca[lender] <= 0:
            continue
        for borrower in regions:
            if ca[borrower] >= 0:
                continue
            # lender finances a share of borrower's deficit proportional to lender's surplus.
            amount = (-ca[borrower]) * (ca[lender] / total_surplus)
            m.loc[f"{via}_{borrower}", f"{via}_{lender}"] += amount
