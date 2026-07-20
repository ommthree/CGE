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


def _add_capital_transfers(m: pd.DataFrame, regions: list[str], sectors: list[str]) -> None:
    """Close each region's current account with a household↔household capital transfer so both the
    per-region and global accounts balance. Region r's current account (exports − imports) is its
    net foreign savings; the sum over regions is zero, so deficits are financed by surpluses."""
    ca = {}
    for r in regions:
        exports = sum(m.loc[f"a_{r}_{s}", f"c_{d}_{s}"] for d in regions if d != r for s in sectors)
        imports = sum(m.loc[f"a_{o}_{s}", f"c_{r}_{s}"] for o in regions if o != r for s in sectors)
        ca[r] = exports - imports  # >0 ⇒ surplus (net lender)
    # For 2 regions the transfer is unambiguous: the surplus region lends its surplus to the deficit
    # region's household (HOH_deficit ← HOH_surplus). Generalises via any settlement of a zero-sum
    # vector; a proportional split is used for R>2.
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
            m.loc[f"HOH_{borrower}", f"HOH_{lender}"] += amount
