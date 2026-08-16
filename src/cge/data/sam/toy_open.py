"""A hand-checkable open-economy toy SAM (Phase 5 — Armington/CET).

Extends the closed 2-sector pilot to a **small open economy** with a rest-of-world (ROW) account,
in the standard Armington/CET structure [Hosoe2010, ch. 7]. Unlike the closed pilot it keeps
**separate activity and commodity accounts**, because Armington needs to distinguish a commodity's
*domestic* and *imported* varieties:

- ``a_BRD``, ``a_MIL`` — **activities**: produce domestic output, sold to the commodity market and
  (as exports) to ROW.
- ``c_BRD``, ``c_MIL`` — **commodities**: Armington composites of domestic output + imports, bought
  by intermediates and the household.
- ``CAP``, ``LAB`` — factors; ``HOH`` — household; ``ROW`` — rest of world.

**Trade** (benchmark): each activity exports some output to ROW; each commodity imports some from
ROW. The aggregate trade account is balanced (imports = exports = 30, i.e. zero foreign savings);
per-commodity trade is *not* balanced (BRD is a net exporter, MIL a net importer) — realistic.

**Convention** (as in ``toy.py``): ``M[r, c]`` is a payment from account ``c`` to account ``r``;
the matrix is balanced (row sum = column sum per account).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from cge.contracts.data_objects import SAM, Provenance

SECTORS = ["BRD", "MIL"]
ACTIVITIES = [f"a_{s}" for s in SECTORS]
COMMODITIES = [f"c_{s}" for s in SECTORS]
FACTORS = ["CAP", "LAB"]
ROW = "ROW"
HOUSEHOLD = "HOH"
ACCOUNTS = ACTIVITIES + COMMODITIES + FACTORS + [HOUSEHOLD, ROW]

# Benchmark trade & production (money units). Domestic sales D, exports E, imports Mimp per good.
_DOMESTIC_SALES = {"BRD": 80.0, "MIL": 110.0}  # activity → home commodity market
_EXPORTS = {"BRD": 20.0, "MIL": 10.0}  # activity → ROW
_IMPORTS = {"BRD": 12.0, "MIL": 18.0}  # ROW → commodity
# Intermediate use: activity (col) buys commodity (row).
_INTERMEDIATE = {("c_MIL", "a_BRD"): 24.0, ("c_BRD", "a_MIL"): 15.0}


def toy_open_sam() -> SAM:
    """The hand-checkable open-economy benchmark SAM (balanced; aggregate trade balanced)."""
    m = pd.DataFrame(0.0, index=ACCOUNTS, columns=ACCOUNTS)

    for s in SECTORS:
        # Activity sells domestic output to its commodity and exports to ROW.
        m.loc[f"a_{s}", f"c_{s}"] = _DOMESTIC_SALES[s]
        m.loc[f"a_{s}", ROW] = _EXPORTS[s]
        # Commodity imports from ROW.
        m.loc[ROW, f"c_{s}"] = _IMPORTS[s]
    # Intermediates (commodity row bought by activity col).
    for (com, act), v in _INTERMEDIATE.items():
        m.loc[com, act] = v
    # Value added = activity output (domestic + exports) − intermediate purchases, split 50/50.
    for s in SECTORS:
        output = _DOMESTIC_SALES[s] + _EXPORTS[s]
        intermediates = sum(m.loc[c, f"a_{s}"] for c in COMMODITIES)
        va = output - intermediates
        m.loc["CAP", f"a_{s}"] = va / 2.0
        m.loc["LAB", f"a_{s}"] = va / 2.0
    # Commodity final demand to the household = commodity supply (col) − intermediate uses (row).
    for s in SECTORS:
        supply = m[f"c_{s}"].sum()
        uses = m.loc[f"c_{s}"].sum()
        m.loc[f"c_{s}", HOUSEHOLD] = supply - uses
    # Factor income to the household.
    m.loc[HOUSEHOLD, "CAP"] = m.loc["CAP", ACTIVITIES].sum()
    m.loc[HOUSEHOLD, "LAB"] = m.loc["LAB", ACTIVITIES].sum()

    prov = Provenance(
        source="toy (hand-built)",
        source_version="open-2sector-v1",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes="Exactly-balanced open-economy 2-sector SAM (Armington/CET) for the CGE.",
    )
    return SAM(provenance=prov, accounts=ACCOUNTS, matrix=m)


def toy_open_gov_sam() -> SAM:
    """The open-economy toy SAM extended with a **government** (``GOV``) and a **savings-
    investment** (``SAVINV``) account, exactly balanced — the dynamic-capable open variant (7.1).

    Same structure as :func:`toy_open_sam` plus, mirroring the closed ``toy_gov_sam``: an imputed
    direct tax from the household spent by the government on commodities, and household savings
    spent by the savings-investment account on gross capital formation. Government and investment
    demand go to the **commodity** accounts (``c_BRD``/``c_MIL``, the Armington composites), as with
    all final demand here. Household consumption falls by exactly (tax + savings) so every account
    still balances. This gives the open CGE a savings-investment account, so it has the benchmark
    stock-flow bridge (``capital_dynamics.available``) the recursive-dynamic wrapper steps forward.
    """
    base = toy_open_sam()
    acc = list(base.accounts) + ["GOV", "SAVINV"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc[base.accounts, base.accounts] = base.matrix
    # Government: an imputed direct tax from the household, spent on commodities.
    m.loc["GOV", HOUSEHOLD] = 18.1
    m.loc["c_BRD", "GOV"] = 10.0
    m.loc["c_MIL", "GOV"] = 8.1
    # Savings-investment: household savings spent on gross capital formation (commodities).
    m.loc["SAVINV", HOUSEHOLD] = 16.29
    m.loc["c_BRD", "SAVINV"] = 9.0
    m.loc["c_MIL", "SAVINV"] = 7.29
    # The household consumes less by exactly (tax + savings) so every account still balances.
    m.loc["c_BRD", HOUSEHOLD] -= 19.0
    m.loc["c_MIL", HOUSEHOLD] -= 15.39
    prov = Provenance(
        source="toy (hand-built)",
        source_version="open-gov-savinv-v1",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes=(
            "Exactly-balanced open-economy 2-sector SAM with a government (GOV) and savings-"
            "investment (SAVINV) account — the dynamic-capable open variant for Phase 7.1."
        ),
    )
    return SAM(provenance=prov, accounts=acc, matrix=m)
