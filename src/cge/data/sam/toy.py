"""A hand-checkable 2-sector toy SAM — the calibration target for the CGE pilot (Phase 5.2a).

A tiny, exactly-balanced closed-economy Social Accounting Matrix that a human can verify cell by
cell. It is the analogue of the toy IO economy used for Engines 1–2: small enough that the
calibrated CGE parameters and the benchmark-replication test can be checked by hand.

**Convention:** row = receipts (income), column = payments (expenditure); cell ``M[r, c]`` is a
payment *from* account ``c`` *to* account ``r``. The matrix is **balanced**: for every account,
row sum (total income) = column sum (total expenditure).

**Accounts** (2 sectors, 2 factors, 1 household — no government/RoW in the pilot; those enter with
the carbon tax and the open economy later):

- ``BRD``, ``MIL`` — activities/commodities (activity and commodity collapsed for the pilot).
- ``CAP``, ``LAB`` — capital and labour factors.
- ``HOH`` — the representative household.

**Structure** (all values are benchmark money units):

- sectors sell to each other (intermediates) and to the household (final demand);
- sectors pay capital and labour (value added);
- factors pay all their income to the household;
- the household spends all its income on the two commodities (closed, no savings in the pilot).

By construction GDP = Σ value added = Σ final demand = 181.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from cge.contracts.data_objects import SAM, Provenance

# Account labels, in a fixed order (activities/commodities, factors, institutions).
SECTORS = ["BRD", "MIL"]
FACTORS = ["CAP", "LAB"]
ACCOUNTS = SECTORS + FACTORS + ["HOH"]

# The balanced benchmark cells (M[row, col] = payment from col to row). See module docstring.
# Intermediate sales (row sector supplies col sector):
_INTERMEDIATE = {("BRD", "MIL"): 24.0, ("MIL", "BRD"): 15.0}
# Value added (factor row paid by sector col):
_VALUE_ADDED = {
    ("CAP", "BRD"): 42.5,
    ("LAB", "BRD"): 42.5,
    ("CAP", "MIL"): 48.0,
    ("LAB", "MIL"): 48.0,
}
# Household final demand (commodity row bought by HOH):
_FINAL_DEMAND = {("BRD", "HOH"): 76.0, ("MIL", "HOH"): 105.0}
# Factor income to the household (HOH row receives from factor col):
_FACTOR_INCOME = {("HOH", "CAP"): 90.5, ("HOH", "LAB"): 90.5}


def toy_sam() -> SAM:
    """The hand-checkable 2-sector benchmark SAM (exactly balanced)."""
    m = pd.DataFrame(0.0, index=ACCOUNTS, columns=ACCOUNTS)
    for cells in (_INTERMEDIATE, _VALUE_ADDED, _FINAL_DEMAND, _FACTOR_INCOME):
        for (r, c), v in cells.items():
            m.loc[r, c] = v
    prov = Provenance(
        source="toy (hand-built)",
        source_version="pilot-2sector-v1",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes="Exactly-balanced 2-sector closed-economy SAM for the CGE replication pilot.",
    )
    return SAM(provenance=prov, accounts=ACCOUNTS, matrix=m)
