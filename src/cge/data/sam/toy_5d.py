"""Hand-checkable toy SAMs for the Phase 5d macro-closure features (government/fiscal account,
savings-investment, and the KL-E-M energy nest).

These are the calibration targets the user guide's Step 8 (macro closures) runs against, so the
government/investment closures and the energy nest can be demonstrated end-to-end **without a data
build** — the same role ``toy_sam`` / ``toy_open_sam`` / ``toy_multi_sam`` play for Steps 7a-7f.

**Convention** (as in ``toy.py``): row = receipts, column = payments; ``M[r, c]`` is a payment
*from* account ``c`` *to* account ``r``; the matrix is exactly **balanced** (row sum = column sum
for every account).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from cge.contracts.data_objects import SAM, Provenance
from cge.data.sam.toy import toy_sam

# Energy-nest toy: sector labels (two energy commodities + one manufacturer).
ENERGY_SECTORS = ["DIRTY", "CLEAN", "MFG"]


def toy_gov_sam() -> SAM:
    """The 2-sector closed toy SAM extended with a **government** (``GOV``) and a **savings-
    investment** (``SAVINV``) account, exactly balanced.

    The government levies a benchmark direct tax on the household (``GOV`` receives 18.1 from
    ``HOH``) and spends it on the two commodities (10.0 on ``BRD``, 8.1 on ``MIL``); the household
    saves 16.29, which the savings-investment account spends on investment goods (9.0 ``BRD``, 7.29
    ``MIL``). Household consumption is reduced by the tax + savings so the SAM stays balanced. This
    is the calibration target for the fiscal (``gov_closure``) and investment (``inv_closure``)
    closures — the same structure the validation suite calibrates and replicates.
    """
    base = toy_sam()
    acc = list(base.accounts) + ["GOV", "SAVINV"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc[base.accounts, base.accounts] = base.matrix
    # Government: an imputed direct tax from the household, spent on commodities.
    m.loc["GOV", "HOH"] = 18.1
    m.loc["BRD", "GOV"] = 10.0
    m.loc["MIL", "GOV"] = 8.1
    # Savings-investment: household savings spent on gross capital formation.
    m.loc["SAVINV", "HOH"] = 16.29
    m.loc["BRD", "SAVINV"] = 9.0
    m.loc["MIL", "SAVINV"] = 7.29
    # The household consumes less by exactly (tax + savings) so every account still balances.
    m.loc["BRD", "HOH"] -= 19.0
    m.loc["MIL", "HOH"] -= 15.39
    prov = Provenance(
        source="toy (hand-built)",
        source_version="5d-gov-savinv-v1",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes=(
            "Exactly-balanced closed SAM with a government (GOV) and savings-investment (SAVINV) "
            "account — the calibration target for the Phase 5d fiscal and investment closures."
        ),
    )
    return SAM(provenance=prov, accounts=acc, matrix=m)


def toy_energy_sam() -> SAM:
    """A 3-sector closed SAM with **two energy commodities** — ``DIRTY`` (fossil) and ``CLEAN``
    (electricity) — both consumed by ``MFG`` (manufacturing), where electricity generation itself
    uses some fossil. Exactly balanced.

    This is the calibration target for the **KL-E-M energy nest**: with ``energy_sectors =
    ['DIRTY', 'CLEAN']`` a carbon price on the fossil commodity lets the model substitute *within*
    the energy bundle (fossil → electricity), not just between energy and other inputs.
    """
    acc = [*ENERGY_SECTORS, "CAP", "LAB", "HOH"]
    m = pd.DataFrame(0.0, index=acc, columns=acc)
    m.loc["DIRTY", "MFG"] = 15.0  # MFG buys fossil energy
    m.loc["CLEAN", "MFG"] = 10.0  # MFG buys electricity
    m.loc["DIRTY", "CLEAN"] = 5.0  # electricity generation uses some fossil
    m.loc["CAP", "DIRTY"] = 10.0
    m.loc["LAB", "DIRTY"] = 10.0
    m.loc["CAP", "CLEAN"] = 12.0
    m.loc["LAB", "CLEAN"] = 13.0
    m.loc["CAP", "MFG"] = 25.0
    m.loc["LAB", "MFG"] = 25.0
    # Final demand = each commodity's supply (column total, what it pays out for inputs) minus its
    # intermediate use (row total so far) — closes the goods markets and balances the SAM.
    for s in ENERGY_SECTORS:
        m.loc[s, "HOH"] = m[s].sum() - m.loc[s].sum()
    # Factor income all flows to the household (closes the income loop).
    m.loc["HOH", "CAP"] = m.loc["CAP", ENERGY_SECTORS].sum()
    m.loc["HOH", "LAB"] = m.loc["LAB", ENERGY_SECTORS].sum()
    prov = Provenance(
        source="toy (hand-built)",
        source_version="5d-energy-nest-v1",
        licence="n/a",
        reference_year=0,
        retrieved=date.today().isoformat(),
        notes=(
            "Exactly-balanced 3-sector closed SAM with two energy commodities (DIRTY/CLEAN) — the "
            "calibration target for the Phase 5d KL-E-M energy nest."
        ),
    )
    return SAM(provenance=prov, accounts=acc, matrix=m)
