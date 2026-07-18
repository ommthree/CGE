"""Macroeconomic accounting layer (roadmap Phase 4b, PE tier).

Rolls an engine's per-good price/volume responses up into the aggregates a macro reader expects
— **gross value added (GVA) per sector**, **GDP per country**, an aggregate **deflator**
(inflation), each in **nominal and real** terms — as a post-processing step over a ``ResultSet``.

This is the *indicative* PE tier: it is arithmetic on the IO engines' outputs, so it inherits
their caveats (fixed technology, Engine-2 elasticity uncertainty). The CGE (Phase 5) will emit
these as native equilibrium variables; the two are cross-checked there. See
``docs/models/macro-aggregates.md``.

Method (equation level)
-----------------------
Base-year value added per good is derived from the IO system itself — no separate value-added
table needed — as gross output minus intermediate use:

    VA_i = x_i · (1 − Σ_j A_{ji})          x = (I−A)⁻¹ y   (Leontief gross output)

The identity Σ_i VA_i = Σ_i y_i (production = expenditure GDP) holds by construction and is
checked in validation.

For a shock producing a fractional output-price change Δp_i and (if the engine emits it) a
fractional volume change Δx_i per good, per uncertainty band b:

    nominal GVA change      g^nom_i = (1+Δp_i)(1+Δx_i) − 1       (no volume ⇒ = Δp_i)
    region deflator (infl.) D_r     = Σ_{i∈r} VA_i·Δp_i / Σ_{i∈r} VA_i
    nominal GDP change      G^nom_r = Σ_{i∈r} VA_i·g^nom_i / Σ_{i∈r} VA_i
    real X change           x^real  = (1 + x^nom)/(1 + D_r) − 1   (deflate nominal by the index)

The deflator is value-added-weighted (a GDP deflator). Real = nominal deflated by it, so at zero
inflation real == nominal (a validation check).

Emitted variables (added to the ``ResultSet``; region-level rows use the sentinel sector
``__economy__``): ``gva_change``, ``gva_change_real`` (per sector); ``gdp_change``,
``gdp_change_real``, ``deflator`` (per region, sector ``__economy__``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cge.contracts.data_objects import IOSystem
from cge.contracts.provenance import content_hash
from cge.contracts.results import RESULT_COLUMNS, ResultSet

# Sentinel 'sector' for economy-wide, region-level rows (GDP, deflator). Chosen so it cannot
# collide with a real sector label and reads clearly in the long-format table.
ECONOMY_SECTOR = "__economy__"

# Base-year value added below this share of total is treated as ~zero weight, so a degenerate
# sector (all intermediates) does not create a divide-by-zero in a region with no value added.
_VA_EPS = 1e-12


def base_year_value_added(io: IOSystem) -> pd.Series:
    """Base-year value added per good (index = 'region:sector' labels), derived from the IO
    system: VA_i = x_i · (1 − Σ_j A_{ji}), with x = (I−A)⁻¹ y. Requires a productive A (checked
    by the engines before this runs). Negative VA (a non-productive column) is clipped to 0 so it
    cannot flip an aggregate weight; such a build would already fail the engines' admissibility
    guard, but we stay defensive."""
    labels = list(io.A.columns)
    A = io.A.to_numpy(dtype=float)
    y = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
    x = np.linalg.solve(np.eye(A.shape[0]) - A, y)
    va_share = 1.0 - A.sum(axis=0)  # column sum = intermediate input share per unit output
    va = np.clip(x * va_share, 0.0, None)
    return pd.Series(va, index=labels, name="value_added")


def _rec(variable: str, sector: str, region: str, year: int, scenario: str, value: float) -> dict:
    return {
        "variable": variable,
        "sector": sector,
        "region": region,
        "year": year,
        "scenario": scenario,
        "value": float(value),
    }


def macro_records(result: ResultSet, io: IOSystem) -> list[dict]:
    """Compute the macro-aggregate rows (GVA/GDP/deflator, nominal + real) for ``result`` given
    the base-year ``io``. Returns long-format records ready to append to the ResultSet. Uses the
    price response always, and the volume response when the engine emitted one (else volumes are
    held fixed, the Engine-1 case). Each uncertainty band is aggregated independently."""
    df = result.data
    price = df[df["variable"] == "price_change"]
    if price.empty:
        return []  # nothing to roll up (e.g. a non-price engine)
    volume = df[df["variable"] == "volume_change"]
    has_volume = not volume.empty

    va = base_year_value_added(io)

    def _lab(region: str, sector: str) -> str:
        return f"{region}:{sector}"

    records: list[dict] = []
    # Price bands are 'central' only (Engine 1); volume carries low/central/high. Aggregate over
    # whatever bands are present in the *volume* response when there is one, else just central.
    bands = sorted(volume["scenario"].unique()) if has_volume else ["central"]

    for year in sorted(price["year"].unique()):
        # Price is per-good, band 'central'; index by label for lookup.
        dp = {
            _lab(r.region, r.sector): float(r.value)
            for r in price[price["year"] == year].itertuples()
        }
        for band in bands:
            if has_volume:
                vb = volume[(volume["year"] == year) & (volume["scenario"] == band)]
                dx = {_lab(r.region, r.sector): float(r.value) for r in vb.itertuples()}
            else:
                dx = {}

            # Per-good nominal GVA change, and gather region weights.
            region_num_nom: dict[str, float] = {}  # Σ VA·g^nom
            region_num_defl: dict[str, float] = {}  # Σ VA·Δp
            region_den: dict[str, float] = {}  # Σ VA
            per_good_nom: dict[str, float] = {}
            for label in va.index:
                region, sector = label.split(":", 1)
                p = dp.get(label, 0.0)
                q = dx.get(label, 0.0)
                g_nom = (1.0 + p) * (1.0 + q) - 1.0  # nominal VA change (= p when q == 0)
                per_good_nom[label] = g_nom
                w = float(va.get(label, 0.0))
                region_num_nom[region] = region_num_nom.get(region, 0.0) + w * g_nom
                region_num_defl[region] = region_num_defl.get(region, 0.0) + w * p
                region_den[region] = region_den.get(region, 0.0) + w

            # Region deflator + real conversion, then emit region + per-good rows.
            region_deflator: dict[str, float] = {}
            for region, den in region_den.items():
                deflator = region_num_defl[region] / den if den > _VA_EPS else 0.0
                gdp_nom = region_num_nom[region] / den if den > _VA_EPS else 0.0
                gdp_real = (1.0 + gdp_nom) / (1.0 + deflator) - 1.0
                region_deflator[region] = deflator
                records.append(_rec("deflator", ECONOMY_SECTOR, region, year, band, deflator))
                records.append(_rec("gdp_change", ECONOMY_SECTOR, region, year, band, gdp_nom))
                records.append(
                    _rec("gdp_change_real", ECONOMY_SECTOR, region, year, band, gdp_real)
                )

            for label, g_nom in per_good_nom.items():
                region, sector = label.split(":", 1)
                d = region_deflator.get(region, 0.0)
                g_real = (1.0 + g_nom) / (1.0 + d) - 1.0
                records.append(_rec("gva_change", sector, region, year, band, g_nom))
                records.append(_rec("gva_change_real", sector, region, year, band, g_real))

    return records


# Version of the macro-accounting post-processor. Bumped when the index-number conventions or the
# emitted variables change, so a result records which accounting logic produced its macro rows.
MACRO_ACCOUNTING_VERSION = "0.2.0"


def augment_with_macro_aggregates(result: ResultSet, io: IOSystem) -> ResultSet:
    """Return a new ``ResultSet`` with GVA/GDP/deflator (nominal + real) rows appended. A no-op
    (returns the input) when the result carries no price response. Idempotent-safe: it does not
    re-add rows for variables already present.

    The post-processor stamps its own identity into the manifest (review P1): its version and a
    content hash of the **effective value-added weights** (derived from the IO system incl. final
    demand). So two runs with identical price rows but different final demand — which produce
    different macro rows — no longer share an identical manifest."""
    existing = set(result.data["variable"].unique())
    macro_vars = {"gva_change", "gva_change_real", "gdp_change", "gdp_change_real", "deflator"}
    if macro_vars & existing:
        return result  # already augmented; don't double-add
    recs = macro_records(result, io)
    if not recs:
        return result
    va = base_year_value_added(io)
    macro_stamp = {
        "version": MACRO_ACCOUNTING_VERSION,
        "index_numbers": (
            "GDP deflator = value-added-weighted mean of sector price changes (a base-weighted / "
            "Laspeyres price index on VA shares); real = nominal / (1+deflator); sector "
            "gva_change_real deflates nominal GVA by the REGION deflator (a real-income measure, "
            "NOT the sector's own production-volume change — use volume_change for that)"
        ),
        "value_added_weights_hash": content_hash({k: round(float(v), 10) for k, v in va.items()}),
    }
    manifest = result.manifest.model_copy(
        update={"assumptions": {**result.manifest.assumptions, "macro_aggregates": macro_stamp}}
    )
    add = pd.DataFrame.from_records(recs, columns=RESULT_COLUMNS)
    merged = pd.concat([result.data, add], ignore_index=True)
    return ResultSet(data=merged, manifest=manifest).validate_schema()
