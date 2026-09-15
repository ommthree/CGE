#!/usr/bin/env python
"""Build ADVERSE / central / FAVOURABLE structural-trajectory sensitivity cases (pipeline step 1e).

    python scripts/build_uncertainty_sets.py          # write trajectories_v1_{adverse,favourable}
    python scripts/build_uncertainty_sets.py --check   # non-zero if the cases are stale

review-9 ask 1e; RELABELLED review-10 P1#2 (2026-09-14). The committed ``trajectories_v1.json`` is
the CENTRAL path. This emits two sibling artifacts — ``trajectories_v1_adverse.json`` and
``trajectories_v1_favourable.json``.

These are DETERMINISTIC SENSITIVITY CASES (stress narratives), NOT statistical uncertainty intervals
and NOT probabilistic bounds — NGFS scenarios are not forecasts and carry no probabilities, and a
CGE's nonlinear outputs are not bounded by simultaneously perturbing every driver. Read them as "a
coherently-adverse world" vs "a coherently-favourable world" for the DRIVERS, not as an output CI.

How the cases are set (honest about basis):
  * EU KLEMS SECTOR drivers (``mfp``, ``sector_productivity``): an EMPIRICAL envelope = the
    per-archetype min/max across the alternative trend windows 2017–21 (central, COVID-affected),
    2015–19 and 2010–19 (pre-COVID comparators). Because min/max is per archetype, a single artifact
    may combine values from DIFFERENT windows and is not one coherent historical calibration.
  * All OTHER drivers (population, participation, productivity, emissions intensity): a JUDGMENTAL
    relative stress — low→±50%, medium→±25%, high→±10% by confidence tier — applied to the central
    rate. These are stress magnitudes with NO statistical-coverage interpretation.

For OUTPUT bounds, do not assume the case labels order the outputs: run all three and take the
realised per-output min/max (``cge.data.structural.structural_sensitivity_bounds`` helps). The cases
are assembled by :func:`scripts.build_structural_trajectories._assemble` (shared schema/provenance
shape; only knot rates differ). A ``--check`` gate (in CI) keeps them reproducible.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_STRUCT = _ROOT / "data" / "structural"
_DIGEST = _STRUCT / "sources" / "inputs.json"
_RAW = _STRUCT / "sources" / "raw"
_LOW = _STRUCT / "trajectories_v1_low.json"
_ADVERSE = _STRUCT / "trajectories_v1_adverse.json"
_FAVOURABLE = _STRUCT / "trajectories_v1_favourable.json"

# Confidence → relative half-width of the judgmental stress band (fraction of the central rate).
# These are DELIBERATE stress magnitudes, NOT statistical confidence intervals (review-10 P1#2).
_TIER_BAND = {"low": 0.50, "medium": 0.25, "high": 0.10}
# EU KLEMS alternative trend windows for the empirical sector band, as (trend_window, end_year).
# Includes PRE-COVID comparators (2010–19, 2015–19) as well as the COVID-affected central 2017–21
# (review-10 P1#2 — every window previously ended in 2021, giving no clean pre-COVID comparator).
_KLEMS_WINDOWS = (
    (5, 2021),  # 2017–21 (the central-artifact window, COVID-affected)
    (5, 2019),  # 2015–19 pre-COVID
    (10, 2019),  # 2010–19 pre-COVID
)


def _klems_envelope() -> dict:
    """min/max of the EU KLEMS mfp + sector_productivity rates across the alternative
    (trend_window, end_year) windows, per archetype. Returns ``{driver: {archetype: (lo, hi)}}``.
    The min/max is taken PER ARCHETYPE independently, so the adverse/favourable artifacts are the
    per-archetype envelope, NOT a single coherent historical calibration (review-10 P1#2)."""
    from scripts.extract_structural_sources import _load_map, extract_euklems, load_euklems_workbook

    imap = _load_map("industry_to_archetype.csv", "isic_section")
    klems = _RAW / "euklems_2023_growth_accounts.xlsx"
    per_window: list[dict] = []
    for w, ey in _KLEMS_WINDOWS:
        with contextlib.redirect_stdout(io.StringIO()):
            out = extract_euklems(load_euklems_workbook(klems, trend_window=w, end_year=ey), imap)
        per_window.append(out)
    env: dict = {}
    for driver in ("mfp", "sector_productivity"):
        env[driver] = {}
        archs = per_window[0][driver].keys()
        for a in archs:
            vals = [w[driver][a][2025] for w in per_window if a in w[driver]]
            env[driver][a] = (min(vals), max(vals))
    return env


def _band_rate(rate: float, half: float, side: str) -> float:
    """The adverse/favourable variant of a central annual ``rate`` with relative half-width
    ``half``. 'adverse' = the less-favourable growth path: a POSITIVE growth rate shrinks
    (×(1−half)); a NEGATIVE rate (emissions-intensity or population decline) moves toward zero
    (×(1−half) in magnitude). 'favourable' is the mirror. NOTE (review-10 P1#2): 'adverse' /
    'favourable' order the INPUT DRIVER rates by favourability — they are NOT guaranteed lower/upper
    bounds on any nonlinear CGE OUTPUT (GDP, prices, emissions). For output bounds, run all cases
    and take the realised per-output min/max (see structural_sensitivity_bounds)."""
    factor = (1.0 - half) if side == "adverse" else (1.0 + half)
    return round(rate * factor, 4)


def _apply_bands(digest: dict, side: str, env: dict) -> dict:
    out = copy.deepcopy(digest)
    # Region drivers: confidence-tiered judgmental stress band.
    for _driver, table in out["region_drivers"].items():
        for _key, entry in table.items():
            half = _TIER_BAND.get(entry.get("confidence", "medium"), 0.25)
            entry["knots"] = {y: _band_rate(r, half, side) for y, r in entry["knots"].items()}
    # Sector drivers: EU KLEMS mfp/sector_productivity use the EMPIRICAL window envelope; others
    # (emissions_intensity) the confidence-tiered stress band.
    for driver, table in out["sector_drivers"].items():
        for key, entry in table.items():
            if driver in env and key in env[driver]:
                lo, hi = env[driver][key]
                chosen = lo if side == "adverse" else hi
                entry["knots"] = {y: round(chosen, 4) for y in entry["knots"]}
            else:
                half = _TIER_BAND.get(entry.get("confidence", "medium"), 0.25)
                entry["knots"] = {y: _band_rate(r, half, side) for y, r in entry["knots"].items()}
    # Stamp the basis into provenance.notes (a first-class contract field that round-trips).
    out["provenance"] = dict(out["provenance"])
    out["provenance"]["source_version"] = out["provenance"]["source_version"] + f"-{side}"
    uncertainty_note = (
        f" SENSITIVITY: {side.upper()} deterministic stress case (review-10 P1#2 2026-09-14) — NOT "
        f"a statistical bound or a probabilistic interval (NGFS scenarios are not forecasts and "
        f"carry no probabilities). EU KLEMS mfp/sector_productivity: per-archetype envelope across "
        f"trend windows 2017–21 (central) / 2015–19 / 2010–19; these mix windows and are "
        f"not one coherent calibration. Other drivers: judgmental relative stress "
        f"(low±50/medium±25/high±10) "
        f"on the central rate ordered by favourability. 'adverse'/'favourable' order the INPUT "
        f"drivers, NOT any CGE output — for output bounds run all cases and take the realised "
        f"per-output min/max. Central path = trajectories_v1.json."
    )
    out["provenance"]["notes"] = (out["provenance"].get("notes", "") or "") + uncertainty_note
    return out


def _variants() -> dict[Path, dict]:
    from scripts.build_structural_trajectories import _assemble

    digest = json.loads(_DIGEST.read_text())
    env = _klems_envelope()
    return {
        _ADVERSE: _assemble(_apply_bands(digest, "adverse", env)),
        _FAVOURABLE: _assemble(_apply_bands(digest, "favourable", env)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="non-zero if variants are stale")
    args = ap.parse_args()
    if not (_RAW / "euklems_2023_growth_accounts.xlsx").exists():
        print(
            "EU KLEMS raw file absent — cannot compute the empirical band; keeping committed sets."
        )
        return 0
    rc = 0
    for path, artifact in _variants().items():
        rendered = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != rendered:
                print(f"{path.name} is OUT OF DATE — re-run without --check + commit.")
                rc = 1
            else:
                print(f"{path.name} matches the central digest.")
        else:
            path.write_text(rendered)
            print(f"Wrote {path}.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
