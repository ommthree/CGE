#!/usr/bin/env python
"""Build low/central/high structural-trajectory variants (pipeline step 1e).

    python scripts/build_uncertainty_sets.py          # write trajectories_v1_{low,high}.json
    python scripts/build_uncertainty_sets.py --check   # non-zero if the variants are stale

review-9 non-blocking ask 1e. The committed ``trajectories_v1.json`` is the CENTRAL path. This emits
two sibling artifacts — ``trajectories_v1_low.json`` and ``trajectories_v1_high.json`` — so a
consulting result can report an INTERVAL (sweep {low, central, high}) rather than a single point.

How the bands are set (honest about basis):
  * EU KLEMS SECTOR drivers (``mfp``, ``sector_productivity``): an EMPIRICAL band from the KLEMS
    alternative trend windows — central = 2017–2021 (5y, the committed value), and the low/high
    envelope = the min/max across {2017–21, 2015–21, 2010–21}. This directly captures the window
    sensitivity review-9 flagged (the COVID window can reverse the goods-vs-services ordering).
  * All OTHER drivers (population, participation, productivity, __all__ emissions): a
    CONFIDENCE-TIERED relative band around the central rate — low→±50%, medium→±25%, high→±10% — a
    transparent, documented sensitivity multiplier (not an empirical CI), applied to each rate.
    "low" trajectory = the less-favourable growth rate (see per-driver sign handling below).

The variants are assembled by :func:`scripts.build_structural_trajectories._assemble`, so they share
the central schema and provenance shape; only the numeric knot rates differ. A ``--check`` gate
(wired into CI) keeps them reproducible from the central digest + raw sources.
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
_HIGH = _STRUCT / "trajectories_v1_high.json"

# Confidence → relative half-width of the sensitivity band (fraction of the central rate).
_TIER_BAND = {"low": 0.50, "medium": 0.25, "high": 0.10}
# EU KLEMS alternative trend windows (trend_window years) for the empirical sector band.
_KLEMS_WINDOWS = (5, 7, 12)  # 2017–21 (central), 2015–21, 2010–21


def _klems_envelope() -> dict:
    """min/max of the EU KLEMS mfp + sector_productivity rates across the alternative windows, per
    archetype. Returns ``{driver: {archetype: (lo, hi)}}``."""
    from scripts.extract_structural_sources import _load_map, extract_euklems, load_euklems_workbook

    imap = _load_map("industry_to_archetype.csv", "isic_section")
    klems = _RAW / "euklems_2023_growth_accounts.xlsx"
    per_window: list[dict] = []
    for w in _KLEMS_WINDOWS:
        with contextlib.redirect_stdout(io.StringIO()):
            out = extract_euklems(load_euklems_workbook(klems, trend_window=w), imap)
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
    """The low/high variant of a central annual ``rate`` with relative half-width ``half``. 'low' =
    the less-favourable path: a POSITIVE growth rate shrinks (×(1−half)); a NEGATIVE rate (e.g.
    emissions-intensity decline, population decline) becomes less negative (also ×(1−half) in
    magnitude toward zero). 'high' is the mirror. So 'low' is uniformly the weaker-growth / slower-
    decarbonisation world and 'high' the stronger one, keeping the sweep monotone."""
    factor = (1.0 - half) if side == "low" else (1.0 + half)
    return round(rate * factor, 4)


def _apply_bands(digest: dict, side: str, env: dict) -> dict:
    out = copy.deepcopy(digest)
    # Region drivers: confidence-tiered relative band.
    for _driver, table in out["region_drivers"].items():
        for _key, entry in table.items():
            half = _TIER_BAND.get(entry.get("confidence", "medium"), 0.25)
            entry["knots"] = {y: _band_rate(r, half, side) for y, r in entry["knots"].items()}
    # Sector drivers: EU KLEMS mfp/sector_productivity use the EMPIRICAL window envelope; others
    # (emissions_intensity) the confidence-tiered band.
    for driver, table in out["sector_drivers"].items():
        for key, entry in table.items():
            if driver in env and key in env[driver]:
                lo, hi = env[driver][key]
                chosen = lo if side == "low" else hi
                entry["knots"] = {y: round(chosen, 4) for y in entry["knots"]}
            else:
                half = _TIER_BAND.get(entry.get("confidence", "medium"), 0.25)
                entry["knots"] = {y: _band_rate(r, half, side) for y, r in entry["knots"].items()}
    # Stamp the band basis into provenance (append to notes, a first-class contract field that
    # round-trips through load_structural_trajectories, so downstream sees it in .provenance.notes).
    out["provenance"] = dict(out["provenance"])
    out["provenance"]["source_version"] = out["provenance"]["source_version"] + f"-{side}"
    uncertainty_note = (
        f" UNCERTAINTY: {side.upper()} sensitivity variant (review-9 1e 2026-09-16). EU KLEMS mfp/"
        f"sector_productivity: empirical envelope across trend windows {_KLEMS_WINDOWS} "
        f"(2017–21/2015–21/2010–21). Other drivers: confidence-tiered relative band "
        f"(low±50%/medium±25%/high±10%) on the central annual rate, 'low' = weaker growth / slower "
        f"decarbonisation. Central path = trajectories_v1.json."
    )
    out["provenance"]["notes"] = (out["provenance"].get("notes", "") or "") + uncertainty_note
    return out


def _variants() -> dict[Path, dict]:
    from scripts.build_structural_trajectories import _assemble

    digest = json.loads(_DIGEST.read_text())
    env = _klems_envelope()
    return {
        _LOW: _assemble(_apply_bands(digest, "low", env)),
        _HIGH: _assemble(_apply_bands(digest, "high", env)),
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
