#!/usr/bin/env python
"""Build the vendored structural-trajectory artifact from the committed source digest.

    python scripts/build_structural_trajectories.py           # (re)write trajectories_v1.json
    python scripts/build_structural_trajectories.py --check    # non-zero if the artifact is stale

Pipeline step 1a (see ``docs/structural-data-pipeline-plan.md``). Reads the structured source digest
``data/structural/sources/inputs.json`` — per-driver, per-key knot rates each carrying a source
citation and confidence — and assembles ``data/structural/trajectories_v1.json`` (the shape the
loader + wrapper consume). This makes the artifact REPRODUCIBLE from committed inputs rather than
hand-edited, and locks the pipeline shape so real per-country/industry extractions (PWT 10.01, UN
WPP 2024, EU KLEMS 2023, NGFS) drop into the digest later without touching this script or schema.

Today the digest holds the same illustrative headline figures the artifact previously carried, so
this build is a NO-OP on the numbers — it introduces the reproducible path, not new data. CI runs
``--check`` so the committed artifact can never drift from the digest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DIGEST = _ROOT / "data" / "structural" / "sources" / "inputs.json"
_ARTIFACT = _ROOT / "data" / "structural" / "trajectories_v1.json"


def _assemble(digest: dict) -> dict:
    """Assemble the artifact dict from the digest, in the artifact's canonical key order:
    provenance, rates, sector_rates, source_labour_shares, sources, confidence."""
    rates: dict = {}
    sector_rates: dict = {}
    sources: dict = {}
    confidence: dict = {}

    def _emit(driver: str, entry: dict, rate_table: dict) -> None:
        rate_table[driver] = {}
        for key, spec in entry.items():
            rate_table[driver][key] = dict(spec["knots"])
            sources[f"{driver}:{key}"] = spec["source"]
            confidence[f"{driver}:{key}"] = spec["confidence"]

    for driver, entry in digest["region_drivers"].items():
        _emit(driver, entry, rates)
    for driver, entry in digest["sector_drivers"].items():
        _emit(driver, entry, sector_rates)

    # source_labour_shares carry value-level provenance in the digest ({key: {value, source,
    # confidence}}); emit the flat {key: value} the contract stores plus source_labour_share:{key}
    # metadata (the contract requires it, review P2 2026-09-06).
    source_labour_shares: dict = {}
    for key, spec in digest["source_labour_shares"].items():
        source_labour_shares[key] = spec["value"]
        sources[f"source_labour_share:{key}"] = spec["source"]
        confidence[f"source_labour_share:{key}"] = spec["confidence"]

    return {
        "provenance": digest["provenance"],
        "rates": rates,
        "sector_rates": sector_rates,
        "source_labour_shares": source_labour_shares,
        "sources": sources,
        "confidence": confidence,
    }


def _render(artifact: dict) -> str:
    """Match the committed artifact's formatting: 2-space indent, trailing newline. The nested
    per-year rate dicts render compactly (one line per key) via a post-pass, matching the hand-
    written file so ``--check`` compares like-for-like."""
    text = json.dumps(artifact, indent=2, ensure_ascii=False)
    return text + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="non-zero if the artifact is stale")
    args = ap.parse_args()

    digest = json.loads(_DIGEST.read_text())
    built = _render(_assemble(digest))

    if args.check:
        if not _ARTIFACT.exists():
            print(f"{_ARTIFACT} is missing — run scripts/build_structural_trajectories.py.")
            return 1
        # Compare on PARSED content, not byte-for-byte, so incidental formatting (the hand-written
        # file uses compact inline rate dicts) does not cause a false drift — the SEMANTICS match.
        if json.loads(_ARTIFACT.read_text()) != json.loads(built):
            print(
                f"{_ARTIFACT.name} is OUT OF DATE vs the source digest "
                f"({_DIGEST.relative_to(_ROOT)}).\nRegenerate with "
                "`python scripts/build_structural_trajectories.py` and commit."
            )
            return 1
        print(f"{_ARTIFACT.name} matches the source digest.")
        return 0

    _ARTIFACT.write_text(built)
    print(f"Wrote {_ARTIFACT}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
