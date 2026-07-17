"""Concordance operations.

A ``ConcordanceMap`` (contract 1) maps source labels to target labels with weights that
sum to 1 out of each source. The contract validates the sum-to-one invariant on
construction; this module adds the coverage (no-orphan) check and the linear-algebra the
aggregation machinery needs.

The **bridge matrix** ``B`` is target×source with ``B[t, s] = weight(s -> t)``. For a
concordance used to *aggregate* (many source sectors into few target sectors), each source
maps to exactly one target with weight 1, so ``B`` is a 0/1 grouping matrix. The general
weighted form supports splitting a source across targets (needed for ENCORE and for
reconciling mismatched classifications in P7).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cge.contracts.data_objects import ConcordanceMap


def check_covers(cmap: ConcordanceMap, source_labels: list[str]) -> list[str]:
    """Return source labels that the concordance does not map (orphans). Empty = full
    coverage. Callers decide whether an orphan is fatal (aggregation) or a warning."""
    return [s for s in source_labels if s not in cmap.weights]


def bridge_matrix(cmap: ConcordanceMap, source_labels: list[str]) -> pd.DataFrame:
    """Build the target×source bridge matrix ``B`` restricted to ``source_labels``.

    Raises on orphans: an aggregation must account for every source label, otherwise
    output would silently drop economic activity.
    """
    orphans = check_covers(cmap, source_labels)
    if orphans:
        raise ValueError(
            f"Concordance leaves {len(orphans)} source labels unmapped, e.g. {orphans[:5]}"
        )
    targets: list[str] = []
    for mapping in cmap.weights.values():
        for t in mapping:
            if t not in targets:
                targets.append(t)

    B = pd.DataFrame(0.0, index=targets, columns=source_labels)
    for s in source_labels:
        for t, w in cmap.weights[s].items():
            B.loc[t, s] = w
    return B


def save_concordance(cmap: ConcordanceMap, path: str | Path) -> None:
    Path(path).write_text(cmap.model_dump_json(indent=2))


def load_concordance(path: str | Path) -> ConcordanceMap:
    return ConcordanceMap.model_validate_json(Path(path).read_text())


def one_to_one(
    mapping: dict[str, str],
    *,
    from_classification: str,
    to_classification: str,
    provenance,
) -> ConcordanceMap:
    """Convenience builder for the common aggregation case: each source maps to one
    target with weight 1 (a pure grouping)."""
    weights = {src: {tgt: 1.0} for src, tgt in mapping.items()}
    return ConcordanceMap(
        provenance=provenance,
        from_classification=from_classification,
        to_classification=to_classification,
        weights=weights,
    )


def to_json_dict(cmap: ConcordanceMap) -> dict:
    return json.loads(cmap.model_dump_json())
