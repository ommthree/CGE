"""ENCORE ingestion and the materiality → numeric scale (Phase 6.1).

ENCORE (Exploring Natural Capital Opportunities, Risks and Exposure) rates, for each production
process, how much it **depends on** each ecosystem service (pollination, surface water, climate
regulation, …) and how much it **impacts** each natural-capital asset via impact drivers. Ratings
are ordinal **materiality classes**: Very High / High / Medium / Low / Very Low (VH/H/M/L/VL).

This module turns that ordinal knowledge base into the numeric, provenance-carrying data objects the
exposure engine (6.3) and the nature→shock translation (6.4) consume. Two deliberate design points
the roadmap flags:

- **The materiality → numeric scale is documented and explicit** (``MATERIALITY_SCALE`` below), not
  buried — it drives every downstream number, so it is a named, cited choice a reviewer can change.
- **Ratings are DATA, not code.** An ``EncoreDependencies`` object carries its own provenance
  (source, version, retrieved date) so a run records exactly which ENCORE snapshot produced it. The
  **real May-2026 ENCORE knowledge base is ingested** via ``load_encore_ratings_wide`` (the raw wide
  ISIC×service export): it distinguishes ``ND`` ("No Data", kept as a first-class state — see
  ``nd_mask``) from ``N/A``/blank (a genuine zero), and the pressure/impact ratings are ingested
  separately (``kind="impact"``). A small **synthetic/expert-designed** fixture is also shipped for
  offline CI; it is illustrative, not published-sourced.

See ``docs/models/nature-encore.md`` for the equations and sourcing.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import pandas as pd
from pydantic import Field, model_validator

from cge.contracts.data_objects import Provenance, _DataObject

# The five ENCORE materiality classes, most-to-least material.
MaterialityClass = Literal["VH", "H", "M", "L", "VL"]

# Materiality → numeric scale (review-visible, load-bearing). A linear 0.2-step ramp VL..VH.
#
# HONEST SOURCING (review P1 2026-08-07): this specific numeric ramp is a **synthetic / expert-
# designed default**, NOT a value published by DNB. DNB "Indebted to nature" (van Toor et al. 2020)
# worked with the ordinal classes and in practice retained only the High/Very-High dependencies; it
# does not document this 1.0/0.8/0.6/0.4/0.2 mapping. The linear ramp is our transparent, easily-
# swapped choice for turning ENCORE's ordinal classes into a [0, 1] weight — it drives every
# propagated exposure score, so it is named here for a reviewer to change (e.g. a convex ramp
# VH=1.0, H=0.5, M=0.25, … to make only the highest classes bite, or a High/Very-High-only screen
# matching DNB's practice). Treat it as an assumption to calibrate, not a published elasticity.
MATERIALITY_SCALE: dict[str, float] = {"VH": 1.0, "H": 0.8, "M": 0.6, "L": 0.4, "VL": 0.2}

# ENCORE's real ratings tables carry two non-scored tokens alongside VH..VL (verified against the
# May-2026 knowledge base):
#   - "ND"  = **No Data**: the dependency was not assessed. Genuinely UNKNOWN; must not be silently
#             treated as "no dependency" — it is kept as a first-class state (nd_mask), so a gap is
#             visible rather than masquerading as zero risk (review P1 2026-08-07).
#   - "N/A" = **Not Applicable**: the service does not apply to that process → a true zero. (Blank
#             cells in the dependency table mean the same: no rated dependency → 0.)
ND_TOKEN = "ND"
NA_TOKENS = frozenset({"N/A", "NA", "NAN", ""})  # not-applicable / blank → zero (not unknown)


def materiality_to_score(cls: str) -> float:
    """Map an ENCORE materiality class to its numeric [0, 1] score (``MATERIALITY_SCALE``).

    ``ND`` (No Data) and ``N/A``/blank are NOT scored here — they are data states, not ratings, and
    the caller decides their policy (``EncoreDependencies`` keeps ND distinct, N/A/blank as 0).
    Passing one of them is a programming error, so this raises."""
    key = str(cls).strip().upper()
    if key not in MATERIALITY_SCALE:
        raise ValueError(
            f"unknown ENCORE materiality class {cls!r}; expected {sorted(MATERIALITY_SCALE)} "
            f"(data states 'ND'/'N/A'/blank are handled by EncoreDependencies, not scored here)"
        )
    return MATERIALITY_SCALE[key]


class EncoreDependencies(_DataObject):
    """ENCORE dependency ratings as first-class, provenance-carrying data.

    ``ratings`` is a long table with columns ``process``, ``service``, ``materiality`` — a scored
    class in {VH,H,M,L,VL} **or** the token ``ND`` (No Data). Not-applicable / blank cells are NOT
    rows here (absent (process, service) = no rated dependency = 0). Each (process, service) pair
    appears at most once.

    **ND is first-class, not zero (review P1 2026-08-07).** ``score_matrix`` maps a rated class via
    ``MATERIALITY_SCALE`` and an absent pair to 0, and — by the documented default policy — treats
    ``ND`` as 0 *for the numeric propagation* too, BUT ``nd_mask`` flags exactly which cells are ND
    and ``data_coverage`` reports the rated fraction, so a data gap is visible and never silently
    read as "no dependency / no risk". A caller can screen or widen ND cells using ``nd_mask``.
    """

    _COLUMNS: ClassVar[tuple[str, ...]] = ("process", "service", "materiality")

    ratings: pd.DataFrame = Field(description="long table: process, service, materiality")
    kind: Literal["dependency", "impact"] = "dependency"

    @model_validator(mode="after")
    def _validate(self) -> EncoreDependencies:
        df = self.ratings
        missing = [c for c in self._COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"EncoreDependencies.ratings is missing columns {missing}")
        # Every materiality value must be a scored class OR the ND (No Data) token. N/A / blank must
        # NOT appear as rows (they mean "no dependency" and are absent by construction); catching
        # them here stops a mis-ingested table silently importing "not applicable" as a state.
        allowed = set(MATERIALITY_SCALE) | {ND_TOKEN}
        bad = sorted({str(m) for m in df["materiality"] if str(m).strip().upper() not in allowed})
        if bad:
            raise ValueError(
                f"EncoreDependencies.ratings has unrecognised materiality value(s) {bad}; expected "
                f"a scored class {sorted(MATERIALITY_SCALE)} or 'ND'. (N/A/blank are not rows; an "
                "absent (process, service) pair already means no rated dependency.)"
            )
        # (process, service) must be unique — a duplicated pair would double-count or silently
        # override, so reject rather than pick one.
        dupes = df.duplicated(subset=["process", "service"])
        if dupes.any():
            example = df[dupes][["process", "service"]].head(3).to_dict("records")
            raise ValueError(
                f"EncoreDependencies.ratings has duplicate (process, service) pairs, e.g. {example}"
            )
        return self

    @property
    def processes(self) -> list[str]:
        return sorted(self.ratings["process"].unique())

    @property
    def services(self) -> list[str]:
        return sorted(self.ratings["service"].unique())

    def _is_nd(self, materiality) -> bool:
        return str(materiality).strip().upper() == ND_TOKEN

    def score_matrix(self) -> pd.DataFrame:
        """A dense process × service numeric dependency matrix in [0, 1]. A rated class is scored
        via ``MATERIALITY_SCALE``; an absent pair is 0 (no rated dependency); an ``ND`` cell is 0 in
        the numeric propagation (documented default) — but see ``nd_mask``/``data_coverage``:
        ND is a KNOWN-UNKNOWN, tracked separately, not conflated with a genuine zero."""
        m = pd.DataFrame(0.0, index=self.processes, columns=self.services)
        for row in self.ratings.itertuples(index=False):
            if not self._is_nd(row.materiality):
                m.loc[row.process, row.service] = materiality_to_score(row.materiality)
        return m

    def nd_mask(self) -> pd.DataFrame:
        """Boolean process × service matrix, True where the cell is ``ND`` (No Data) — the explicit
        record of what is unknown (as opposed to a rated 0 or an absent/not-applicable pair)."""
        m = pd.DataFrame(False, index=self.processes, columns=self.services)
        for row in self.ratings.itertuples(index=False):
            if self._is_nd(row.materiality):
                m.loc[row.process, row.service] = True
        return m

    def data_coverage(self) -> float:
        """Fraction of the rated (process, service) cells that carry an actual rating rather than
        ``ND`` — a headline data-quality number so a run can report how much of the ENCORE input is
        genuinely known vs No-Data. 1.0 when nothing is ND; lower as ND cells accumulate."""
        total = len(self.ratings)
        if total == 0:
            return 1.0
        nd = int(self.ratings["materiality"].map(self._is_nd).sum())
        return (total - nd) / total


def load_encore_csv(
    path: str,
    *,
    provenance: Provenance,
    kind: Literal["dependency", "impact"] = "dependency",
) -> EncoreDependencies:
    """Ingest an ALREADY-TIDY ratings CSV (long columns ``process``, ``service``, ``materiality``)
    into an ``EncoreDependencies`` object. This is the simple/pre-processed path; for the **raw
    ENCORE knowledge-base export** (a wide ISIC×service matrix with ND/N-A cells) use
    ``load_encore_ratings_wide`` below, which does the real melt + N/A-vs-ND handling.
    """
    df = pd.read_csv(path)
    return EncoreDependencies(provenance=provenance, ratings=df, kind=kind)


# The leading columns of ENCORE's wide ratings tables (06/07) are the ISIC identifier hierarchy;
# every remaining column is an ecosystem service (06) or a pressure/impact driver (07).
_ISIC_ID_COLUMNS = (
    "ISIC Unique code",
    "ISIC Section",
    "ISIC Division",
    "ISIC Group",
    "ISIC Class",
)


def _encore_process_ids(raw: pd.DataFrame, process_col: str) -> pd.Series:
    """A unique, human-meaningful process id per row of an ENCORE wide ratings table.

    Base is the ISIC code (``process_col``). Where that code appears on more than one row (ENCORE
    split it into finer production processes), append the finest-populated ISIC name — Class, else
    Group, else Division — to disambiguate, as ``"<code> — <name>"``. Rows whose code is already
    unique keep the bare code. Raises if disambiguation still leaves a collision (unexpected)."""
    code = raw[process_col].astype(str).str.strip()

    def _finest_name(row) -> str:
        for col in ("ISIC Class", "ISIC Group", "ISIC Division"):
            val = row.get(col)
            if val is not None and str(val).strip() and str(val).strip().lower() != "nan":
                return str(val).strip()
        return ""

    dup_code = code.duplicated(keep=False)
    ids = []
    for i, (_, row) in enumerate(raw.iterrows()):
        c = code.iloc[i]
        if dup_code.iloc[i]:
            name = _finest_name(row)
            ids.append(f"{c} — {name}" if name else c)
        else:
            ids.append(c)
    out = pd.Series(ids, index=raw.index)
    if out.duplicated().any():
        clashes = sorted(out[out.duplicated(keep=False)].unique())[:5]
        raise ValueError(
            f"ENCORE process ids still collide after disambiguation, e.g. {clashes}; the ISIC "
            "code + finest name was not unique. Inspect the ratings file."
        )
    return out


def load_encore_ratings_wide(
    path: str,
    *,
    provenance: Provenance,
    kind: Literal["dependency", "impact"] = "dependency",
    process_col: str = "ISIC Unique code",
) -> EncoreDependencies:
    """Ingest a **raw ENCORE knowledge-base ratings CSV** — the wide ISIC×(service|pressure) matrix
    shipped as ``06. Dependency mat ratings.csv`` / ``07. Pressure mat ratings.csv`` — into an
    ``EncoreDependencies`` object, doing the real transformations the tidy path assumes away:

    - **wide → long melt.** Each non-ISIC column is a service (dependencies) or pressure/impact
      driver (pressures); melt to one (process, service, materiality) row per rated cell.
    - **process id.** ``process_col`` (default the ``ISIC Unique code``, e.g. ``A_1_14_141``) is the
      stable process identifier used downstream; the other ISIC hierarchy columns are dropped (kept
      in the raw file for the concordance step).
    - **N/A vs ND vs blank (review P1 2026-08-07).** ``ND`` (No Data) rows are KEPT with materiality
      ``ND`` so the contract can track the gap distinctly; ``N/A`` and blank cells mean *not
      applicable / no dependency* and are DROPPED (an absent pair already scores 0). This is the one
      transformation the tidy CSV can't express and the whole reason a real adapter is needed.

    **Process identity (real-data subtlety).** The ``ISIC Unique code`` is NOT unique on its own:
    ENCORE splits some codes into finer *production processes* (e.g. ``D_35_351`` →
    fossil / nuclear / hydro / solar / wind… energy production), distinguished by the ``ISIC Class``
    name, while most rows analysed at Group/Division level leave ``ISIC Class`` blank. So the id is
    composed as ``<ISIC code>`` plus, only where that code repeats, the finest-populated ISIC
    name (Class → Group → Division) to disambiguate — giving a stable, unique, human-meaningful key.

    Whitespace in ENCORE's headers (e.g. a trailing space on some service names) is stripped, and a
    UTF-8 BOM on the first column is tolerated (matched on the cleaned name)."""
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lstrip("﻿") for c in raw.columns]
    if process_col not in raw.columns:
        raise ValueError(
            f"load_encore_ratings_wide: process column {process_col!r} not found; columns start "
            f"{list(raw.columns)[:6]}"
        )
    id_cols = [c for c in _ISIC_ID_COLUMNS if c in raw.columns]
    extra_meta = [c for c in raw.columns if c.startswith("ISIC level")]  # a non-service note column
    value_cols = [c for c in raw.columns if c not in id_cols and c not in extra_meta]

    raw = raw.copy()
    raw["process"] = _encore_process_ids(raw, process_col)
    long = raw.melt(
        id_vars=["process"], value_vars=value_cols, var_name="service", value_name="materiality"
    )
    long["service"] = long["service"].astype(str).str.strip()
    long["materiality"] = long["materiality"].map(lambda v: str(v).strip().upper())

    # Drop not-applicable / blank cells (they mean 0 and are absent by construction); keep rated
    # classes AND ND (No Data), which the contract tracks as a distinct known-unknown.
    keep = ~long["materiality"].isin(NA_TOKENS)
    long = long.loc[keep, ["process", "service", "materiality"]].reset_index(drop=True)
    return EncoreDependencies(provenance=provenance, ratings=long, kind=kind)
