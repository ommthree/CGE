"""Double-counting reconciliation between nature-state (6b) and climate-damage (7c) channels (6b.5).

Some physical mechanisms are claimed by BOTH a Phase-6b nature-state channel and a Phase-7c
physical-climate-risk channel. For example, heat- and drought-driven water stress reduces water
availability (a 6b ``water_availability`` mechanism) AND is a climate physical-risk pathway (7c's
water-availability channel); soil degradation likewise. If a scenario applies both, the SAME
physical
effect on the economy is counted twice.

Phase 7c does not exist yet, so this module ships the **reconciliation RULE** as a documented
ownership registry plus an automated **conflict check** that a scenario runner can call once 7c
lands
(and that the test-suite exercises now against a drafted 7c channel list). The rule is deliberately
conservative: a mechanism is **owned by exactly one channel type**, and a scenario that drives the
same owned mechanism through both a nature-state pathway and a climate-damage channel is REJECTED
with
a message naming the mechanism and both claimants — it is not silently summed.

Ownership rule (drafted 2026-08-15, see ``docs/models/nature-state.md`` §double-counting):

* ``water_availability`` and ``soil_quality`` are **shared** mechanisms. The reconciliation assigns
  their *physical-climate* portion to the **7c climate channel** (it models the hazard — heat,
  drought — that drives the state change) and their *non-climate* portion (e.g. aquifer over-
  abstraction, tillage practice) to the **6b nature channel**. A scenario may run EITHER as the
  owner
  of a mechanism, but not both on the same mechanism, unless it explicitly declares them disjoint.
* ``pollination``, ``forestry_stock``, ``fisheries_stock`` are **nature-owned** — 7c has no channel
  for them, so no conflict arises.

This is a Definition-of-Done criterion for Phase 6b, not an afterthought: it is exactly the kind of
gap a later review would catch (roadmap 6b.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cge.nature.state.channels import Mechanism

# Mechanisms a Phase-7c physical-climate-risk channel is expected to model too (drafted from the
# roadmap's 7c channel library: heat labour productivity, water availability, agriculture/soil,
# energy, transport). Only the ones that OVERLAP a 6b nature mechanism matter here.
CLIMATE_SHARED_MECHANISMS: frozenset[Mechanism] = frozenset({"water_availability", "soil_quality"})

# Mechanisms owned outright by nature-state channels (7c has no counterpart) — never a conflict.
NATURE_OWNED_MECHANISMS: frozenset[Mechanism] = frozenset(
    {"pollination", "forestry_stock", "fisheries_stock"}
)


class DoubleCountError(ValueError):
    """A scenario drives the same shared physical mechanism through BOTH a nature-state channel and
    a
    climate-damage channel — the same physical effect would be counted twice (Phase 6b.5)."""


@dataclass
class DoubleCountReport:
    """Result of the reconciliation check."""

    conflicts: dict[str, list[str]] = field(default_factory=dict)  # mechanism -> [claimants]
    ok: bool = True

    def raise_if_conflict(self) -> None:
        if not self.ok:
            detail = "; ".join(
                f"{mech}: claimed by {sorted(who)}" for mech, who in sorted(self.conflicts.items())
            )
            raise DoubleCountError(
                "nature and climate-damage channels double-count the same physical mechanism(s): "
                f"{detail}. Run one channel as the mechanism's owner, or give them disjoint "
                "coverage. See docs/models/nature-state.md §double-counting."
            )


def check_double_counting(
    nature_mechanisms: set[Mechanism] | list[Mechanism],
    climate_mechanisms: set[Mechanism] | list[Mechanism],
) -> DoubleCountReport:
    """Flag mechanisms driven by BOTH a nature-state channel and a climate-damage channel (6b.5).

    ``nature_mechanisms`` are the ``ServiceStateChannel.mechanism`` values a scenario's nature-state
    pathways drive; ``climate_mechanisms`` are the physical mechanisms its Phase-7c climate-damage
    channels drive (an empty set until 7c exists). A mechanism in the intersection that is a SHARED
    mechanism is a conflict; a nature-owned mechanism is never a conflict even if a (mis-configured)
    climate channel claims it — but such a claim is surfaced too, since 7c should not own it."""
    nat = set(nature_mechanisms)
    clim = set(climate_mechanisms)
    report = DoubleCountReport()
    for mech in nat & clim:
        report.conflicts[mech] = ["nature-state", "climate-damage"]
    if report.conflicts:
        report.ok = False
    return report


def nature_mechanisms_of(channels) -> set[Mechanism]:
    """The set of physical mechanisms a collection of ``ServiceStateChannel`` drives — the input to
    ``check_double_counting`` for a scenario's nature side."""
    return {ch.mechanism for ch in channels}
