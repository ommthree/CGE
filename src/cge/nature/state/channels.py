"""Physical ecosystem-service **state** channels (Phase 6b.1/6b.4).

Phase 6 answers *how exposed* a sector is to an ecosystem service. Phase 6b adds the layer
*upstream*
of that: a **physical state variable** for the service itself (a water stock, a pollinator-abundance
index, a soil-quality index, a timber/fish stock), its **baseline** level from published
environmental
accounts, and a documented **state → service-degradation severity** response. A scenario then
specifies
a *physical* degradation/restoration pathway (Phase 6b.2) rather than a bare ``severity`` number,
and
6b.3 translates the resulting state into the existing ``NatureStress`` vocabulary Phase 6.4
consumes.

**State → severity response (6b.4).** The default is **linear** in the fractional shortfall of the
state below baseline — transparent, with no hidden nonlinearity:

    shortfall = clamp((baseline - state) / baseline, 0, 1)      # 0 at baseline, 1 at total collapse
    severity  = clamp(sensitivity * shortfall, 0, 1)

Two **opt-in, documented** nonlinearities are exposed as first-class parameters, never silently
applied:

* ``threshold`` — below a critical state fraction ``x_crit`` (of baseline) the response becomes
  **convex** (a tipping point): degradation accelerates as the state collapses. Above ``x_crit`` the
  linear response is unchanged.
* recovery **hysteresis** — handled in the pathway layer (6b.2), not here: restoration of the state
  lags degradation, so the severity a scenario sees does not instantly rebound when the physical
  state does.

Baselines and sensitivities are **data with citations**, not code constants - every channel carries
a
``Provenance`` and a human-readable ``source_note``. Values shipped here are documented published
central estimates or transparent expert defaults labelled as such; they are illustrative-of-method,
exactly like the ENCORE materiality ramp (see ``docs/models/nature-state.md``).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from cge.contracts.data_objects import Provenance

# The physical mechanism a channel represents. Used by the 6b.5 double-counting check to reason
# about
# which channel "owns" a mechanism that a climate-damage channel (Phase 7c) might also claim.
Mechanism = Literal[
    "water_availability",
    "pollination",
    "soil_quality",
    "forestry_stock",
    "fisheries_stock",
]


class StateResponse(BaseModel):
    """The documented **state → severity** response for one channel (Phase 6b.4).

    ``severity(state)`` returns the fractional service degradation in [0, 1] implied by a physical
    state level, relative to ``baseline``. Linear by default; ``threshold`` opts into a convex
    tipping-point response below ``x_crit``. All parameters are explicit and documented - the
    default
    bakes in no nonlinearity."""

    baseline: float = Field(gt=0.0, description="physical state level at which severity = 0")
    sensitivity: float = Field(
        default=1.0,
        ge=0.0,
        description="linear gain: severity per unit fractional shortfall (1.0 = proportional)",
    )
    threshold: float | None = Field(
        default=None,
        description=(
            "OPT-IN tipping point: state fraction of baseline (in (0,1)) below which the response "
            "turns convex. None = purely linear (the default)."
        ),
    )
    threshold_exponent: float = Field(
        default=2.0,
        gt=1.0,
        description="convexity of the below-threshold response (>1; 2 = quadratic acceleration)",
    )

    @model_validator(mode="after")
    def _check(self) -> StateResponse:
        if self.threshold is not None and not (0.0 < self.threshold < 1.0):
            raise ValueError(
                f"threshold must be a state fraction in (0, 1); got {self.threshold!r}"
            )
        return self

    def shortfall(self, state: float) -> float:
        """Fractional shortfall of ``state`` below ``baseline``, clamped to [0, 1] (0 at/above
        baseline, 1 at total collapse). A state above baseline is a surplus → no shortfall."""
        if not math.isfinite(state):
            raise ValueError(f"state must be finite; got {state!r}")
        return min(1.0, max(0.0, (self.baseline - state) / self.baseline))

    def severity(self, state: float) -> float:
        """Service-degradation severity in [0, 1] implied by the physical ``state``.

        Linear (``sensitivity * shortfall``) unless ``threshold`` is set, in which case the state
        fraction below ``x_crit`` gets an additional convex penalty (a documented tipping point)."""
        sf = self.shortfall(state)
        sev = self.sensitivity * sf
        if self.threshold is not None:
            state_frac = max(0.0, state / self.baseline)
            if state_frac < self.threshold:
                # How far past the threshold, normalised to [0, 1] over (x_crit -> 0).
                depth = (self.threshold - state_frac) / self.threshold
                sev += (1.0 - sev) * depth**self.threshold_exponent
        return min(1.0, max(0.0, sev))


class ServiceStateChannel(BaseModel):
    """A physical **state channel** binding one ecosystem-service state variable to the ENCORE
    service(s) it drives (Phase 6b.1).

    ``mechanism`` names the physical process (used by the 6b.5 double-counting check). ``services``
    are the exact ENCORE service labels a degraded state maps to in ``NatureStress`` (Phase 6b.3).
    ``state_variable``/``unit`` document what is being modelled; ``response`` carries the baseline +
    the state→severity translation. ``provenance``/``source_note`` cite where the baseline and
    sensitivity come from — this is reviewable data, not opaque constants."""

    channel_id: str = Field(description="stable id, e.g. 'water_availability'")
    mechanism: Mechanism
    services: tuple[str, ...] = Field(
        description="ENCORE service label(s) this channel degrades (must be non-empty)"
    )
    state_variable: str = Field(
        description="what the physical state is, e.g. 'renewable water stock'"
    )
    unit: str = Field(
        description="physical unit of the state variable, e.g. 'index (baseline=100)'"
    )
    response: StateResponse
    provenance: Provenance
    source_note: str = Field(description="human-readable citation for baseline + sensitivity")

    @model_validator(mode="after")
    def _check(self) -> ServiceStateChannel:
        if not self.services:
            raise ValueError(f"channel {self.channel_id!r} must drive at least one ENCORE service")
        if any(not s or not s.strip() for s in self.services):
            raise ValueError(f"channel {self.channel_id!r} has a blank ENCORE service label")
        return self

    def severity(self, state: float) -> float:
        """The service-degradation severity in [0, 1] this channel's ``response`` implies for a
        physical ``state`` level."""
        return self.response.severity(state)
