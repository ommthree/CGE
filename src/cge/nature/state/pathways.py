"""Scenario-driven physical **degradation / restoration pathways** (Phase 6b.2, with 6b.4 recovery).

A ``StatePathway`` says how one channel's physical state index evolves over the scenario years — the
*physical* trajectory a scenario specifies instead of a bare productivity-shock number. Two forms:

* an explicit per-year **state index** path (``states={2030: 80, 2040: 65}``), or
* a **rate** form: a per-year degradation (or restoration, if negative) as a fraction of baseline,
  optionally from a start year (``degradation_rate=0.02`` → the index falls 2 points/yr from
  baseline).

**Recovery hysteresis (6b.4).** Restoration of a *physical* state does not instantly restore the
*service* — soils, aquifers and populations recover slowly. When ``recovery_rate`` is set (< 1), the
**effective** state the severity response sees rises toward a recovering physical state by at most
``recovery_rate`` of baseline per year, so a scenario that restores the stock overnight still shows
the
service lagging. Degradation is NOT lagged (damage is felt promptly); only recovery is. The default
(``recovery_rate = None``) applies no hysteresis — nothing nonlinear is assumed silently.

Everything here is deterministic bookkeeping over a supplied year list; there is no stochastic or
optimisation content. See ``docs/models/nature-state.md``.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator


class StatePathway(BaseModel):
    """A physical-state trajectory for one channel over the scenario years (Phase 6b.2).

    Exactly one of ``states`` (explicit year→index) or ``degradation_rate`` (index points of
    baseline
    lost per year) must be given. ``recovery_rate`` opts into recovery hysteresis (6b.4)."""

    channel_id: str = Field(description="the ServiceStateChannel this pathway drives")
    baseline: float = Field(default=100.0, gt=0.0, description="the channel's baseline index level")
    states: dict[int, float] | None = Field(
        default=None, description="explicit year -> physical state index (piecewise-linear between)"
    )
    degradation_rate: float | None = Field(
        default=None,
        description=(
            "index points of baseline lost per year from ``start_year`` (negative = restoration). "
            "Mutually exclusive with ``states``."
        ),
    )
    start_year: int | None = Field(
        default=None,
        description="year the degradation_rate begins (defaults to the first run year)",
    )
    recovery_rate: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "OPT-IN recovery hysteresis: max index points of baseline the EFFECTIVE state may "
            "recover per year (6b.4). None = no lag (the default)."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> StatePathway:
        if (self.states is None) == (self.degradation_rate is None):
            raise ValueError(
                f"pathway {self.channel_id!r}: give exactly one of `states` or `degradation_rate`"
            )
        if self.states is not None:
            for y, v in self.states.items():
                if not math.isfinite(v) or v < 0.0:
                    raise ValueError(
                        f"pathway {self.channel_id!r}: state at {y} must be a non-negative index; "
                        f"got {v!r}"
                    )
        return self

    def _raw_state_at(self, year: int, first_year: int) -> float:
        """The physical state index at ``year`` BEFORE recovery hysteresis."""
        if self.states is not None:
            return _piecewise_linear(self.states, year, self.baseline)
        start = self.start_year if self.start_year is not None else first_year
        elapsed = max(0, year - start)
        return max(0.0, self.baseline - self.degradation_rate * elapsed)

    def state_path(self, years: list[int]) -> dict[int, float]:
        """The EFFECTIVE physical state index per year, applying recovery hysteresis if configured.

        Degradation is felt promptly; recovery is capped at ``recovery_rate`` index points/yr so a
        restored physical state lets the service recover only gradually (6b.4)."""
        if not years:
            return {}
        ordered = sorted(years)
        first = ordered[0]
        raw = {y: self._raw_state_at(y, first) for y in ordered}
        if self.recovery_rate is None:
            return raw
        # Apply the recovery cap forward in time: the effective state may fall freely but rise by at
        # most recovery_rate per year.
        eff: dict[int, float] = {}
        prev_year = None
        prev_val = None
        for y in ordered:
            target = raw[y]
            if prev_val is None:
                eff[y] = target
            else:
                dt = max(1, y - prev_year)
                if target >= prev_val:  # recovering → cap the rise
                    eff[y] = min(target, prev_val + self.recovery_rate * dt)
                else:  # degrading → prompt
                    eff[y] = target
            prev_year, prev_val = y, eff[y]
        return eff


def _piecewise_linear(points: dict[int, float], year: int, baseline: float) -> float:
    """Piecewise-linear interpolation of ``points`` at ``year``; flat-extrapolated outside the
    range.

    Before the first given year the level is the ``baseline`` (the scenario starts at reference
    condition and only the specified trajectory departs from it)."""
    if not points:
        return baseline
    xs = sorted(points)
    if year <= xs[0]:
        # Ramp from baseline at the run start to the first specified point is the scenario author's
        # job to express; before the first point we hold at the first point's level (a step) only if
        # they placed it at/after the start. To keep it simple and explicit: flat at first point.
        return points[xs[0]] if year >= xs[0] else baseline
    if year >= xs[-1]:
        return points[xs[-1]]
    for lo, hi in zip(xs, xs[1:], strict=False):
        if lo <= year <= hi:
            if hi == lo:
                return points[lo]
            frac = (year - lo) / (hi - lo)
            return points[lo] + frac * (points[hi] - points[lo])
    return points[xs[-1]]
