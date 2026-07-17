"""Contract 5 — module slots for the pathway stack (P7).

Climate (emissions -> temperature) and damages (temperature -> shocks) are interfaces
with, eventually, one implementation each (FaIR; a DICE-style damage function). They
are deliberately *swappable and omittable*: the core cost/volume tool never depends on
them. Defined as Protocols here so the recursive-dynamic wrapper can be written against
the interface before any implementation exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cge.contracts.shocks import Shock


@runtime_checkable
class ClimateModule(Protocol):
    """emissions time series -> temperature time series."""

    name: str
    version: str

    def temperature(self, emissions: dict[int, float]) -> dict[int, float]:
        """Map year -> emissions (GtCO2e) to year -> temperature anomaly (°C)."""
        ...


@runtime_checkable
class DamageModule(Protocol):
    """temperature time series -> productivity shocks fed back to engines.

    Damage functions are the most contested object in climate economics; an
    implementation must name its published source and results must be labelled
    illustrative (see roadmap P7.4).
    """

    name: str
    version: str
    source: str

    def shocks(self, temperature: dict[int, float]) -> list[Shock]: ...
