"""Contract 3 — the engine protocol + registry.

An engine declares its capabilities, the shock types it understands, and the data it
requires. The registry lists engines; the GUI renders run pages purely from this
metadata (see ADR-0002), so adding an engine adds a GUI option with no GUI code.

``Engine`` is a runtime-checkable Protocol rather than a base class: engines only need
to *satisfy the shape*, keeping them decoupled from this module.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from cge.contracts.results import ResultSet
    from cge.contracts.shocks import Shock


class Capability(StrEnum):
    """What an engine can produce. The GUI uses these to route questions to engines."""

    PRICES = "prices"
    VOLUMES = "volumes"
    GENERAL_EQUILIBRIUM = "general_equilibrium"
    DYNAMIC = "dynamic"


class EngineMeta(BaseModel):
    """Static, declarative description of an engine — the only thing the GUI needs."""

    name: str
    version: str
    description: str
    capabilities: list[Capability]
    supported_shocks: list[str] = Field(description="shock ``type`` strings understood")
    required_data: list[str] = Field(
        default_factory=list, description="e.g. ['IOSystem', 'SatelliteAccount']"
    )

    def supports(self, shock: Shock) -> bool:
        return shock.type in self.supported_shocks


@runtime_checkable
class Engine(Protocol):
    """The behavioural contract. Implementations live in ``cge.engines.*``."""

    meta: EngineMeta

    def run(self, *, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
        """Run ``shocks`` against ``data`` over ``years`` and return a ResultSet.

        ``data`` is a dict of harmonised data objects keyed by type name (matching
        ``meta.required_data``). Static engines may ignore all but one ``year``.
        """
        ...


class Registry:
    """Process-wide registry of engines. Engines register themselves at import time;
    the GUI and CLI enumerate via ``all_meta()``."""

    def __init__(self) -> None:
        self._engines: dict[str, Engine] = {}

    def register(self, engine: Engine) -> Engine:
        if not isinstance(engine, Engine):
            raise TypeError(f"{engine!r} does not satisfy the Engine protocol")
        self._engines[engine.meta.name] = engine
        return engine

    def get(self, name: str) -> Engine:
        return self._engines[name]

    def names(self) -> list[str]:
        return sorted(self._engines)

    def all_meta(self) -> list[EngineMeta]:
        return [self._engines[n].meta for n in self.names()]


# The one shared registry instance.
registry = Registry()
