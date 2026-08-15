"""Physical ecosystem-service **state** modelling (Phase 6b).

Sits *upstream* of the Phase-6 ENCORE exposure layer: model a physical state variable for an
ecosystem service (``channels`` / ``baselines``), specify a scenario degradation/restoration
**pathway** for it (``pathways``), map the resulting state through a documented state→severity
response (Phase 6b.1/6b.4) and emit the Phase-6.4 ``NatureStress`` vocabulary
(``translate_state``) — so a scenario specifies a *physical* trajectory, not a bare severity number.
``double_count`` reconciles shared physical mechanisms with Phase-7c climate-damage channels (6b.5).

See ``docs/models/nature-state.md``.
"""

from cge.nature.state.baselines import (
    SHIPPED_CHANNELS,
    TOY_CHANNELS,
    get_channel,
    shipped_channels,
    toy_channels,
)
from cge.nature.state.channels import (
    Mechanism,
    ServiceStateChannel,
    StateResponse,
)
from cge.nature.state.double_count import (
    CLIMATE_SHARED_MECHANISMS,
    NATURE_OWNED_MECHANISMS,
    DoubleCountError,
    DoubleCountReport,
    check_double_counting,
    nature_mechanisms_of,
)
from cge.nature.state.pathways import StatePathway
from cge.nature.state.translate_state import (
    build_state_scenario,
    state_severity_path,
    state_to_nature_stresses,
)

__all__ = [
    # channel model + response (6b.1/6b.4)
    "ServiceStateChannel",
    "StateResponse",
    "Mechanism",
    # shipped registry (6b.1) + illustrative toy channels (offline tutorial)
    "SHIPPED_CHANNELS",
    "shipped_channels",
    "TOY_CHANNELS",
    "toy_channels",
    "get_channel",
    # pathways (6b.2)
    "StatePathway",
    # translation to NatureStress (6b.3)
    "state_severity_path",
    "state_to_nature_stresses",
    "build_state_scenario",
    # double-counting reconciliation (6b.5)
    "check_double_counting",
    "nature_mechanisms_of",
    "DoubleCountError",
    "DoubleCountReport",
    "CLIMATE_SHARED_MECHANISMS",
    "NATURE_OWNED_MECHANISMS",
]
