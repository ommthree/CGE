"""Scenario loading. A scenario is a declarative YAML file: metadata + a list of typed
shocks + the years to run. It round-trips through the shock discriminated union so the
correct ``Shock`` subclass is restored on load."""

from cge.scenarios.loader import Scenario, load_scenario

__all__ = ["Scenario", "load_scenario"]
