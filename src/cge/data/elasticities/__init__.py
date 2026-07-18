"""Elasticity library (task 4.1).

Assembled behavioural elasticities with per-value sourcing and low/central/high uncertainty,
as ``ElasticitySet`` objects. There is no clean open elasticity database, so coverage is
partial and every value is tagged with its confidence and source; goods without a value get a
documented default (tagged ``default``) so the PE engine can always run, transparently.

The default demand set is keyed to the coarse sector classification the default EXIOBASE
aggregation produces (``cge.data.build._coarse_sector``), so it lines up with a runnable build.
"""

from cge.data.elasticities.library import DEFAULT_DEMAND_ELASTICITY, default_demand_set

__all__ = ["default_demand_set", "DEFAULT_DEMAND_ELASTICITY"]
