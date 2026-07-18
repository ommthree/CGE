"""Live EXIOBASE known-answer validation suite (opt-in).

Registers checks into the standing ``cge validate`` suite **only when** a real EXIOBASE
archive is available (``CGE_EXIOBASE_ARCHIVE`` points at an ``IOT_YYYY_pxp.zip``). Offline —
the default — this suite registers nothing, so ``cge validate`` stays fast and green. This is
the "the standing suite tightens when live data exists" behaviour promised in docs/validation.md.

The heavy parse is done once, lazily, and cached across checks.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from cge.validation.framework import check

SUITE = "exiobase_live"

_ARCHIVE = os.environ.get("CGE_EXIOBASE_ARCHIVE")
_ENABLED = bool(_ARCHIVE and Path(_ARCHIVE).exists())


@lru_cache(maxsize=1)
def _adapted():
    from cge.data.adapters.exiobase import adapt_pymrio, parse_exiobase

    pio = parse_exiobase(_ARCHIVE)
    io, sats = adapt_pymrio(pio, source="EXIOBASE", source_version="live", reference_year=0)
    return pio, io, sats


if _ENABLED:  # register checks only when a real archive is present

    @check(SUITE, "adapter_reproduces_full_mrio")
    def _shape():
        _, io, sats = _adapted()
        ok = io.A.shape == (9800, 9800) and any(s.name == "GHG" for s in sats)
        return (
            ok,
            f"MRIO shape {io.A.shape}, GHG account present={any(s.name == 'GHG' for s in sats)}",
        )

    @check(SUITE, "adapter_preserves_global_co2")
    def _preserve():
        import numpy as np

        from cge.data.adapters.exiobase import _stressor_unit_to_tonne

        pio, io, sats = _adapted()
        ghg = next(s for s in sats if s.name == "GHG")
        x = pio.x["indout"].reindex(pio.A.columns).to_numpy(float)
        adapter = float(np.sum(ghg.data.loc["CO2"].reindex(list(io.A.columns)).to_numpy(float) * x))
        ext = pio.satellite
        raw = sum(
            float(ext.F.loc[[s]].sum(axis=0).sum()) * _stressor_unit_to_tonne(ext, s)
            for s in ext.F.index
            if str(s).upper().startswith("CO2")
        )
        rel = abs(adapter - raw) / raw
        return rel < 1e-6, f"adapter vs raw global CO2 rel diff = {rel:.2e}", rel, 1e-6

    @check(SUITE, "global_co2_plausible_magnitude")
    def _magnitude():
        import numpy as np

        pio, io, sats = _adapted()
        ghg = next(s for s in sats if s.name == "GHG")
        x = pio.x["indout"].reindex(pio.A.columns).to_numpy(float)
        gt = (
            float(np.sum(ghg.data.loc["CO2"].reindex(list(io.A.columns)).to_numpy(float) * x)) / 1e9
        )
        return 20.0 < gt < 45.0, f"global CO2 = {gt:.1f} Gt (plausible 20-45)", gt
