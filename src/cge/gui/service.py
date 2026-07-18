"""GUI service layer.

A thin façade over the store, registry and runner so GUI pages depend on *one* module
instead of reaching into internals. Nothing here imports Streamlit — it is plain Python and
unit-testable, which keeps the actual page code (which does import Streamlit) trivial.

Everything the GUI can do is a method here: enumerate builds, load a build's frames for the
explorer, read quality, list engines, run a scenario, kick off a data build as a background
job. Pages render what these return.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import pandas as pd

import cge.engines  # noqa: F401  (registers engines)
from cge.contracts.engine import EngineMeta, registry
from cge.contracts.results import ResultSet
from cge.data.store import DataStore, default_store
from cge.scenarios.loader import Scenario


@dataclass
class FrameView:
    """A named 2-D frame from a build, ready for the spreadsheet-style explorer."""

    name: str
    df: pd.DataFrame
    description: str


class GuiService:
    def __init__(self, store: DataStore | None = None) -> None:
        self.store = store or default_store()

    # -- data catalogue --------------------------------------------------------
    def catalogue(self) -> pd.DataFrame:
        return self.store.catalogue()

    def build_ids(self) -> list[str]:
        return self.store.build_ids()

    def build_meta(self, build_id: str):
        return self.store.load_meta(build_id)

    # -- data explorer ---------------------------------------------------------
    def frames(self, build_id: str) -> dict[str, FrameView]:
        """Return the browsable frames of a build (A-matrix, final demand, satellites).

        The A-matrix can be huge; the explorer page slices (row/column label filters + a
        cell cap) before rendering. This method returns the frames as loaded — callers must
        slice before display.
        """
        data = self.store.load(build_id)
        io = data["IOSystem"]
        out: dict[str, FrameView] = {
            "A (technical coefficients)": FrameView(
                "A", io.A, "Input i (row) required per unit output of j (column)."
            ),
            "Final demand": FrameView("final_demand", io.final_demand, "Final demand per product."),
        }
        for name, sat in data.get("satellites", {}).items():
            out[f"Satellite: {name}"] = FrameView(
                name, sat.data, f"{name} intensities (per unit output), stressor × product."
            )
        return out

    def label_axis(self, build_id: str) -> list[str]:
        """The sector×region labels of a build (for slice pickers/search)."""
        io = self.store.load(build_id)["IOSystem"]
        return list(io.A.columns)

    def sectors(self, data_source: str) -> list[str]:
        """Distinct sector labels for a data source ('toy' or a build id) — used to populate the
        energy-carrier picker on the Run page. Works for the toy fixture as well as store builds."""
        from cge.runner import load_data

        io = load_data(data_source, store=self.store)["IOSystem"]
        return sorted({lab.split(":", 1)[1] for lab in io.A.columns})

    # -- quality ---------------------------------------------------------------
    def quality(self, build_id: str):
        return self.store.load_quality(build_id)

    # -- engines ---------------------------------------------------------------
    def engines(self) -> list[EngineMeta]:
        return registry.all_meta()

    def engine_meta(self, name: str) -> EngineMeta:
        return registry.get(name).meta

    # -- runs ------------------------------------------------------------------
    def run(self, scenario: Scenario, *, data_source: str) -> ResultSet:
        """Run a scenario in-process (fine for the small build; see roadmap P3 decisions).

        Passes this service's store so runs resolve builds from the same store the GUI
        browses (not only the process-default store)."""
        from cge.runner import run_scenario

        return run_scenario(scenario, data_source=data_source, store=self.store)

    def start_build(self, *, test: bool = True, year: int = 2019) -> subprocess.Popen:
        """Kick off a data build as a background subprocess so a long download doesn't block
        the UI. Returns the Popen; the caller streams ``stdout``. (Job wrapper, task 3.4.)"""
        cmd = [sys.executable, "-m", "cge.cli", "build"]
        cmd += ["--test"] if test else ["--exiobase", "--year", str(year)]
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )


def get_service() -> GuiService:
    return GuiService()
