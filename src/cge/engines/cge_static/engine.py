"""Engine 3 — static computable general equilibrium (roadmap Phase 5).

Wraps the calibrated pilot CGE (``calibrate`` + ``model`` + ``solver``) behind the ``Engine``
protocol, so the GUI/CLI pick it up via the registry with no changes. Given a benchmark ``SAM``,
it calibrates the model to reproduce the base year exactly, applies a ``CarbonPrice`` as a
per-unit emissions cost wedge, solves for the new equilibrium, and emits a ``ResultSet`` of price
and volume changes plus GE outputs (factor prices, real/volume GDP, current-price GDP, a relative
GDP deflator, welfare, carbon revenue). The CPI is the numéraire, so the deflator is a *relative*
GDP-price index (GDP basket vs the CPI numéraire), NOT absolute inflation.

**Scope:** the correctness-first pilot from `docs/phase-5-plan.md` §5.2a — Leontief intermediates,
CES/Cobb-Douglas value added, Cobb-Douglas household demand, CPI numéraire, with **revenue
recycling** (lump_sum / labour_tax_cut). Three variants share this engine, selected automatically
from the SAM's account structure: a **closed** single-region economy; an **open** economy
(Armington imports + CET exports + a rest-of-world account, CES value added, an endogenous
exchange rate) when the SAM carries a ``ROW`` account; and a **multi-region** economy with true
bilateral trade among the build's own regions (destination-specific route prices, explicit
bilateral market clearing — see `model_multi.py`) when the SAM carries several region-tagged
households. All three pass benchmark replication, homogeneity and Walras (the `cge_static`
validation suite); the open and multi-region variants additionally show carbon leakage, and the
multi-region variant clears every bilateral trade route and factor market under shock, not just at
the benchmark. A **non-zero current account** is supported (foreign savings enter household income
as the ROW capital transfer er·Sf in the open model, or a bilateral capital transfer per region in
the multi-region model). A **government account** (Phase 5d.1) is supported in ALL THREE variants:
a ``GOV`` SAM account (closed/open) or one ``GOV_<r>`` per region (multi-region) makes government
a real institution — it collects carbon revenue (and an optional benchmark household→government
direct tax, stored as a rate on factor income) and spends on its own calibrated demand vector
under a balanced budget, with ``fiscal_balance``/``gov_spending`` emitted. A **savings-investment
account** (Phase 5d.2, all variants) is supported via a ``SAVINV`` SAM account (``SAVINV_<r>`` per
region in multi): household savings (a calibrated rate on disposable income) become investment
demand with its own sectoral composition, under a ``savings_driven`` (default) or ``fixed_real``
closure, with ``investment``/``savings`` emitted. In the open/multi variants the foreign-savings
inflow re-routes into the investment pool (financing investment, not consumption). A **labour-
market closure** (Phase 5d.4, closed variant) adds an optional ``labour_floor`` (a wage floor,
via a regime-switch) with involuntary ``unemployment``, alongside the default flexible-wage /
full-employment closure. A **KL-E-M energy nest** (Phase 5d.5, all three variants) makes energy a
separable, substitutable input (opt-in via ``energy_sectors``), so a carbon price shifts
substitution within the energy bundle rather than only across sectors — the shared CES algebra is
in ``energy_nest.py`` (one nest per region in the multi-region variant). An optional
**adaptation/transition investment** earmark (Phase 5d.6, closed variant, via
``adaptation_investment``) crowds out ordinary investment from the same savings pool. The
capital-accumulation identity (5d.3) lives in ``capital.py`` (a standalone module for Phase 7.1,
not wired into the solve).

Data contract (``data`` dict): either a ``SAM`` supplied directly (validated: aligned, finite,
non-negative, balanced) with an optional per-sector dimensionless ``carbon_cost_share``, OR an
``IOSystem`` (+ ``SatelliteAccount``) — a real build — from which the SAM is built + quality-gated
and the carbon cost is derived the SAME way as Engine 1 (gases, coverage, and the 1e-6 M→currency
scaling). Emission provenance (satellite identity + effective cost-share hash) is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.contracts.data_objects import SAM, IOSystem, SatelliteAccount
from cge.contracts.engine import Capability, EngineMeta, registry
from cge.contracts.provenance import RunManifest, content_hash, data_source_id, input_identity
from cge.contracts.results import ResultSet
from cge.contracts.shocks import CarbonPrice, ProductivityShock, Shock
from cge.engines.cge_static import model as M
from cge.engines.cge_static.calibrate import calibrate
from cge.engines.cge_static.solver import solve

VERSION = "0.13.0"

# Default factor accounts for the pilot SAM (capital, labour). The engine treats every SAM
# account that is neither a factor nor an institution as a sector.
_DEFAULT_FACTORS = ("CAP", "LAB")

# A government institution is recognised BY NAME (Phase 5d.1) — the same explicit-account
# convention the engine already uses for variant dispatch (``ROW`` selects the open variant,
# several ``HOH_<r>`` select multi-region). Closed variant only for now; a GOV account in an
# open/multi-region SAM is rejected by those variants' own account validation.
_GOV_ACCOUNT = "GOV"

# The savings-investment institution (Phase 5d.2), also recognised by name. Closed variant only
# for now (open/multi generalisation — where foreign savings route into the investment pool
# instead of household income — is the remaining 5d.2 work).
_SAVINV_ACCOUNT = "SAVINV"

# Strict per-variant configuration contract (review P1 rounds 13/14/16). The promise is: EVERY
# accepted public ``data`` key affects the SELECTED (variant, entry) path — no key is accepted only
# to be ignored (that would make the recorded ``effective_config`` misleading). Keys starting with
# ``_`` are engine-internal and rejected from public input entirely (see ``run``). Round-16: the
# earlier union-of-shared-sets construction was too broad — it accepted IO-only controls across
# every IO variant and ``carbon_cost_share`` / ``sectors`` on paths that ignore them. So the allowed
# set is now an EXPLICIT table per (variant, entry), each entry listing exactly the keys that path
# reads.
#
# What each key drives, and hence where it is allowed:
#   SAM                    — the supplied-SAM entry only.
#   IOSystem               — the IO-backed entry only.
#   SatelliteAccount       — the satellite carbon source the engine READS; IO entry only (a supplied
#                            SAM takes its carbon from carbon_cost_share, never a satellite).
#   satellites             — the store's raw bundle of ALL satellites (a DATA passthrough the
#                            loader attaches alongside IOSystem, NOT a control the engine reads —
#                            the engine reads SatelliteAccount). Accepted on the IO entry as a data
#                            input, excluded from the recorded effective_config.
#   carbon_cost_share      — the supplied-SAM carbon wedge; on the IO entry the wedge comes from
#                            the satellite, so it is IGNORED there → not allowed on IO.
#   va_elast/energy_*      — model controls every variant's calibrator reads (both entries).
#   sectors                — an explicit sector ORDER: read by the supplied-SAM resolver (any
#                            variant) and by the multi IO build; the closed/open IO build infers its
#                            own sectors, so a hint there is ignored → not allowed.
#   regions                — multi only (supplied multi validates it; multi IO reads it).
#   capital_share          — the build VA split — IO entry only.
#   multi_materiality      — the dust threshold — multi IO only.
#   open_home_region       — open IO build dispatch — open IO only.
#   multi_region           — multi IO build dispatch flag — multi IO only.
#   closure controls       — per variant (gov/inv/labour/adaptation/trade/elasticities), both
#                            entries of that variant.
_MODEL_CONTROLS = frozenset({"va_elast", "energy_sectors", "energy_elasticities"})
_CLOSURE_KEYS = {
    "closed": frozenset(
        {
            "gov_closure",
            "carbon_revenue_recipient",
            "inv_closure",
            "labour_floor",
            "adaptation_investment",
        }
    ),
    "open": frozenset({"inv_closure", "armington_elast", "cet_elast", "trade_closure"}),
    "multi": frozenset({"inv_closure", "armington_elast", "cet_elast", "regions"}),
}


def _build_allowed_table() -> dict[tuple[str, str], frozenset[str]]:
    """The EXPLICIT allowed-key table: (variant, entry) → the exact public keys that path READS.
    A key is listed ONLY if that path consumes it — no key is accepted merely to be ignored (review
    P1 rounds 16/17). Notably: the multi IO build DERIVES its own regions/sectors axes and
    OVERWRITES any hint, so ``sectors``/``regions`` are NOT accepted there; ``satellites`` is a data
    passthrough (the engine reads ``SatelliteAccount``) recorded as a data input, not a control."""
    tbl: dict[tuple[str, str], frozenset[str]] = {}
    for variant in ("closed", "open", "multi"):
        common = _MODEL_CONTROLS | _CLOSURE_KEYS[variant]
        # Supplied-SAM entry: SAM + supplied carbon wedge + sector/region hints the resolver reads.
        supplied = {"SAM", "carbon_cost_share"} | common | {"sectors"}
        if variant == "multi":
            supplied = supplied | {"regions"}  # already in closure keys, explicit for clarity
        # IO entry: IOSystem (+ the SatelliteAccount read + the store's raw `satellites` bundle
        # passthrough) + build controls; NO carbon_cost_share (satellite-based), and NO
        # sectors/regions on multi (the builder derives and overwrites them).
        io = {"IOSystem", "SatelliteAccount", "satellites", "capital_share"} | common
        if variant == "open":
            io = io | {"open_home_region"}
        if variant == "multi":
            io = (io | {"multi_region", "multi_materiality"}) - {"regions"}
        tbl[(variant, "supplied")] = frozenset(supplied)
        tbl[(variant, "io")] = frozenset(io)
    return tbl


_ALLOWED_CONFIG = _build_allowed_table()
# Data-input keys (kept for _effective_public_config to exclude from the recorded policy surface).
# ``satellites`` is the store's raw multi-satellite bundle — a data passthrough, not a policy
# control — so it is excluded from the effective config too.
_DATA_INPUT_KEYS = frozenset({"SAM", "IOSystem", "SatelliteAccount", "satellites"})
_ENUM_CONFIG = {
    "gov_closure": ("balanced_budget", "deficit_financed"),
    "inv_closure": ("savings_driven", "fixed_real"),
    "carbon_revenue_recipient": ("government", "household"),
    "trade_closure": ("fixed_foreign_savings", "flexible_balance"),
}


def _allowed_config_keys(variant: str, entry: str) -> set[str]:
    """The PUBLIC config keys a (variant, entry) combination may carry — an EXPLICIT per-cell table
    (review P1 round 16), so no key is accepted only to be ignored on the selected path."""
    return set(_ALLOWED_CONFIG[(variant, entry)])


def _validate_config(data: dict, variant: str, entry: str) -> None:
    """Strict, ENTRY-PATH-SPECIFIC config gate (review P1 rounds 13/14). Reject, BEFORE calibration:
    (1) any non-internal ``data`` key the (variant, entry) does not USE — a control another variant
    supports, OR an IO-build key (``capital_share``/``open_home_region``/...) on a SUPPLIED SAM
    where it is silently ignored; (2) any enum control with an invalid value; (3) an out-of-range
    ``capital_share`` or an ``open_home_region`` not in the build. Keys starting with ``_`` are
    engine-internal (IO-path injected state) and always allowed — they are set by the engine, never
    by a scenario file. This makes the manifest's recorded config truthful."""
    # Ambiguous entry (review P1 round 15): a SUPPLIED SAM takes precedence and the IOSystem is
    # silently ignored — so a run carrying BOTH is under-specified (which one produced the result?).
    # Checked FIRST so it gives its clear message rather than a generic "unsupported key: IOSystem".
    if entry == "supplied" and data.get("IOSystem") is not None:
        raise ValueError(
            "both a supplied 'SAM' and an 'IOSystem' were provided: the supplied SAM would be used "
            "and the IOSystem silently ignored. Supply exactly one entry (a SAM, or an IOSystem to "
            "build from) so the run's provenance is unambiguous."
        )
    allowed = _allowed_config_keys(variant, entry)
    unknown = sorted(k for k in data if not k.startswith("_") and k not in allowed)
    if unknown:
        raise ValueError(
            f"unsupported config key(s) for the {variant!r} CGE variant on the {entry!r} entry "
            f"path: {unknown}. These are either unknown, supported only by another variant, or "
            f"(for capital_share / axis / build-dispatch keys) meaningful only on an IO-backed "
            f"build — not a supplied SAM. Allowed here: {sorted(allowed)}."
        )
    for key, choices in _ENUM_CONFIG.items():
        if key in data and data[key] not in choices:
            raise ValueError(
                f"invalid {key}={data[key]!r} for the {variant!r} variant; use one of {choices}."
            )
    # Value checks for the IO-build controls.
    if "capital_share" in data:
        cs = data["capital_share"]
        if not isinstance(cs, (int, float)) or not (0.0 < float(cs) < 1.0):
            raise ValueError(f"capital_share must be a number in (0, 1); got {cs!r}.")
    if entry == "io" and variant == "open" and "open_home_region" in data:
        io = data.get("IOSystem")
        if io is not None and data["open_home_region"] not in list(io.regions.labels):
            raise ValueError(
                f"open_home_region={data['open_home_region']!r} is not a region of the build "
                f"{list(io.regions.labels)}."
            )
    # Axis hints on a SUPPLIED SAM must MATCH the SAM (review P1 round 15): a bogus 'sectors' or
    # 'regions' axis was previously accepted and could silently reorder/mislabel results. Validate
    # them against the SAM's own accounts rather than trusting the caller's list.
    if entry == "supplied":
        sam = data.get("SAM")
        if sam is not None and variant == "multi" and ("sectors" in data or "regions" in data):
            # Anchor on the UNAMBIGUOUS regions (the full suffix of each HOH_<r>, never split on a
            # further '_'), then require the claimed axes to REPRODUCE the SAM's a_<r>_<s> accounts
            # as a cross product. This validates a bogus 'regions'-only or 'sectors'-only hint too,
            # without brittle string-splitting of names that may contain '_' (review P1 round 15).
            true_regions = sorted({a[len("HOH_") :] for a in sam.accounts if a.startswith("HOH_")})
            acts = {a for a in sam.accounts if a.startswith("a_")}
            claimed_r = (
                sorted(str(r) for r in data["regions"]) if "regions" in data else true_regions
            )
            if "regions" in data and claimed_r != true_regions:
                raise ValueError(
                    f"the supplied 'regions' axis {claimed_r} does not match the multi-region "
                    f"SAM's regions {true_regions} (from its HOH_<r> households); a mismatched "
                    "axis silently reorders/mislabels results — omit the hint or supply the SAM's."
                )
            if "sectors" in data:
                claimed_s = [str(s) for s in data["sectors"]]
                expected = {f"a_{r}_{s}" for r in true_regions for s in claimed_s}
                if expected != acts:
                    raise ValueError(
                        f"the supplied 'sectors' axis {sorted(claimed_s)} does not reproduce the "
                        f"multi-region SAM's activity accounts (expected {sorted(expected)[:3]}…, "
                        f"SAM has {sorted(acts)[:3]}…); a mismatched axis silently reorders/"
                        "mislabels results — omit the hint or supply the SAM's own sectors."
                    )
        elif sam is not None and "sectors" in data:
            factors = [f for f in _DEFAULT_FACTORS if f in sam.accounts]
            # If the SAM's matrix axes are themselves inconsistent with its accounts, let the
            # deeper _validate_supplied_sam raise its clean error rather than tripping on a KeyError
            # here — _infer_sectors reads sam.matrix.loc, which a malformed SAM cannot satisfy.
            try:
                actual = sorted(_infer_sectors(sam, factors))
            except (KeyError, ValueError):
                actual = None
            if actual is not None and sorted(str(s) for s in data["sectors"]) != actual:
                raise ValueError(
                    f"the supplied 'sectors' axis {sorted(str(s) for s in data['sectors'])} does "
                    f"not match the SAM's sectors {actual}; a mismatched axis silently "
                    "reorders/mislabels results — supply the SAM's own sectors or omit the hint."
                )

    # CROSS-FIELD applicability (review P1 round 17): a control that only takes effect together with
    # another is rejected when the other is absent — a static per-key allow-list cannot express
    # this, so it must be checked explicitly, or the recorded effective_config claims a control had
    # effect when it did not.
    # (a) energy_elasticities parameterise the energy nest; without energy_sectors there is no nest.
    if "energy_elasticities" in data and not data.get("energy_sectors"):
        raise ValueError(
            "energy_elasticities was supplied without energy_sectors: the elasticities "
            "parameterise the KL-E-M energy nest, which is inactive unless energy_sectors names "
            "the energy commodities — so they have no effect. Supply energy_sectors, or drop the "
            "elasticities."
        )
    # (a′) energy_elasticities keys must be EXACTLY the three nest elasticities the calibrator reads
    # (kle_m/kl_e/energy) — anything else (e.g. a typo) is silently dropped by the ``el.get(name)``
    # lookups and recorded as effective when it had no effect (review P2 round 18).
    if isinstance(data.get("energy_elasticities"), dict):
        allowed_elast = {"kle_m", "kl_e", "energy"}
        bad = sorted(k for k in data["energy_elasticities"] if k not in allowed_elast)
        if bad:
            raise ValueError(
                f"energy_elasticities has unknown key(s) {bad}; the KL-E-M nest reads exactly "
                f"{sorted(allowed_elast)} (a mistyped key is silently ignored and would be "
                "recorded as effective though it had no effect)."
            )
    # (b) carbon_revenue_recipient only bites when a government account exists. On a supplied SAM we
    # can see this directly (a GOV account); reject the control if there is none, so the manifest
    # does not record a recipient choice that the run reports as 'n/a (no government)'. On an IO
    # build the government comes from the institution split, which is not resolved until build time,
    # so it is accepted there (and recorded honestly by the run).
    if "carbon_revenue_recipient" in data and entry == "supplied":
        sam = data.get("SAM")
        if sam is not None and _GOV_ACCOUNT not in sam.accounts:
            raise ValueError(
                "carbon_revenue_recipient was supplied but the SAM has no government (GOV) "
                "account, so the choice has no effect (revenue recycles to the household). Add a "
                "GOV account or drop carbon_revenue_recipient."
            )


def _effective_public_config(data: dict, variant: str, entry: str) -> dict:
    """The complete EFFECTIVE public configuration for the manifest (review P1 round 14): every
    validated public control the run actually uses, with its effective value (the supplied value or
    the documented default). Data inputs and the SAM itself are recorded elsewhere; this is the
    policy/parameter surface, so a reader can see exactly what closure/parameters produced the run.
    """
    keys = _allowed_config_keys(variant, entry) - _DATA_INPUT_KEYS
    defaults = {
        "gov_closure": "balanced_budget",
        "inv_closure": "savings_driven",
        "carbon_revenue_recipient": "government",
        "trade_closure": "fixed_foreign_savings",
        "capital_share": 0.4 if entry == "io" else None,  # build.DEFAULT_CAPITAL_SHARE
    }
    cfg = {"variant": variant, "entry": entry}
    for k in sorted(keys):
        if k in data:
            v = data[k]
            if isinstance(v, (str, int, float, bool)):
                cfg[k] = v
            else:
                # Non-scalar controls (e.g. adaptation_investment's sector→share vector,
                # energy_elasticities) were previously collapsed to the opaque string "supplied",
                # which hid the COMPOSITION: two runs allocating the same total adaptation amount to
                # DIFFERENT sectors produced different results yet identical manifests (review P1
                # round 17). Record a CONTENT HASH (+ a small readable summary) so the effective
                # config — and the scenario hash derived from it — distinguishes them.
                cfg[k] = {"content_hash": content_hash(v), "summary": _summarise_config_value(v)}
        elif k in defaults and defaults[k] is not None:
            cfg[k] = defaults[k]
    return cfg


def _summarise_config_value(v):
    """A short, JSON-safe readable summary of a non-scalar config value for the manifest (the
    content hash is authoritative; this is the human-readable companion)."""
    if isinstance(v, dict):
        return {
            str(k): (round(float(x), 6) if isinstance(x, (int, float)) else str(x))
            for k, x in list(v.items())[:16]
        }
    if isinstance(v, (list, tuple)):
        return [round(float(x), 6) if isinstance(x, (int, float)) else str(x) for x in v[:16]]
    return str(v)[:120]


ASSUMPTIONS = {
    "model": (
        "static CGE pilot: Leontief intermediates + CES/Cobb-Douglas value added and "
        "Cobb-Douglas household demand; fixed factor endowments; CPI numéraire"
    ),
    "scope": (
        "single region, one representative household; revenue recycling supported "
        "(none/lump_sum/labour_tax_cut, the last two equivalent with one household). An optional "
        "GOVERNMENT account (Phase 5d.1, all variants): a GOV SAM account (GOV_<r> per region in "
        "the multi-region variant) makes government a real institution collecting carbon revenue "
        "(+ an optional benchmark direct tax) and spending on its own calibrated demand vector "
        "under a balanced budget — see the government_account/gov_closure keys. This is the "
        "CLOSED variant; an OPEN variant "
        "(Armington/CET + a rest-of-world account) runs when the SAM carries a ROW account, and a "
        "true MULTI-REGION variant (bilateral trade among several region-tagged households) runs "
        "when the SAM carries multiple households — see OPEN_ASSUMPTIONS / MULTI_ASSUMPTIONS for "
        "those variants' own scope text."
    ),
    "carbon_price": "per-unit emissions cost wedge τ·e[i] in the zero-profit condition",
    "productivity": (
        "per-sector Hicks-neutral productivity multiplier θ[i] (Phase 6.4 GE tier): a nature "
        "degradation, translated by the exposure engine into ProductivityShocks, divides sector "
        "i's "
        "zero-profit unit cost by θ[i] (θ<1 raises its price and reallocates production in the GE "
        "solve). θ=1 (no shock) is byte-identical to a pure carbon run. Closed variant; the open/"
        "multi variants are a follow-up."
    ),
    "revenue_recycling": (
        "carbon revenue R = Σ τ·e[i]·X[i] is returned to the household (lump_sum/labour_tax_cut). "
        "Established: CD welfare falls only slightly under a recycled carbon price, and at those "
        "prices the transfer raises utility. NOT established: a full GE welfare comparison against "
        "a valid no-recycling closure (the `none` mode does not close — it violates Walras' law — "
        "so it is not a valid counterfactual; a government/external account is a follow-up)."
    ),
    "closure": (
        "numéraire = the exact Cobb-Douglas consumer price index (Π p_i^γ_i = 1), the unit of "
        "account, so no ABSOLUTE inflation is identified; the reported GDP deflator is a RELATIVE "
        "index (GDP basket vs the CPI numéraire). Factor SUPPLY is fixed. Factor-market closure: "
        "flexible wage / full "
        "employment by default (see labour_closure); an optional labour_floor switches on a wage "
        "floor with involuntary unemployment (Phase 5d.4). Savings/investment and a government "
        "account are opt-in via SAM accounts (Phase 5d.1/5d.2; see the government_account / "
        "savings_investment_account keys)."
    ),
    "solver_rule": (
        "non-optimal solve raises (well-posedness); solver backend, termination status, and "
        "max residual norm recorded in the manifest"
    ),
    "interpretation": (
        "GENERAL-EQUILIBRIUM price and volume response with factor-market feedback and input "
        "substitution via the value-added nest — the mechanism Engines 1-2 cannot capture. "
        "Indicative magnitudes (pilot calibration); brackets Engine 1 prices, same-sign Engine 2 "
        "volumes."
    ),
    "reference": "Hosoe, Gasawa & Hashimoto (2010), Textbook of CGE Modeling [Hosoe2010]",
}


def _va_nest_description(va_elast: np.ndarray) -> str:
    """Describe the value-added nest that was actually calibrated (review P2: the manifest said
    'Cobb-Douglas' even for a CES run). σ_va = 1 everywhere ⇒ Cobb-Douglas; otherwise CES."""
    if np.all(np.abs(np.asarray(va_elast) - 1.0) < 1e-12):
        return "Cobb-Douglas value added (σ_va = 1)"
    return (
        "CES value added (non-unitary σ_va ⇒ capital/labour substitution as relative factor "
        "prices move; the values are in the 'va_elast' key)"
    )


def _energy_manifest(cal, sectors: list[str]) -> dict:
    """Energy-nest manifest keys (Phase 5d.5), shared by every variant: flat Leontief production or
    the KL-E-M nest with the named energy commodities + its elasticities. Two runs differing only
    in the nest / its elasticities must therefore differ in assumptions. Handles both the
    single-region ``cal.energy_nest`` and the multi-region ``cal.energy_nests`` (per region — the
    elasticities are the same across regions, so region 0's are representative)."""
    if not cal.has_energy_nest:
        return {
            "production_structure": "flat Leontief intermediates",
            "energy_sectors": [],
            "energy_nest_elasticities": None,
        }
    nest = cal.energy_nests[0] if hasattr(cal, "energy_nests") else cal.energy_nest
    return {
        "production_structure": "KL-E-M energy nest",
        "energy_sectors": [sectors[j] for j in nest.energy_idx.tolist()],
        "energy_nest_elasticities": {
            "kle_m": round(float(nest.kle_m_elast[0]), 12),
            "kl_e": round(float(nest.kl_e_elast[0]), 12),
            "energy": round(float(nest.energy_elast[0]), 12),
        },
    }


# Assumptions for the OPEN-economy variant. It shares the solver rule and reference but has a
# genuinely different model (Armington/CET, separate activity/commodity accounts, CES value added,
# an exchange rate) — recording the closed assumptions verbatim would misdescribe an open run
# (review P1/P3). Only keys that actually differ are overridden here; the rest are inherited.
OPEN_ASSUMPTIONS = {
    **ASSUMPTIONS,
    "model": (
        "static open-economy CGE: Leontief intermediates over an Armington composite commodity; "
        "value added a CES (or Cobb-Douglas) nest over factors; CET transformation of output into "
        "domestic sales and exports; small open economy (world prices + foreign savings fixed, "
        "exchange rate endogenous); CPI numéraire"
    ),
    "model_variant": (
        "open economy (Armington imports + CET exports) with a rest-of-world account; single "
        "region + RoW (true multi-region is a follow-up); a non-zero current account is supported "
        "— foreign savings Sf = Σimports−Σexports is fixed at its benchmark level and enters "
        "household income as er·Sf (the ROW capital transfer)"
    ),
    "scope": (
        "single region + rest-of-world, one representative household; revenue recycling to the "
        "household; Armington/CET trade IS modelled here; true multiple regions are a later phase"
    ),
    "trade": (
        "Armington import composite (elasticity σ) and CET export transformation (elasticity Ω) "
        "per sector; import/export world prices fixed at 1 in foreign currency; the exchange rate "
        "clears the fixed-foreign-savings trade balance; a carbon price causes carbon leakage"
    ),
    "value_added": (
        "CES value-added nest with per-sector substitution elasticity σ_va (σ_va=1 ⇒ "
        "Cobb-Douglas); non-unitary σ_va enables capital/labour substitution. NOTE: this is NOT a "
        "double-dividend model — there is no distortionary labour-tax wedge, and with one "
        "household labour_tax_cut recycling is allocation-equivalent to lump_sum (the "
        "double-dividend channel needs heterogeneous households or a labour-tax distortion — a "
        "documented follow-up)"
    ),
    "interpretation": (
        "GENERAL-EQUILIBRIUM open-economy price and volume response with factor-market feedback, "
        "input substitution and trade reallocation (imports/exports respond to relative prices). "
        "Indicative magnitudes (pilot calibration)."
    ),
}

# Assumptions for the MULTI-REGION variant (true bilateral trade between build regions).
MULTI_ASSUMPTIONS = {
    **OPEN_ASSUMPTIONS,
    "model": (
        "static multi-region CGE: R regions trading bilaterally in a closed global economy; per "
        "region Leontief intermediates over an Armington composite (CES over the domestic variety "
        "+ imports from every partner region), CES/Cobb-Douglas value added, and a CET transform "
        "of output into domestic sales + exports to every partner; region-specific factors; global "
        "CPI numéraire"
    ),
    "model_variant": (
        "multi-region (bilateral Armington/CET between build regions); region-specific immobile "
        "factors and one household per region; each trade route o→d has its own price "
        "pe[o,s,d] and every bilateral goods market clears explicitly (M[d,s,o]=EX[o,s,d]); "
        "foreign savings per region fixed at benchmark and globally zero-sum"
    ),
    "trade": (
        "bilateral Armington imports (each region's composite is a CES over its domestic "
        "variety + imports from every partner) and CET exports (output split over domestic "
        "sales + exports to every partner); a **destination-specific price** on each route "
        "clears each bilateral goods market; a carbon price in one region causes cross-region "
        "leakage"
    ),
    "closure": (
        "region-specific fixed factor supply; one global numéraire (region-0's CPI, "
        "Π pq[0,s]^γ=1); no external rest-of-world and no exchange rate (a closed global "
        "economy of the build regions); one factor market dropped by Walras' law"
    ),
    "scope": (
        "R regions with bilateral trade, one representative household per region; carbon pricing "
        "in one region causes CROSS-REGION leakage (production/imports shift to partner regions)"
    ),
    "interpretation": (
        "GENERAL-EQUILIBRIUM multi-region price/volume response with bilateral trade reallocation: "
        "a carbon price in one region relocates production and raises imports from partners "
        "(cross-region carbon leakage). Every bilateral goods and factor market clears (verified). "
        "Indicative magnitudes (pilot calibration)."
    ),
}


def _adaptation_spec(data: dict, sectors: list[str], cal, inv_closure: str):
    """Parse ``data['adaptation_investment']`` (Phase 5d.6) into ``(amount, gamma)`` for the model.

    The value is a ``dict[sector, float]`` of adaptation/transition capex **as a share of benchmark
    GDP** (the same normalisation as every other reported quantity — the benchmark GDP is 1 after
    calibration). Returns the total nominal amount and the normalised sectoral composition
    (``gamma``, summing to 1). Absent ⇒ ``(0.0, None)`` (no adaptation, pre-5d.6 behaviour).

    Adaptation crowds out ordinary investment from the same savings pool, so it needs a
    savings-investment account and the ``savings_driven`` closure — rejected loudly otherwise, not
    silently ignored."""
    spec = data.get("adaptation_investment")
    if not spec:
        return 0.0, None
    unknown = [s for s in spec if s not in sectors]
    if unknown:
        raise ValueError(f"adaptation_investment sectors {unknown} are not in {sectors}")
    vec = np.array([float(spec.get(s, 0.0)) for s in sectors])
    if np.any(vec < 0) or not np.all(np.isfinite(vec)):
        raise ValueError("adaptation_investment values must be finite and non-negative")
    amount = float(vec.sum())
    if amount <= 0:
        return 0.0, None
    if not cal.has_investment:
        raise ValueError(
            "adaptation_investment needs a savings-investment account (a SAVINV SAM account, "
            "Phase 5d.2) to crowd out; none is present."
        )
    if inv_closure != "savings_driven":
        raise ValueError(
            f"adaptation_investment is only modelled under the savings_driven closure; got "
            f"{inv_closure!r} (it crowds out ordinary investment from the same savings pool)."
        )
    return amount, vec / amount


def _carbon_cost_share(data: dict, sectors: list[str]) -> np.ndarray | None:
    """Per-sector **dimensionless carbon cost-share** supplied directly (the toy pilot path).

    Reads ``data['carbon_cost_share']`` — the cost added to a sector's unit price *per €1 of carbon
    price*, i.e. already the τ=1 cost wedge (dimensionless, 1e-6-scaled if it came from real data).
    The engine multiplies it by the scenario's τ. This replaces the old raw ``emission_intensity``
    (t/MEUR), which was applied *without* the M€→€ conversion and so was ~1e6 too large (review P0).
    Returns None when absent (a real-build run derives the cost from the satellite via Engine 1)."""
    ei = data.get("carbon_cost_share")
    if ei is None:
        return None
    if isinstance(ei, dict):
        # Keys must belong to the declared sectors — a typo would otherwise be silently dropped
        # (review P1). Values must be finite and non-negative (a negative share is an undocumented
        # subsidy: it lowers the dirty-sector price and generates negative revenue).
        unknown = [k for k in ei if k not in sectors]
        if unknown:
            raise ValueError(f"carbon_cost_share keys not in the SAM sectors {sectors}: {unknown}")
        arr = np.array([float(ei.get(s, 0.0)) for s in sectors])
    else:
        arr = np.asarray(ei, dtype=float)
        if arr.shape != (len(sectors),):
            raise ValueError(
                f"carbon_cost_share must have one value per sector ({len(sectors)}), "
                f"got shape {arr.shape}"
            )
    if not np.isfinite(arr).all():
        raise ValueError("carbon_cost_share values must be finite")
    if float(arr.min()) < 0.0:
        raise ValueError(
            "carbon_cost_share values must be non-negative (a negative share is a carbon subsidy, "
            "not a price; it would lower the dirty-sector price and generate negative revenue)"
        )
    return arr


class CGEStaticEngine:
    """Static CGE pilot. Satisfies the ``Engine`` protocol."""

    meta = EngineMeta(
        name="cge_static",
        version=VERSION,
        description="Static CGE pilot: GE price + volume response with factor-market feedback.",
        capabilities=[Capability.GENERAL_EQUILIBRIUM, Capability.PRICES, Capability.VOLUMES],
        supported_shocks=["carbon_price", "productivity"],
        # Accepts either a supplied SAM (toy pilot) or an IOSystem (a real build, from which the
        # SAM is built + quality-gated). Validated in _resolve_sam, so no hard required_data here.
        required_data=[],
    )

    def run(self, *, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
        # SECURITY BOUNDARY (review P1 round 15): underscore-prefixed keys are ENGINE-INTERNAL state
        # (``_io_backed``, ``_effective_cc_by_year``, ``_emission_intensity``, ``_sam_quality``,
        # ``_emissions_provenance``, …) that the IO-backed build paths inject into a FRESH ``inner``
        # dict AFTER dispatch. A public caller must never be able to supply them: doing so let a
        # supplied-SAM run forge an IO-backed carbon wedge and mislabel its own provenance. So
        # reject ANY ``_``-prefixed key in the incoming ``data`` up front — legitimate callers never
        # one (run_scenario / data_overrides / the store loaders write only public keys). This runs
        # before dispatch so it cannot be bypassed by the variant/entry choice.
        injected = sorted(k for k in data if str(k).startswith("_"))
        if injected:
            raise ValueError(
                f"reserved engine-internal config key(s) supplied by caller: {injected}. Keys "
                "starting with '_' are injected by the engine's own IO-backed build paths and must "
                "not appear in scenario data / data_overrides (they would forge IO-backed state or "
                "provenance on a run that did not build from an IOSystem)."
            )
        # Dispatch by SAM structure: MULTI-REGION (bilateral trade — several HOH_<r> households) →
        # _run_multi; OPEN (a single ROW account) → _run_open; an IOSystem + open_home_region builds
        # an open SAM; an IOSystem + multi_region=True builds an R-region SAM (Phase 5.1b);
        # otherwise the CLOSED pilot. Each variant has its own calibration/model.
        supplied_sam = data.get("SAM")
        if supplied_sam is not None and _is_multi_region_sam(supplied_sam):
            variant, entry = "multi", "supplied"
        elif supplied_sam is not None and "ROW" in supplied_sam.accounts:
            variant, entry = "open", "supplied"
        elif supplied_sam is not None:
            variant, entry = "closed", "supplied"
        elif data.get("IOSystem") is not None and data.get("open_home_region") is not None:
            variant, entry = "open", "io"  # IO-backed open (build_open_sam)
        elif data.get("IOSystem") is not None and data.get("multi_region"):
            variant, entry = "multi", "io"  # IO-backed multi (build_multi_sam)
        else:
            variant, entry = "closed", "io"  # IO-backed closed (build_sam)
        # Strict, entry-path-specific config contract (review P1 rounds 13/14): reject unknown keys,
        # keys meaningful only on another variant or only on an IO build, invalid enum values, and
        # out-of-range values — BEFORE calibration — so a scenario file cannot silently request a
        # closure/param that is never executed.
        _validate_config(data, variant, entry)

        # Boundary integrity guard for EVERY IO-backed entry (review P1 round 17). The closed IO
        # path checked this inside _resolve_inputs (assert_io_aligned), but the open and multi IO
        # paths dispatched straight to their builders — so a caller who MUTATED io.A (e.g. reversed
        # its rows) after construction slipped past the contract's construction-time validator and
        # got plausible-but-wrong results with a passing SAM quality report. Re-run the FULL
        # IOSystem integrity check (A alignment, final-demand shape, institution-split alignment and
        # per-cell consistency) here so all three IO variants are guarded identically.
        if entry == "io":
            io = data.get("IOSystem")
            if io is not None:
                io.assert_integrity()

        if variant == "multi":
            if supplied_sam is not None and _is_multi_region_sam(supplied_sam):
                return _run_multi(self.meta, data, shocks, years)
            return _run_multi_from_io(self.meta, data, shocks, years)
        if variant == "open":
            if supplied_sam is not None:
                return _run_open(self.meta, data, shocks, years)
            return _run_open_from_io(self.meta, data, shocks, years)

        inp = _resolve_inputs(data)
        sam, sectors, factors = (
            inp.sam,
            inp.sectors,
            [f for f in _DEFAULT_FACTORS if f in inp.sam.accounts],
        )
        # Government account (Phase 5d.1) and savings-investment account (Phase 5d.2), both
        # recognised by name: GOV collects carbon revenue (and any benchmark direct tax) and
        # spends on its own calibrated demand vector under a balanced budget; SAVINV turns
        # household savings into investment demand under the chosen closure. The household is the
        # remaining institution.
        special = {_GOV_ACCOUNT, _SAVINV_ACCOUNT} & set(sam.accounts)
        institutions = None
        if special:
            others = [
                a
                for a in sam.accounts
                if a not in sectors and a not in factors and a not in special
            ]
            if len(others) != 1:
                raise ValueError(
                    f"a SAM with {sorted(special)} accounts needs exactly one household account "
                    f"besides them; got {others}"
                )
            institutions = {"household": others[0]}
            if _GOV_ACCOUNT in special:
                institutions["government"] = _GOV_ACCOUNT
            if _SAVINV_ACCOUNT in special:
                institutions["savings_investment"] = _SAVINV_ACCOUNT
        inv_closure = data.get("inv_closure", "savings_driven")
        if inv_closure not in ("savings_driven", "fixed_real"):
            raise ValueError(
                f"unsupported inv_closure {inv_closure!r}; use 'savings_driven' or 'fixed_real'."
            )
        # Government financing closure (Phase 5d.1 / 5d.7): balanced_budget (default) or
        # deficit_financed (fixed real government spending; the deficit crowds out private
        # investment from the savings pool — review remediation 2026-07-26, a real financing
        # channel, not the earlier household-residual pass-through that cancelled the tax out).
        gov_closure = data.get("gov_closure", "balanced_budget")
        if gov_closure not in ("balanced_budget", "deficit_financed"):
            raise ValueError(
                f"unsupported gov_closure {gov_closure!r}; use 'balanced_budget' or "
                "'deficit_financed'."
            )
        # Carbon-revenue recipient (review P1, 2026-07-27): who receives carbon revenue is a
        # SEPARATE choice from the fiscal closure. Default 'government' (consistent with
        # balanced_budget, where revenue funds government spending). 'household' recycles it to the
        # household. Only meaningful under deficit_financed with a government; balanced_budget
        # always routes revenue to the government budget by construction.
        carbon_revenue_recipient = data.get("carbon_revenue_recipient", "government")
        if carbon_revenue_recipient not in ("government", "household"):
            raise ValueError(
                f"unsupported carbon_revenue_recipient {carbon_revenue_recipient!r}; use "
                "'government' (default) or 'household'."
            )
        # Energy nest (Phase 5d.5): opt-in via ``energy_sectors`` (which commodities are energy),
        # with optional per-layer elasticities. With none declared, production stays flat Leontief
        # (bit-identical to pre-5d.5).
        cal = calibrate(
            sam,
            sectors=sectors,
            factors=factors,
            va_elast=data.get("va_elast", 1.0),
            institutions=institutions,
            energy_sectors=data.get("energy_sectors"),
            energy_elasticities=data.get("energy_elasticities"),
        )
        if inv_closure != "savings_driven" and not cal.has_investment:
            raise ValueError(
                f"inv_closure={inv_closure!r} needs a {_SAVINV_ACCOUNT!r} account in the SAM; "
                "this SAM has none, so the closure choice would silently do nothing."
            )
        if gov_closure != "balanced_budget" and not cal.has_government:
            raise ValueError(
                f"gov_closure={gov_closure!r} needs a {_GOV_ACCOUNT!r} account in the SAM; this "
                "SAM has none, so the closure choice would silently do nothing."
            )
        # POST-BUILD applicability of carbon_revenue_recipient (review P2 round 18). The
        # construction-time check only sees a SUPPLIED SAM's accounts; on an IO build the government
        # comes from the institution split and is not known until here. If the caller EXPLICITLY set
        # a recipient but the built SAM has no government, the choice is a no-op (revenue recycles
        # to the household regardless) — reject so the effective_config never records a recipient
        # the run reports as 'n/a (no government)'.
        if "carbon_revenue_recipient" in data and not cal.has_government:
            raise ValueError(
                "carbon_revenue_recipient was supplied but the built SAM has no government (GOV) "
                "account, so the choice has no effect (revenue recycles to the household). Provide "
                "an institution split with a government, or drop carbon_revenue_recipient."
            )
        if gov_closure == "deficit_financed" and not cal.has_investment:
            # The deficit is financed by crowding out private investment (review remediation
            # 2026-07-26); with no savings-investment account there is no financing channel.
            raise ValueError(
                f"gov_closure='deficit_financed' needs a {_SAVINV_ACCOUNT!r} account in the SAM — "
                "the deficit is financed by crowding out private investment; this SAM has none."
            )
        # Labour-market closure (Phase 5d.4): default flexible-wage/full-employment; an optional
        # ``labour_floor`` (a wage floor, in benchmark CPI-numéraire units where the benchmark
        # wage = 1) switches on the wage-floor closure via a regime-switch in _solve.
        labour_floor = data.get("labour_floor")
        if labour_floor is not None and (not np.isfinite(labour_floor) or float(labour_floor) <= 0):
            raise ValueError(f"labour_floor must be a positive wage; got {labour_floor!r}.")
        # Adaptation/transition investment (Phase 5d.6): an exogenous, sector-tagged capex earmark
        # that crowds out ordinary investment from the same savings pool (under savings_driven).
        adapt_amount, adapt_gamma = _adaptation_spec(data, sectors, cal, inv_closure)
        ns = len(sectors)

        carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
        prod_shocks = [s for s in shocks if isinstance(s, ProductivityShock)]
        _assert_no_region_scoped_productivity(prod_shocks, "closed")
        _validate_cge_shock_controls(inp, carbon_shocks)
        # One government ⇒ one recycling rule; a scenario cannot mix modes.
        modes = {s.revenue_recycling for s in carbon_shocks} or {"none"}
        if len(modes) > 1:
            raise ValueError(
                f"cge_static needs a single revenue_recycling mode across carbon shocks; "
                f"got {sorted(modes)}."
            )
        recycling = modes.pop()

        # A positive carbon price needs an emissions input, or it silently becomes a zero-impact
        # run. Require the input whenever ANY year has a nonzero effective carbon price; tolerate a
        # missing input only for a genuine zero-price baseline (review P1).
        positive_price = any(s.price_at(y) > 0 for s in carbon_shocks for y in years)
        if positive_price:
            if inp.io is not None and inp.sat is None:
                raise ValueError(
                    "a positive carbon price on an IO-backed CGE run requires a 'SatelliteAccount' "
                    "(emission intensities); none supplied — the run would be silently zero-impact."
                )
            if inp.io is None and inp.cost_share is None:
                raise ValueError(
                    "a positive carbon price on a supplied-SAM CGE run requires a "
                    "'carbon_cost_share'; none supplied — the run would be silently zero-impact."
                )

        # Per-year carbon cost share (dimensionless, gas/coverage/units handled like Engine 1).
        cc_by_year = {y: _carbon_cost_by_sector(inp, carbon_shocks, y) for y in years}
        # Per-year per-sector productivity multiplier θ (Phase 6.4 GE tier). A nature degradation,
        # translated by the exposure engine into ProductivityShocks, enters the CGE here as a
        # supply-side technology hit that the general-equilibrium solve responds to (relative-price
        # and factor reallocation) — not just a first-round quantity change like Engine 2. None when
        # no productivity shock is present, so the run is byte-identical to a pure carbon run.
        theta_by_year = (
            {y: _productivity_by_sector(prod_shocks, inp.sectors, y) for y in years}
            if prod_shocks
            else None
        )
        emissions_priced = any(np.any(cc != 0.0) for cc, _ in cc_by_year.values())
        # Price-FREE emission intensity (review P1 round 14) — computed once, year-independent, so
        # covered emissions are emitted for EVERY year including a zero-price year.
        emission_intensity = _carbon_intensity_by_sector(inp, carbon_shocks)

        # A closed CGE cannot destroy carbon revenue (it breaks Walras' law). When a positive
        # carbon price would raise revenue but the scenario left recycling at the default `none`,
        # default to `lump_sum` (the standard closed-economy choice) and record it, rather than
        # solve a non-closing model. Engine 1 gives the pure price-side / no-recycling view.
        recycling_defaulted = False
        if recycling == "none" and emissions_priced:
            recycling = "lump_sum"
            recycling_defaulted = True

        # Benchmark solve (zero shock) — the replication point, and the base for % changes. The
        # labour floor is NOT applied here: the benchmark is the calibration point, full employment
        # at wage 1 by construction (that is what the SAM represents). The floor is a floor on the
        # POST-SHOCK wage — it only binds when a shock pushes the wage below it, so it is a well-
        # posed floor iff it is below 1 (a floor ≥ 1 would try to bind at the benchmark, which is
        # nonsensical; rejected up front so the run fails loudly rather than at the replication
        # gate).
        if labour_floor is not None and labour_floor >= 1.0 - 1e-12:
            raise ValueError(
                f"labour_floor={labour_floor} is ≥ the benchmark wage (1.0); a wage floor is a "
                "floor on the POST-SHOCK wage and must be below the benchmark wage to be "
                "meaningful (the benchmark is full employment at wage 1 by construction)."
            )
        base_sol, _ = _solve(
            cal,
            carbon_cost=np.zeros(ns),
            recycling="none",
            inv_closure=inv_closure,
            gov_closure=gov_closure,
            carbon_revenue_recipient=carbon_revenue_recipient,
        )
        base = M.derive_state(
            cal,
            base_sol.x[:ns],
            base_sol.x[ns:],
            inv_closure=inv_closure,
            gov_closure=gov_closure,
            carbon_revenue_recipient=carbon_revenue_recipient,
        )
        # Universal post-calibration replication gate (review P1): a balanced SAM can pass the
        # structural validators yet not conform to the implemented model (e.g. an unsupported
        # household↔commodity loop), in which case the benchmark does not reproduce the SAM and
        # every % change is silently measured against a wrong base. Assert the benchmark state
        # reproduces the calibrated quantities, or refuse the run.
        _assert_closed_replicates(cal, base, base_sol.x)

        records: list[dict] = []
        backends: set[str] = {base_sol.backend}
        statuses: set[str] = {base_sol.status}
        resid_max: float = base_sol.residual_norm
        floor_ever_bound = False
        for year in years:
            cc, _prov = cc_by_year[year]
            theta = theta_by_year[year] if theta_by_year is not None else None
            sol, floor_applied = _solve(
                cal,
                carbon_cost=cc,
                recycling=recycling,
                inv_closure=inv_closure,
                gov_closure=gov_closure,
                carbon_revenue_recipient=carbon_revenue_recipient,
                labour_floor=labour_floor,
                adapt_amount=adapt_amount,
                adapt_gamma=adapt_gamma,
                productivity=theta,
            )
            floor_ever_bound = floor_ever_bound or floor_applied is not None
            backends.add(sol.backend)
            statuses.add(sol.status)
            resid_max = max(resid_max, sol.residual_norm)
            # strict=True: the recycling k<1 feasibility guard applies to the accepted equilibrium
            # only; residual evaluations inside _solve ran non-strict to keep the solve continuous.
            st = M.derive_state(
                cal,
                sol.x[:ns],
                sol.x[ns:],
                carbon_cost=cc,
                recycling=recycling,
                strict=True,
                inv_closure=inv_closure,
                gov_closure=gov_closure,
                carbon_revenue_recipient=carbon_revenue_recipient,
                labour_floor=floor_applied,
                adapt_amount=adapt_amount,
                adapt_gamma=adapt_gamma,
                productivity=theta,
            )
            _emit(records, cal, base, st, year, cc=cc, intensity=emission_intensity)

        # Emissions provenance: the effective aligned cost-share vector per year + the satellite
        # identity, so a changed satellite / gas selection / doubled emissions moves the manifest
        # (review P1). The cost share already folds in gases, coverage and the 1e-6 scaling.
        emissions_inputs = _emissions_provenance(inp, cc_by_year, sectors)
        manifest = RunManifest.build(
            engine_name=self.meta.name,
            engine_version=self.meta.version,
            data_source=data_source_id(sam.provenance),
            scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
            assumptions={
                **ASSUMPTIONS,
                # The complete effective PUBLIC configuration (review P1 round 14): every validated
                # control the run used, with its effective value — so provenance is truthful.
                "effective_config": _effective_public_config(data, variant, entry),
                "sectors": sectors,
                "factors": factors,
                # VA elasticity materially changes results, so it belongs in the manifest — two runs
                # differing only in va_elast must have different assumptions (review P1). The nest
                # description reflects the CALIBRATED nest, not a hardcoded 'Cobb-Douglas' (P2).
                "va_elast": [round(float(v), 12) for v in cal.va_elast.tolist()],
                "value_added_nest": _va_nest_description(cal.va_elast),
                # Energy nest (Phase 5d.5): flat Leontief or the KL-E-M nest (shared helper).
                **_energy_manifest(cal, sectors),
                "recycling_mode": recycling,
                "recycling_defaulted_from_none": recycling_defaulted,
                # Government account (Phase 5d.1): which account (or none), its closure, and the
                # benchmark direct-tax share of factor income — two runs differing only in the
                # presence/size of the government must differ in assumptions. With a government,
                # `recycling_mode` routes carbon revenue TO THE GOVERNMENT BUDGET (spent on
                # gov_gamma under balanced_budget), not to household income.
                "government_account": _GOV_ACCOUNT if cal.has_government else "none",
                "gov_closure": gov_closure if cal.has_government else "n/a (no government)",
                # Who receives carbon revenue (Phase 5d.7 / review 2026-07-27) — a choice separate
                # from the fiscal closure. 'government' (default) funds the budget; 'household'
                # recycles to the household. Under balanced_budget revenue funds the government by
                # construction, so this only bites under deficit_financed.
                "carbon_revenue_recipient": (
                    carbon_revenue_recipient if cal.has_government else "n/a (no government)"
                ),
                "gov_benchmark_tax_share_of_factor_income": (
                    round(float(cal.gov_tax_rate0), 12) if cal.has_government else 0.0
                ),
                # Savings-investment account (Phase 5d.2): which account (or none), the active
                # closure, and the benchmark savings rate — runs differing only in the investment
                # closure must differ in assumptions.
                "savings_investment_account": (_SAVINV_ACCOUNT if cal.has_investment else "none"),
                "inv_closure": inv_closure if cal.has_investment else "n/a (no investment)",
                "benchmark_savings_rate_of_disposable_income": (
                    round(float(cal.sav_rate0), 12) if cal.has_investment else 0.0
                ),
                # Labour-market closure (Phase 5d.4): the default flexible-wage/full-employment, or
                # a wage floor. ``labour_floor_bound`` records whether the floor actually bound in
                # any year (a configured-but-slack floor leaves the full-employment result and is
                # honestly reported as not binding).
                "labour_closure": (
                    "wage_floor" if labour_floor is not None else "flexible_wage_full_employment"
                ),
                "labour_floor": float(labour_floor) if labour_floor is not None else None,
                "labour_floor_bound": floor_ever_bound,
                # Adaptation/transition investment (Phase 5d.6): the earmarked capex (GDP share)
                # that crowds out ordinary investment. 0 when none.
                "adaptation_investment_share_of_gdp": round(adapt_amount, 12),
                "adaptation_crowds_out_ordinary_investment": adapt_amount > 0.0,
                "solver_backends": sorted(backends),
                "solver_statuses": sorted(statuses),
                # The strongest numerical convergence evidence: max ‖F(x)‖∞ over all solves. The
                # solver already re-verifies this < tol before returning (else it raises), so this
                # records HOW converged the equilibrium is (review P2).
                "solver_max_residual_norm": resid_max,
                "emissions_priced": emissions_priced,
                "carbon_cost_path": cc_by_year[years[0]][1].get("path"),
                "benchmark_gdp_normalised": cal.gdp0,
                # SAM credibility surface: worst quality severity + per-check summary, so a run
                # states how much the SAM data was helped (roadmap 5.1c). None when a SAM was
                # supplied directly (validated separately, review P1).
                "sam_quality": (
                    {"worst": inp.sam_quality.worst.value, "summary": inp.sam_quality.summary()}
                    if inp.sam_quality is not None
                    else "supplied directly (validated: aligned, finite, non-negative, balanced)"
                ),
                "inputs": [
                    input_identity("SAM", sam.provenance, content=_sam_fingerprint(sam)),
                    *emissions_inputs,
                ],
            },
        )
        return ResultSet.from_records(records, manifest)


@dataclass(frozen=True)
class _Inputs:
    sam: SAM
    sectors: list[str]
    sam_quality: object  # QualityReport | None
    io: IOSystem | None  # present on the real-build path (drives the carbon cost)
    sat: SatelliteAccount | None
    cost_share: np.ndarray | None  # per-sector τ=1 cost share, supplied-SAM path only


def _resolve_inputs(data: dict) -> _Inputs:
    """Resolve the CGE's inputs from ``data``, validating both entry paths.

    - ``SAM`` supplied directly: **validated** (account alignment, finite, non-negative, balanced;
      review P1) before use; the carbon cost comes from a supplied per-sector ``carbon_cost_share``.
    - ``IOSystem`` (a real build): the SAM is built + quality-gated (a failing SAM is rejected), and
      the carbon cost is computed from the satellite the SAME way as Engine 1 (units, gases,
      coverage, and the 1e-6 M€→€ scaling), aggregated to sectors."""
    if "SAM" in data:
        sam: SAM = data["SAM"]
        factors = [f for f in _DEFAULT_FACTORS if f in sam.accounts]
        sectors = data.get("sectors") or _infer_sectors(sam, factors)
        _validate_supplied_sam(sam, sectors, factors)
        return _Inputs(sam, sectors, None, None, None, _carbon_cost_share(data, sectors))

    io = data.get("IOSystem")
    if io is None:
        raise ValueError("cge_static needs a 'SAM' or an 'IOSystem' in data")
    from cge.data.sam import build_sam
    from cge.engines.io_price.engine import assert_io_aligned

    assert_io_aligned(io)  # boundary guard before the SAM is built from raw A (review P2)
    sam, quality, sectors = build_sam(io)
    if not quality.passed:
        failed = [c.name for c in quality.checks if c.severity.value == "fail"]
        raise ValueError(f"SAM quality gate failed for the build: {failed}; refusing to calibrate.")
    return _Inputs(sam, sectors, quality, io, data.get("SatelliteAccount"), None)


def _validate_supplied_sam(sam: SAM, sectors: list[str], factors: list[str]) -> None:
    """Gate a directly-supplied SAM (review P1: a supplied SAM bypassed every check). Requires the
    named sector/factor accounts to exist, the **matrix axes to be unique and aligned to the
    accounts** used (review P2: a renamed-axis matrix passed the name check then raised a raw
    KeyError during calibration), all cells finite and non-negative, and the matrix balanced (row
    sum = column sum per account). The engine will not calibrate on a bad SAM."""
    from cge.data.sam.balance import is_balanced

    m = sam.matrix
    missing = [a for a in sectors + factors if a not in sam.accounts]
    if missing:
        raise ValueError(f"supplied SAM is missing named accounts: {missing}")
    # Axis alignment: the matrix index and columns must be unique and **equal** the declared
    # accounts (review P2: containment is not enough — an extra balanced account passed the check,
    # was silently ignored by calibration, yet the manifest called the SAM 'aligned'). A renamed
    # axis would also raise a raw KeyError during calibration.
    idx, cols = list(m.index), list(m.columns)
    if len(set(idx)) != len(idx) or len(set(cols)) != len(cols):
        raise ValueError("supplied SAM matrix has duplicate row or column labels")
    accts = set(sam.accounts)
    if set(idx) != accts or set(cols) != accts:
        extra = sorted((set(idx) | set(cols)) - accts)
        missing_axis = sorted(accts - (set(idx) & set(cols)))
        raise ValueError(
            f"supplied SAM matrix axes must equal the declared accounts exactly; "
            f"extra axis labels not in accounts: {extra}; accounts missing from an axis: "
            f"{missing_axis}."
        )
    arr = m.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("supplied SAM has non-finite cells")
    if float(arr.min()) < -1e-9:
        raise ValueError("supplied SAM has negative cells; a SAM must be non-negative")
    if not is_balanced(m, tol=1e-6):
        from cge.data.sam.balance import imbalance

        worst = float(imbalance(m).abs().max())
        raise ValueError(
            f"supplied SAM is not balanced (max |row−col| = {worst:.3e} > 1e-6); "
            f"the CGE calibrates only on a balanced SAM."
        )


def _emissions_provenance(inp: _Inputs, cc_by_year: dict, sectors: list[str]) -> list[dict]:
    """Reproducibility records for the carbon-cost inputs (review P1: emissions were unrecorded).

    Records the satellite identity + content hash (real-build path) and a content hash of the
    **effective aligned cost-share vector** per year — so a changed satellite, doubled emissions, or
    a different gas/coverage selection all move the manifest. Empty when no carbon cost applies."""
    effective = {
        str(y): [round(float(v), 12) for v in cc.tolist()] for y, (cc, _p) in cc_by_year.items()
    }
    if not any(any(v != 0.0 for v in row) for row in effective.values()):
        return []  # no carbon cost priced; nothing substantive to fingerprint
    out = [
        {
            "name": "EffectiveCarbonCostShare",
            "sectors": sectors,
            "content_hash": content_hash(effective),
        }
    ]
    if inp.sat is not None:
        from cge.engines.io_price.engine import _df_fingerprint

        out.append(
            input_identity(
                "SatelliteAccount", inp.sat.provenance, content=_df_fingerprint(inp.sat.data)
            )
        )
    return out


def _assert_cge_units(io: IOSystem, sat: SatelliteAccount) -> None:
    """Currency-flexible unit gate for the CGE carbon cost. The 1e-6 M→unit scaling in
    ``carbon_cost_vector`` is valid for any *millions*-denominated currency, so we require the
    monetary unit to be ``M<CUR>`` matching the build currency, and every satellite row to be
    ``t/M<CUR>`` (physical gas) or ``tCO2e/M<CUR>`` (the CO2e row). A ``kg/…`` unit is 1000× off and
    rejected. The carbon price is then interpreted in ``<CUR>``/tonne."""
    from cge.engines.io_price.engine import _CO2E_ROW

    cur = io.currency
    if io.unit != f"M{cur}":
        raise ValueError(
            f"cge_static carbon cost needs a millions-denominated monetary base 'M{cur}'; "
            f"build unit is {io.unit!r} (currency {cur!r}). Aggregate/convert the build first."
        )
    if not sat.units:
        raise ValueError(f"satellite {sat.name!r} has no unit metadata; cannot verify t/M{cur}.")
    for row in sat.data.index:
        expected = f"tCO2e/M{cur}" if row == _CO2E_ROW else f"t/M{cur}"
        if sat.units.get(row) != expected:
            raise ValueError(
                f"satellite row {row!r} has unit {sat.units.get(row)!r}, expected {expected!r}; "
                f"the M→{cur} carbon cost-share scaling assumes exactly this unit."
            )


def _validate_cge_shock_controls(inp: _Inputs, carbon_shocks: list[CarbonPrice]) -> None:
    """Reject shock controls the CGE cannot honour, rather than silently ignore them (review P1).

    - **IO-backed path:** coverage labels (sectors/regions) must exist in the build, exactly as
      Engine 1 requires — a typo would otherwise give a silent zero-impact scenario.
    - **Supplied-SAM path:** the dimensionless ``carbon_cost_share`` cannot express per-gas or
      spatial coverage, so any non-default ``gases`` / ``coverage_sectors`` / ``coverage_regions``
      is **rejected** (not applied to a global vector as if honoured)."""
    if inp.io is not None:
        from cge.engines.io_price.engine import _assert_coverage_labels

        _assert_coverage_labels(carbon_shocks, inp.io)
        return
    # Supplied-SAM path: reject controls that this path structurally cannot apply.
    for s in carbon_shocks:
        if s.gases != ["CO2"]:
            raise ValueError(
                f"the supplied-SAM CGE path applies a single dimensionless carbon_cost_share and "
                f"cannot select gases; got gases={s.gases}. Use an IOSystem+satellite build for "
                f"gas selection, or leave gases at the default ['CO2']."
            )
        if s.coverage_sectors or s.coverage_regions:
            raise ValueError(
                "the supplied-SAM CGE path cannot apply sector/region coverage (the cost share is "
                "already per-sector and single-region); set coverage via carbon_cost_share values, "
                "or use an IOSystem+satellite build."
            )


def _carbon_intensity_by_sector(
    inp: _Inputs, carbon_shocks: list[CarbonPrice]
) -> np.ndarray | None:
    """Per-sector **price-free** emission intensity for the covered-emissions output (review P1
    round 14). This is the effective cost share at a UNIT carbon price (price=1) — it honours the
    scenario's gases and coverage but NOT the price level, so ``covered_emissions_change`` is
    well-defined **every year, including a zero-price year** (an emissions quantity must not vanish
    because the policy price is 0). Returns None only when there is genuinely nothing to weight (no
    satellite / no carbon shocks / no coverage hits).

    Supplied-SAM path: the supplied ``cost_share`` IS the price-free intensity (the τ=1 wedge).
    IO path: re-run ``carbon_cost_vector`` with the shocks' price forced to 1."""
    if not carbon_shocks:
        return None
    if inp.io is None:
        return inp.cost_share  # already price-free (the τ=1 dimensionless wedge), or None
    if inp.sat is None:
        return None
    # Force a flat unit price (clearing any price path) so the vector is intensity-only: gases and
    # coverage are honoured, the price level is removed.
    unit_shocks = [s.model_copy(update={"price": 1.0, "path": None}) for s in carbon_shocks]
    cc, _prov = _carbon_cost_by_sector(
        inp, unit_shocks, 0
    )  # any year: unit price is year-invariant
    return cc if np.any(cc != 0.0) else None


def _carbon_cost_by_sector(
    inp: _Inputs, carbon_shocks: list[CarbonPrice], year: int
) -> tuple[np.ndarray, dict]:
    """Per-sector **dimensionless** carbon cost share for ``year``, plus a provenance dict.

    Real-build path: reuse Engine 1's ``carbon_cost_vector`` (which honours gases, coverage, per-gas
    GWP, and the 1e-6 M€→€ scaling — fixing review P0/P1) on the multi-regional labels, then
    aggregate to sectors output-weighted. Supplied-SAM path: ``cost_share`` × Σ τ (the caller
    supplies the dimensionless τ=1 wedge). Returns ``(cc_sector, provenance)``; provenance feeds the
    manifest so a changed satellite / gas selection moves the result's identity (review P1)."""
    ns = len(inp.sectors)
    if inp.io is None:
        # Supplied-SAM (toy) path: dimensionless cost share × total carbon price.
        share = inp.cost_share if inp.cost_share is not None else np.zeros(ns)
        tau = sum(s.price_at(year) for s in carbon_shocks)
        return tau * share, {"path": "supplied_cost_share"}

    from cge.engines.io_price.engine import carbon_cost_vector

    io, sat = inp.io, inp.sat
    if sat is None:
        return np.zeros(ns), {"path": "no_satellite", "emissions_priced": False}
    # Reject a build whose units make the 1e-6 M-currency→currency scaling wrong. Unlike Engine 1
    # (euro-specific), the CGE accepts any millions-denominated currency, requiring the satellite
    # intensity to be t / M<currency> (or tCO2e / M<currency> for the CO2e row) so the 1e-6 in
    # carbon_cost_vector is exact and the carbon price is in <currency>/tonne (review P0/P2).
    _assert_cge_units(io, sat)
    labels = list(io.A.columns)
    # Per-label direct carbon cost share (dimensionless): honours gases + coverage + 1e-6 scaling.
    cost, descs = carbon_cost_vector(carbon_shocks, sat, labels, year)
    # Aggregate to sectors, output-weighted (cost is a per-unit-output share; weight by output).
    A = io.A.to_numpy(dtype=float)
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
    x = np.linalg.solve(np.eye(A.shape[0]) - A, fd)
    s_index = {s: k for k, s in enumerate(inp.sectors)}
    num = np.zeros(ns)
    den = np.zeros(ns)
    for lb, c_i, xi in zip(labels, cost, x, strict=True):
        k = s_index[lb.split(":", 1)[1]]
        num[k] += c_i * xi
        den[k] += xi
    cc = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    prov = {
        "path": "engine1_carbon_cost_vector",
        "contributions": descs,
        "gases": sorted({g for s in carbon_shocks for g in s.gases}),
        "emissions_priced": bool(np.any(cc != 0.0)),
    }
    return cc, prov


def _assert_no_region_scoped_productivity(
    prod_shocks: list[ProductivityShock], variant: str
) -> None:
    """The closed/open CGE is **single-region** (the SAM collapses regions), so a ProductivityShock
    that carries region coverage is ILL-POSED here — the model has no region dimension to apply it
    to, and any aggregation of "this region only" against a region-less economy is a guess that has
    twice produced silent wrong answers (review P1 rounds 2 and 3: a region-A-only shock came out
    identical to an economy-wide shock). So reject it, with guidance to the multi-region variant.

    A region-scoped shock is one whose ``coverage_regions`` is non-empty. An economy-wide shock
    (empty ``coverage_regions``) is unambiguous and allowed."""
    scoped = sorted({r for s in prod_shocks for r in s.coverage_regions})
    if scoped:
        raise ValueError(
            f"the {variant!r} CGE variant is single-region (the SAM collapses regions), so a "
            f"region-scoped ProductivityShock (coverage_regions={scoped}) cannot be applied "
            "unambiguously — there is no region dimension to target. Use the multi-region CGE "
            "variant for region-specific nature shocks, or drop the region coverage to apply the "
            "shock economy-wide."
        )


def _productivity_by_sector(
    prod_shocks: list[ProductivityShock], sectors: list[str], year: int
) -> np.ndarray:
    """Per-sector Hicks-neutral productivity multiplier θ[i] for ``year`` (Phase 6.4 GE tier).

    Assumes ``prod_shocks`` are **economy-wide** (region coverage already rejected by
    ``_assert_no_region_scoped_productivity`` — the collapsed single-region model has no region
    dimension). Multiple shocks on one sector compose multiplicatively (independent hits): two −10%
    shocks give 0.81. A sector with no shock has θ = 1 (byte-identical to a pure carbon run). θ is
    floored at a small positive value so a fully-degraded sector cannot drive the unit cost
    non-finite."""
    theta = np.ones(len(sectors))
    for i, sec in enumerate(sectors):
        for s in prod_shocks:
            if s.coverage_sectors and sec not in s.coverage_sectors:
                continue
            theta[i] *= 1.0 + s._path_level_at(year, s.delta)
    return np.clip(theta, 1e-6, None)


def _productivity_by_region_sector(
    prod_shocks: list[ProductivityShock], regions: list[str], sectors: list[str], year: int
) -> np.ndarray:
    """Per-(region,sector) productivity multiplier θ[r,i] for ``year`` (Phase 6.4 GE tier, multi
    variant). Unlike the single-region closed/open variants, the multi CGE HAS a region dimension,
    so a shock's region coverage IS honoured via ``applies_to(sector, region)`` — a nature
    degradation built for one region hits only that region's sectors. Composes multiplicatively;
    θ=1 where no shock covers (r, i); floored positive."""
    theta = np.ones((len(regions), len(sectors)))
    for ri, reg in enumerate(regions):
        for si, sec in enumerate(sectors):
            for s in prod_shocks:
                if s.applies_to(sec, reg):
                    theta[ri, si] *= 1.0 + s._path_level_at(year, s.delta)
    return np.clip(theta, 1e-6, None)


def _infer_sectors(sam: SAM, factors: list[str]) -> list[str]:
    """Sectors = SAM accounts that are neither factors nor institutions (household/government)."""
    non_factor = [a for a in sam.accounts if a not in factors]
    # The household is the account that receives from factors (a factor row pays it). A ``GOV``
    # account is an institution BY NAME (Phase 5d.1, same convention as ``ROW``/``HOH_<r>`` for
    # variant dispatch): a government with a zero benchmark row receives nothing from factors, so
    # the receives-from-factors test alone would misclassify it as a zero-output sector.
    institutions = [a for a in non_factor if any(sam.matrix.loc[a, f] != 0 for f in factors)]
    return [
        a for a in non_factor if a not in institutions and a not in (_GOV_ACCOUNT, _SAVINV_ACCOUNT)
    ]


def _solve(
    cal,
    *,
    carbon_cost,
    recycling="none",
    inv_closure="savings_driven",
    gov_closure="balanced_budget",
    carbon_revenue_recipient="government",
    labour_floor=None,
    adapt_amount=0.0,
    adapt_gamma=None,
    productivity=None,
):
    # prefer='scipy' explicitly: the CGE model residual is numeric-only (it evaluates the Leontief
    # inverse and Cobb-Douglas cost functions with numpy), so it cannot build a symbolic Pyomo
    # model. Auto-selecting IPOPT when its binary is present would therefore FAIL (review P1). A
    # symbolic residual to enable IPOPT is a documented follow-up; scipy solves the small model.
    #
    # Labour-market regime-switch (Phase 5d.4): first solve the DEFAULT full-employment system. If
    # no floor is configured, or the equilibrium labour wage already sits at/above the floor (it is
    # slack), that solution stands — returned with ``floor_applied=None`` so the caller derives the
    # state with the same (no-floor) closure. Only when the unconstrained wage would fall BELOW the
    # floor do we re-solve the wage-floor system (LAB clears on the pinned wage, not on quantity),
    # returning the floor so the caller reports the resulting unemployment. This avoids a genuine
    # mixed-complementarity solver — the scipy backend can't do MCP — while giving the right
    # economics for the documented case.
    def _resid(z, floor):
        return M.residuals(
            cal,
            z,
            carbon_cost=carbon_cost,
            recycling=recycling,
            inv_closure=inv_closure,
            gov_closure=gov_closure,
            carbon_revenue_recipient=carbon_revenue_recipient,
            labour_floor=floor,
            adapt_amount=adapt_amount,
            adapt_gamma=adapt_gamma,
            productivity=productivity,
        )

    sol = solve(lambda z: _resid(z, None), M.initial_guess(cal), prefer="scipy")
    if labour_floor is None or "LAB" not in cal.factors:
        return sol, None
    lab = cal.factors.index("LAB")
    w_lab = float(sol.x[len(cal.sectors) + lab])
    if w_lab >= labour_floor - 1e-12:
        return sol, None  # floor is slack — full employment stands
    floor_sol = solve(lambda z: _resid(z, labour_floor), M.initial_guess(cal), prefer="scipy")
    return floor_sol, labour_floor


def _emit(records, cal, base, st, year: int, cc=None, intensity=None) -> None:
    """Append price/volume changes and GE outputs (relative to the benchmark) for one year. ``cc``
    is the year's per-sector carbon cost; ``intensity`` is the PRICE-FREE per-sector emission
    intensity (the supplied carbon_cost_share) for the physical covered-emissions output."""
    for i, sector in enumerate(cal.sectors):
        records.append(_rec("price_change", sector, year, st.p[i] / base.p[i] - 1.0))
        records.append(_rec("volume_change", sector, year, st.X[i] / base.X[i] - 1.0))
        # Sector GVA — INCOME (current factor payments) AND VOLUME (factor quantity at benchmark
        # prices), named separately so a report does not read a nominal move as a volume move
        # (review P1 round 13). gva_change is kept as the income series for back-compat.
        records.append(
            _rec("gva_change", sector, year, _gva_income(st, i) / _gva_income(base, i) - 1.0)
        )
        records.append(
            _rec("gva_income_change", sector, year, _gva_income(st, i) / _gva_income(base, i) - 1.0)
        )
        records.append(
            _rec("gva_volume_change", sector, year, _gva_volume(st, i) / _gva_volume(base, i) - 1.0)
        )
    # Factor prices: the generic change AND the named wage / capital-return variables (standard
    # schema — a report should not have to know that CAP's factor price IS the capital return).
    for f_idx, factor in enumerate(cal.factors):
        records.append(_rec("factor_price_change", factor, year, st.w[f_idx] / base.w[f_idx] - 1.0))
        named = {"LAB": "wage_change", "CAP": "capital_return_change"}.get(factor)
        if named is not None:
            records.append(_rec(named, factor, year, st.w[f_idx] / base.w[f_idx] - 1.0))
        # Employment / factor use: total demand for the factor. Under full employment this is the
        # (fixed) endowment so the change is ~0; it is emitted named so the schema is complete and a
        # wage-floor run's fall in employment is visible.
        emp = float(st.F[f_idx, :].sum())
        emp_base = float(base.F[f_idx, :].sum())
        var = "employment_change" if factor == "LAB" else f"factor_use_change_{factor}"
        records.append(_rec(var, factor, year, emp / emp_base - 1.0))
    # GDP (review P1 round 13, 2026-07-28). Expenditure GDP = C + G + I = Σ (FD + GD + ID) (GD/ID
    # zero without their accounts). Three named series, consistent with open/multi:
    #   • gdp_change_real  — VOLUME: quantities at BENCHMARK prices (a Laspeyres quantity index).
    #     This is the fix — the earlier code valued at CURRENT prices, which under the CPI numéraire
    #     is a real-INCOME/purchasing-power measure, not GDP volume.
    #   • gdp_change_nominal — CURRENT-PRICE GDP (the same basket at current prices).
    #   • gdp_deflator_change — current / volume: a genuine relative GDP-price index vs the CPI
    #     numéraire, NOT hard-coded 0.
    fd_agg = st.FD + st.GD + st.ID
    fd_agg_base = base.FD + base.GD + base.ID
    gdp_vol, gdp_cur, gdp_defl = _gdp_measures(st.p, base.p, fd_agg, fd_agg_base)
    records.append(_rec("gdp_change_real", "__economy__", year, gdp_vol))
    records.append(_rec("gdp_change_nominal", "__economy__", year, gdp_cur))
    records.append(_rec("gdp_deflator_change", "__economy__", year, gdp_defl))
    # Welfare: the change in Cobb-Douglas utility U = Π FD_i^γ_i (the correct welfare measure for a
    # CD household — review P1: the earlier Σ FD sum is not utility). With a government account
    # this values HOUSEHOLD consumption only — government-provided goods carry no utility (a
    # documented 5d.1 scope choice; see model.derive_state).
    u = float(np.prod(np.power(st.FD, cal.gamma)))
    u_base = float(np.prod(np.power(base.FD, cal.gamma)))
    records.append(_rec("welfare_change", "__economy__", year, u / u_base - 1.0))
    records.append(_rec("carbon_revenue", "__economy__", year, st.carbon_revenue / cal.gdp0))
    # Household consumption VOLUME (benchmark prices; review P1 round 13 — same convention as GDP).
    cons_vol = float(np.dot(base.p, st.FD))
    cons_vol_base = float(np.dot(base.p, base.FD))
    records.append(_rec("consumption_change", "__economy__", year, cons_vol / cons_vol_base - 1.0))
    _emit_covered_emissions(records, intensity, base.X, st.X, year)
    # Energy-sector output volume (only with the energy nest): total real output of the energy
    # sectors. Renamed from energy_use_change (review P2 round 13): it is energy-sector PRODUCTION
    # volume, not physical energy USE (it excludes imported energy and includes exported energy).
    if cal.has_energy_nest:
        eidx = cal.energy_nest.energy_idx
        eu = float(st.X[eidx].sum())
        eu_base = float(base.X[eidx].sum())
        records.append(
            _rec("energy_sector_output_volume_change", "__economy__", year, eu / eu_base - 1.0)
        )
    # Government account (Phase 5d.1): fiscal balance (≡0 under balanced_budget — emitted so the
    # identity is visible/pinned and a future deficit_financed closure has its output slot) and
    # government spending, as shares of benchmark GDP like carbon_revenue. Emitted ONLY when a
    # government account exists, so no-government runs stay byte-identical to pre-5d.1 output.
    if cal.has_government:
        records.append(_rec("fiscal_balance", "__economy__", year, st.fiscal_balance / cal.gdp0))
        gov_spend = float(np.dot(st.p, st.GD))
        records.append(_rec("gov_spending", "__economy__", year, gov_spend / cal.gdp0))
    # Savings-investment account (Phase 5d.2): nominal investment and household savings as shares
    # of benchmark GDP (equal by construction under savings_driven — the identity is emitted so
    # it is visible and pinned, like fiscal_balance). Emitted only when the account exists.
    if cal.has_investment:
        inv_spend = float(np.dot(st.p, st.ID))
        records.append(_rec("investment", "__economy__", year, inv_spend / cal.gdp0))
        records.append(_rec("savings", "__economy__", year, st.savings / cal.gdp0))
    # Adaptation/transition investment (Phase 5d.6): the earmarked adaptation capex AND the
    # ordinary investment it displaced (total − adaptation), both as GDP shares — so a report
    # shows "X on adaptation, Y of ordinary investment crowded out". Emitted only when adaptation
    # is active, so an ordinary run stays byte-identical.
    if st.adaptation_investment > 1e-12:
        records.append(
            _rec("adaptation_investment", "__economy__", year, st.adaptation_investment / cal.gdp0)
        )
        other = float(np.dot(st.p, st.ID)) - st.adaptation_investment
        records.append(_rec("ordinary_investment", "__economy__", year, other / cal.gdp0))
    # Labour market (Phase 5d.4): the unemployment RATE (unemployed labour ÷ labour endowment),
    # emitted only when a wage floor actually binds — a no-floor / full-employment run stays
    # byte-identical to pre-5d.4 output.
    if st.unemployment > 1e-9 and "LAB" in cal.factors:
        lab_endow = float(cal.endowment[cal.factors.index("LAB")])
        records.append(_rec("unemployment", "__economy__", year, st.unemployment / lab_endow))


def _rec(variable: str, sector: str, year: int, value: float) -> dict:
    return {
        "variable": variable,
        "sector": sector,
        "region": "R",
        "year": year,
        "scenario": "central",
        "value": float(value),
    }


def _gva_income(st, i: int) -> float:
    """Sector ``i`` gross value added at CURRENT prices = the factor payments it makes = Σ_f
    w[f]·F[f,i] (value added is, by construction, what the sector pays its factors). Works for every
    variant/state with ``.w`` [f] and ``.F`` [f, i]. This is a nominal (income) measure — its change
    conflates factor-price and factor-quantity moves."""
    return float(np.dot(st.w, st.F[:, i]))


def _gva_volume(st, i: int) -> float:
    """Sector ``i`` gross value added VOLUME = factor QUANTITY valued at BENCHMARK factor prices
    (all = 1) = Σ_f F[f,i] (review P1 round 13, 2026-07-28). A Laspeyres quantity index, so its
    change is a genuine volume move, not a factor-price artifact — distinct from ``_gva_income``."""
    return float(st.F[:, i].sum())


def _gdp_measures(p, base_p, final_demand, base_final_demand):
    """Return ``(volume_change, current_price_change, deflator_change)`` for an expenditure GDP
    aggregate ``final_demand`` (= FD + GD + ID, per variant) given current prices ``p`` and
    benchmark prices ``base_p`` (all 1 at benchmark). Review P1 round 13:
      - VOLUME GDP values quantities at BENCHMARK prices (Laspeyres): Σ base_p·fd → a real quantity
        index. This is ``gdp_change_real``.
      - CURRENT-PRICE GDP values them at current prices: Σ p·fd. This is ``gdp_change_nominal``.
      - The GDP DEFLATOR is current-price / volume; its change is (nominal_ratio / volume_ratio) − 1
        — a genuine relative-price index of the GDP basket vs the CPI numéraire, NOT hard-coded 0.
    Consistent across closed/open/multi so the variants no longer disagree."""
    vol = float(np.dot(base_p, final_demand))
    vol_base = float(np.dot(base_p, base_final_demand))
    cur = float(np.dot(p, final_demand))
    cur_base = float(
        np.dot(base_p, base_final_demand)
    )  # benchmark current-price = benchmark volume
    vol_ratio = vol / vol_base
    cur_ratio = cur / cur_base
    deflator_change = cur_ratio / vol_ratio - 1.0 if vol_ratio != 0 else 0.0
    return vol_ratio - 1.0, cur_ratio - 1.0, deflator_change


def _covered_emissions_change(intensity, base_X, st_X) -> float | None:
    """Change in **covered physical emissions** ``Σ e_i·X_i`` between benchmark and shocked output,
    where ``intensity`` is the PRICE-FREE per-sector emission intensity — the supplied
    ``carbon_cost_share`` (cost per €1 of carbon price = e_i × scaling), NOT the priced cost
    ``cc = τ·share`` (review P1 round 13, 2026-07-28). Weighting by the price-free intensity is what
    makes this a physical-emissions measure: it is the emissions of the COVERED (priced-eligible)
    sectors, independent of the carbon price level, so it is well-defined and emittable even in a
    zero-price year. The earlier version weighted by ``cc`` and returned nothing at zero price — a
    physical quantity should not vanish because the policy price is zero.

    It is **covered** emissions, not territorial: only sectors with a nonzero intensity (the priced
    subset) are counted, and it is value-weighted (the cost share), not measured in joules/tCO2 — a
    true physical inventory needs a satellite emission-intensity vector (documented follow-up).
    Returns None only when NO sector has a nonzero intensity (nothing to measure)."""
    denom = float(np.dot(intensity, base_X))
    if denom <= 1e-15:
        return None
    return float(np.dot(intensity, st_X)) / denom - 1.0


def _emit_covered_emissions(records, intensity, base_out, st_out, year, *, region=None) -> None:
    """Emit ``covered_emissions_change`` = the change in Σ intensity·output, where ``intensity`` is
    the PRICE-FREE per-sector emission intensity (the supplied ``carbon_cost_share``), so the
    physical quantity is well-defined **every year, including zero-price years** (review P1 round
    13). ``base_out``/``st_out`` are the benchmark/shocked output (X closed, Z open/multi)."""
    if intensity is None:
        return
    ch = _covered_emissions_change(np.asarray(intensity, dtype=float), base_out, st_out)
    if ch is None:
        return
    rec = (
        _rec("covered_emissions_change", "__economy__", year, ch)
        if region is None
        else _rec_r("covered_emissions_change", "__economy__", region, year, ch)
    )
    records.append(rec)


def _sam_fingerprint(sam: SAM) -> dict:
    """Content fingerprint of a SAM for the manifest. **Canonicalised by account label** so the
    identity depends on the *economics*, not the axis order (review P1: the old fingerprint stored
    only ``accounts`` + the raw flattened array, so two matrices whose numeric block was identical
    but whose axes were permuted — different economies, since calibration reads cells by label —
    collided to the same hash). We reindex the matrix to the sorted account order and record that
    ordering alongside the values, so relabelling either axis changes the fingerprint."""
    order = sorted(sam.accounts)
    m = sam.matrix.reindex(index=order, columns=order)
    return {
        "accounts": list(sam.accounts),
        "canonical_order": order,
        "values": [round(float(v), 10) for v in m.to_numpy(dtype=float).ravel().tolist()],
    }


_REPLICATION_TOL = 1e-6


def _assert_closed_replicates(cal, base, x0: np.ndarray) -> None:
    """Assert the benchmark solve reproduces the closed model's calibrated quantities (review P1).

    A SAM that passes the structural validators but carries flows outside the implemented topology
    (Leontief intermediates + CD/CES VA + a single CD household) will *not* replicate: the derived
    benchmark output/final-demand/factor demand drift from the calibrated ``X0/FD0/F0``. Because
    every reported change is relative to this base, a non-replicating base silently corrupts every
    result — so we refuse the run rather than report changes against a wrong benchmark."""
    checks = {
        "prices": (x0, np.ones_like(x0)),
        "output X": (base.X, cal.X0),
        "final demand FD": (base.FD, cal.FD0),
        "factor demand F": (base.F, cal.F0),
    }
    if cal.has_government:
        # The benchmark government must reproduce its SAM column too (Phase 5d.1): tax-funded
        # spending at benchmark prices equals the calibrated GD0 exactly, or the run is refused.
        checks["government demand GD"] = (base.GD, cal.GD0)
    if cal.has_investment:
        # Likewise the benchmark investment column (Phase 5d.2), under either closure.
        checks["investment demand ID"] = (base.ID, cal.INV0)
    _raise_if_not_replicating(checks, "closed")


def _assert_open_replicates(
    cal, base, x0: np.ndarray, trade_closure="fixed_foreign_savings"
) -> None:
    """Assert the benchmark solve reproduces the OPEN model's calibrated quantities (review P1).
    Same rationale as the closed gate, over the open benchmark set (Z/D/E/M/Q/FD/F) and unit
    prices. The last unknown is er (fixed_foreign_savings — benchmark 1) or Sf (flexible_balance —
    benchmark cal.foreign_savings), so it is checked against its own benchmark, not blanket 1."""
    prices = x0[:-1]
    last = x0[-1]
    last_benchmark = 1.0 if trade_closure == "fixed_foreign_savings" else float(cal.foreign_savings)
    checks = {
        "prices": (prices, np.ones_like(prices)),
        "last unknown (er or Sf)": (np.array([last]), np.array([last_benchmark])),
        "output Z": (base.Z, cal.Z0),
        "domestic D": (base.D, cal.D0),
        "exports E": (base.E, cal.E0),
        "imports M": (base.M, cal.M0),
        "composite Q": (base.Q, cal.Q0),
        "final demand FD": (base.FD, cal.FD0),
        "factor demand F": (base.F, cal.F0),
    }
    if cal.has_government:
        # Benchmark government demand must reproduce its SAM column too (Phase 5d.1).
        checks["government demand GD"] = (base.GD, cal.GD0)
    if cal.has_investment:
        # Likewise the benchmark investment column (Phase 5d.2), under either closure.
        checks["investment demand ID"] = (base.ID, cal.INV0)
    _raise_if_not_replicating(checks, "open")


def _assert_multi_replicates(cal, base, x0: np.ndarray) -> None:
    """Assert the benchmark solve reproduces the MULTI-region model's calibrated quantities and
    unit prices (review P1: the multi path skipped this gate). Same rationale as closed/open."""
    checks = {
        "prices": (x0, np.ones_like(x0)),
        "output Z": (base.Z, cal.Z0),
        "domestic D": (base.D, cal.D0),
        "imports M": (base.M, cal.M0),
        "exports EX": (base.EX, cal.EX0),
        "final demand FD": (base.FD, cal.FD0),
        "factor demand F": (base.F, cal.F0),
    }
    if cal.has_government:
        # Benchmark government demand must reproduce its SAM columns too (Phase 5d.1).
        checks["government demand GD"] = (base.GD, cal.GD0)
    if cal.has_investment:
        # Likewise the benchmark investment columns (Phase 5d.2), under either closure.
        checks["investment demand ID"] = (base.ID, cal.INV0)
    _raise_if_not_replicating(checks, "multi-region")


def _raise_if_not_replicating(checks: dict, variant: str) -> None:
    worst_name, worst_err = None, 0.0
    for name, (got, want) in checks.items():
        err = float(np.max(np.abs(np.asarray(got, dtype=float) - np.asarray(want, dtype=float))))
        if err > worst_err:
            worst_name, worst_err = name, err
    if worst_err > _REPLICATION_TOL:
        raise ValueError(
            f"{variant} CGE benchmark does not replicate the SAM (worst: {worst_name} error "
            f"{worst_err:.2e} > {_REPLICATION_TOL:.0e}). The SAM is balanced but carries flows "
            f"outside the implemented model topology, so every reported change would be measured "
            f"against a wrong benchmark. Reject the run rather than return corrupted results."
        )


def _validate_open_sam(sam: SAM, sectors: list[str], factors: list[str]) -> None:
    """Structural gate for a supplied OPEN SAM (review P1: the open path bypassed every SAM check).

    Requires the ``a_<s>``/``c_<s>`` activity/commodity pair for each sector, the named factors, a
    single household, exactly one ``ROW`` account, unique/aligned axes, finite non-negative cells,
    and an overall-balanced matrix — the same standard the closed path enforces, adapted to the
    open account structure. A bad SAM is rejected rather than silently calibrated."""
    from cge.data.sam.balance import imbalance, is_balanced

    need = [f"a_{s}" for s in sectors] + [f"c_{s}" for s in sectors] + list(factors) + ["ROW"]
    missing = [a for a in need if a not in sam.accounts]
    if missing:
        raise ValueError(f"supplied open SAM is missing required accounts: {missing}")
    households = [
        a
        for a in sam.accounts
        if not a.startswith(("a_", "c_"))
        and a not in factors
        and a != "ROW"
        # GOV/SAVINV are institutions by name (Phase 5d.1/5d.2), not households.
        and a not in (_GOV_ACCOUNT, _SAVINV_ACCOUNT)
    ]
    if len(households) != 1:
        raise ValueError(f"open SAM expects exactly one household account, got {households}")
    m = sam.matrix
    idx, cols = list(m.index), list(m.columns)
    if len(set(idx)) != len(idx) or len(set(cols)) != len(cols):
        raise ValueError("supplied open SAM matrix has duplicate row or column labels")
    accts = set(sam.accounts)
    if set(idx) != accts or set(cols) != accts:
        raise ValueError("supplied open SAM matrix axes must equal the declared accounts exactly")
    arr = m.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("supplied open SAM has non-finite cells")
    if float(arr.min()) < -1e-9:
        raise ValueError("supplied open SAM has negative cells; a SAM must be non-negative")
    if not is_balanced(m, tol=1e-6):
        worst = float(imbalance(m).abs().max())
        raise ValueError(
            f"supplied open SAM is not balanced (max |row−col| = {worst:.3e} > 1e-6); "
            f"the open CGE calibrates only on a balanced SAM."
        )


def _run_open_from_io(meta, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
    """Open-economy CGE on a **real build**: construct an open SAM (home region + rest-of-world)
    from the supplied ``IOSystem`` via ``build_open_sam``, derive the per-sector carbon cost share
    from the satellite the SAME way as Engine 1 (units/gases/coverage/1e-6 scaling, aggregated to
    the home sectors), then delegate to ``_run_open`` with the built SAM + cost share injected.

    ``data`` keys: ``IOSystem`` (+ ``SatelliteAccount``), ``open_home_region`` (which build region
    is the home economy; the rest become ROW), optional ``capital_share``."""
    from cge.data.sam import build_open_sam

    io: IOSystem = data["IOSystem"]
    sat = data.get("SatelliteAccount")
    home = data["open_home_region"]
    kwargs = {}
    if "capital_share" in data:
        kwargs["capital_share"] = data["capital_share"]
    sam, quality, sectors = build_open_sam(io, home_region=home, **kwargs)
    if quality.worst.value == "fail":
        raise ValueError(
            f"open SAM from {io.provenance.build_id} failed quality gates: {quality.summary()}"
        )

    # Per-sector carbon cost share from the satellite, restricted to the HOME region's labels and
    # aggregated to sectors output-weighted (same construction as the closed IO path).
    carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
    # A positive carbon price needs a satellite, or this silently becomes a zero-impact run — the
    # SAME gate the closed IO path already has (review P1: the open path lacked it, so a €100/t
    # request with no SatelliteAccount was silently accepted with emissions_priced=False and zero
    # impact instead of raising). This is distinct from a legitimate zero: a satellite that is
    # present but has zero intensity on the home region's labels, or a coverage selection that
    # excludes the home region, both still produce a real (zero) EffectiveCarbonCost and are left
    # to _open_effective_cc_from_io / _run_open — only a missing satellite is rejected here.
    positive_price = any(s.price_at(y) > 0 for s in carbon_shocks for y in years)
    if positive_price and sat is None:
        raise ValueError(
            "a positive carbon price on an IO-backed open-economy CGE run requires a "
            "'SatelliteAccount' (emission intensities); none supplied — the run would be "
            "silently zero-impact."
        )
    # Gas/coverage controls ARE honoured on this path (carbon_cost_vector applies them), so — like
    # the closed IO path — validate that requested coverage labels exist in the build up front.
    if carbon_shocks:
        from cge.engines.io_price.engine import _assert_coverage_labels

        _assert_coverage_labels(carbon_shocks, io)
    # Build the EFFECTIVE per-year carbon-cost vector (already price × intensity × 1e-6,
    # honouring gases/coverage/paths). This is NOT a price-free share: it is the finished
    # per-year cost, so the open path must consume it directly rather than re-multiply by the
    # price (review P0: the earlier code stuffed this into carbon_cost_share and _run_open
    # re-applied the price → double-counted).
    cc_by_year = _open_effective_cc_from_io(io, sat, carbon_shocks, home, sectors, years)

    inner = {k: v for k, v in data.items() if k not in ("IOSystem", "SatelliteAccount")}
    inner["SAM"] = sam
    inner["_sam_quality"] = quality  # surfaced in the manifest by _run_open
    # Mark the run IO-backed even when the effective cost comes out empty (e.g. coverage excludes
    # the home region): gas/coverage controls were honoured here, so _run_open must not apply its
    # supplied-SAM rejection or demand a carbon_cost_share.
    inner["_io_backed"] = True
    if cc_by_year is not None:
        inner["_effective_cc_by_year"] = (
            cc_by_year  # per-year [ns] cost, consumed as-is by _run_open
        )
    # Price-FREE emission intensity (review P1 round 14): the effective cost at a UNIT carbon price,
    # so covered emissions are well-defined every year including zero-price. Honours gases/coverage,
    # not the price level.
    inner["_emission_intensity"] = _open_intensity_from_io(io, sat, carbon_shocks, home, sectors)
    # Record the satellite's identity whenever one was actually consulted, even if the effective
    # cost it produced is zero (coverage excludes home, or genuinely zero intensity there) — the
    # manifest should reflect that a real satellite was read, not just that pricing happened to be
    # nonzero (review P1 follow-up).
    if sat is not None:
        inner["_emissions_provenance"] = _sat_identity(sat)
    return _run_open(meta, inner, shocks, years)


def _open_intensity_from_io(io, sat, carbon_shocks, home, sectors):
    """Price-free per-sector emission intensity for the open IO path (review P1 round 14): the
    effective cost at a UNIT carbon price (gases/coverage honoured, price removed). Returns None if
    nothing to weight."""
    if not carbon_shocks or sat is None:
        return None
    unit_shocks = [s.model_copy(update={"price": 1.0, "path": None}) for s in carbon_shocks]
    out = _open_effective_cc_from_io(io, sat, unit_shocks, home, sectors, [0])
    if out is None:
        return None
    v = out[0]
    return v if np.any(v != 0.0) else None


def _open_effective_cc_from_io(io, sat, carbon_shocks, home, sectors, years):
    """Effective per-year carbon-cost vector for the HOME region: ``{year: cc[ns]}`` or None if not
    priced. Reuses Engine 1's ``carbon_cost_vector`` (units/gases/coverage/paths/1e-6 scaling) PER
    YEAR on the home region's labels, then aggregates to sectors output-weighted — the same recipe
    the closed path uses. The result already includes the carbon price, so the caller must not
    re-multiply it (review P0)."""
    positive = any(s.price_at(y) > 0 for s in carbon_shocks for y in years)
    if not positive or sat is None:
        return None
    from cge.engines.io_price.engine import carbon_cost_vector

    _assert_cge_units(io, sat)
    labels = list(io.A.columns)
    home_mask = np.array([lb.split(":", 1)[0] == home for lb in labels])
    A = io.A.to_numpy(dtype=float)
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
    x = np.linalg.solve(np.eye(A.shape[0]) - A, fd)
    s_index = {s: k for k, s in enumerate(sectors)}
    out: dict[int, np.ndarray] = {}
    for year in years:
        cost, _descs = carbon_cost_vector(
            carbon_shocks, sat, labels, year
        )  # already price-included
        num = np.zeros(len(sectors))
        den = np.zeros(len(sectors))
        for lb, c_i, xi, is_home in zip(labels, cost, x, home_mask, strict=True):
            if not is_home:
                continue
            k = s_index[lb.split(":", 1)[1]]
            num[k] += c_i * xi
            den[k] += xi
        out[year] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out if any(np.any(v != 0.0) for v in out.values()) else None


def _sat_identity(sat) -> dict:
    """SatelliteAccount source/version/generation identity for the manifest (review P1: it was
    dropped when the IO-backed open path delegated to _run_open)."""
    from cge.engines.io_price.engine import _df_fingerprint

    return input_identity("SatelliteAccount", sat.provenance, content=_df_fingerprint(sat.data))


def _run_open(meta, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
    """Open-economy (Armington/CET) CGE run. The SAM has ``a_<s>``/``c_<s>`` activity/commodity
    accounts, factors, a household and a ``ROW`` account. Carbon cost is a supplied per-sector
    dimensionless ``carbon_cost_share`` (either supplied directly, or built from an IOSystem by
    ``_run_open_from_io``)."""
    from cge.engines.cge_static import model_open as MO
    from cge.engines.cge_static.calibrate_open import calibrate_open

    sam: SAM = data["SAM"]
    sectors = [a[2:] for a in sam.accounts if a.startswith("a_")]
    factors = [f for f in _DEFAULT_FACTORS if f in sam.accounts]
    ns = len(sectors)

    # Gate the supplied SAM BEFORE calibration (review P1: the open path skipped this).
    _validate_open_sam(sam, sectors, factors)

    carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
    prod_shocks = [s for s in shocks if isinstance(s, ProductivityShock)]
    _assert_no_region_scoped_productivity(prod_shocks, "open")
    # Two cost sources: an EFFECTIVE per-year cost from the IO path (already price-included — used
    # verbatim, review P0), or a supplied dimensionless share re-scaled by the price per year.
    io_backed = bool(data.get("_io_backed"))
    eff_by_year = data.get("_effective_cc_by_year")
    io_intensity = data.get(
        "_emission_intensity"
    )  # price-free intensity from the IO path (or None)
    # Apply/reject every CarbonPrice control (review P1: the open path ignored gases + coverage).
    # The supplied-SAM path carries a single dimensionless cost share, so — like the supplied-SAM
    # closed path — it cannot express gas selection or spatial coverage; reject them rather than
    # silently returning the same result regardless of the control. The IO-backed path is different:
    # its effective cost came from carbon_cost_vector, which HONOURS gases/coverage/paths, so those
    # controls are legitimate there and must not be rejected (review P0 follow-up). This branches on
    # io_backed, NOT on eff_by_year: a legitimate coverage selection can leave the home region
    # unpriced (empty effective cost) and must not fall back into the supplied-SAM rejection.
    if not io_backed:
        for s in carbon_shocks:
            if s.gases != ["CO2"]:
                raise ValueError(
                    f"the open-economy CGE applies a single dimensionless carbon_cost_share and "
                    f"cannot select gases; got gases={s.gases}. Use an IOSystem+satellite build "
                    f"for gas selection, or leave gases at the default ['CO2']."
                )
            if s.coverage_sectors or s.coverage_regions:
                raise ValueError(
                    "the open-economy CGE cannot apply sector/region coverage on a supplied SAM "
                    "(the cost share is already per-sector and single-region); express coverage "
                    "via carbon_cost_share, or use an IOSystem+satellite build."
                )

    modes = {s.revenue_recycling for s in carbon_shocks} or {"none"}
    if len(modes) > 1:
        raise ValueError(f"cge_static needs a single recycling mode; got {sorted(modes)}.")
    requested_recycling = modes.pop()

    positive_price = any(s.price_at(y) > 0 for s in carbon_shocks for y in years)
    if eff_by_year is not None:
        share = None
        emissions_priced = any(np.any(v != 0.0) for v in eff_by_year.values())
    elif io_backed:
        # IO-backed but nothing priced (no satellite, zero price, or coverage excludes the home
        # region): a zero cost is the honest answer — do not demand a carbon_cost_share.
        share = np.zeros(ns)
        emissions_priced = False
    else:
        share = _carbon_cost_share(data, sectors)
        if positive_price and share is None:
            raise ValueError(
                "a positive carbon price on the open-economy CGE requires a 'carbon_cost_share'."
            )
        share = share if share is not None else np.zeros(ns)
        emissions_priced = bool(positive_price and np.any(share != 0.0))

    # Only a scenario that actually generates carbon revenue can be "defaulted from none" — and
    # only such a scenario should actually HAVE its recycling mode switched (review P2: the switch
    # previously ran whenever requested_recycling=="none" regardless of emissions_priced, so a
    # zero-impact run's manifest could report recycling_mode="lump_sum" alongside
    # recycling_defaulted_from_none=False — internally contradictory, even though it made no
    # numerical difference since derive_open_state already gates the recycling fixed-point on
    # cc != 0).
    recycling = requested_recycling
    recycling_defaulted = requested_recycling == "none" and emissions_priced
    if recycling_defaulted:
        recycling = "lump_sum"  # the open economy also circulates carbon revenue to the household

    # Government (Phase 5d.1) and savings-investment (Phase 5d.2) accounts: same by-name
    # convention as the closed variant. The household is the unique remaining institution
    # (validated in _validate_open_sam).
    special = {_GOV_ACCOUNT, _SAVINV_ACCOUNT} & set(sam.accounts)
    institutions = None
    if special:
        hoh = next(
            a
            for a in sam.accounts
            if not a.startswith(("a_", "c_"))
            and a not in factors
            and a != "ROW"
            and a not in special
        )
        institutions = {"household": hoh}
        if _GOV_ACCOUNT in special:
            institutions["government"] = _GOV_ACCOUNT
        if _SAVINV_ACCOUNT in special:
            institutions["savings_investment"] = _SAVINV_ACCOUNT
    inv_closure = data.get("inv_closure", "savings_driven")
    if inv_closure not in ("savings_driven", "fixed_real"):
        raise ValueError(
            f"unsupported inv_closure {inv_closure!r}; use 'savings_driven' or 'fixed_real'."
        )
    gov_closure = data.get("gov_closure", "balanced_budget")
    if gov_closure != "balanced_budget":
        # The deficit_financed government closure is closed-variant only for now (the open/multi
        # generalisation of the fiscal-residual absorption is a documented follow-up).
        raise ValueError(
            f"gov_closure={gov_closure!r} is not yet supported in the open-economy variant "
            "(closed variant only); use 'balanced_budget'."
        )
    # Trade closure (Phase 5d.7): fixed_foreign_savings (default — er adjusts) or flexible_balance
    # (er fixed at 1, Sf adjusts so the current account balances freely).
    trade_closure = data.get("trade_closure", "fixed_foreign_savings")
    if trade_closure not in ("fixed_foreign_savings", "flexible_balance"):
        raise ValueError(
            f"unsupported trade_closure {trade_closure!r}; use 'fixed_foreign_savings' "
            "(default) or 'flexible_balance'."
        )

    cal = calibrate_open(
        sam,
        sectors=sectors,
        factors=factors,
        va_elast=data.get("va_elast", 1.0),
        arm_elast=data.get("armington_elast", 2.0),
        cet_elast=data.get("cet_elast", 2.0),
        institutions=institutions,
        energy_sectors=data.get("energy_sectors"),  # Phase 5d.5 (opt-in KL-E-M nest)
        energy_elasticities=data.get("energy_elasticities"),
    )
    if inv_closure != "savings_driven" and not cal.has_investment:
        raise ValueError(
            f"inv_closure={inv_closure!r} needs a {_SAVINV_ACCOUNT!r} account in the SAM; "
            "this SAM has none, so the closure choice would silently do nothing."
        )
    if gov_closure != "balanced_budget" and not cal.has_government:
        raise ValueError(
            f"gov_closure={gov_closure!r} needs a {_GOV_ACCOUNT!r} account in the SAM; none there."
        )

    # Under flexible_balance the last unknown is Sf (not er); seed it at the benchmark foreign
    # savings so the solve starts near the benchmark.
    def _guess():
        g = MO.initial_guess(cal)
        if trade_closure == "flexible_balance":
            g = g.copy()
            g[-1] = cal.foreign_savings
        return g

    # Foreign savings Sf is economically SIGNED (a trade surplus is Sf < 0). Under flexible_balance
    # Sf is the last solver unknown, so it must NOT carry the default positivity floor — a benchmark
    # exporter (Sf<0) or a shock that pushes the current account into surplus would otherwise be
    # infeasible (review P1, 2026-07-26). Relax the last bound to effectively −∞; every other
    # unknown (prices, er) keeps the 1e-9 floor.
    # The solver's log-transform z=log(x−lower) needs a WELL-SCALED lower bound: a huge offset
    # (−1e12) makes tiny z-steps map to enormous Sf swings and destroys conditioning near the small
    # benchmark Sf. Sf is a GDP-share (|Sf| ≪ 1 in practice); a floor of −1.0 (a capital outflow of
    # a full year's GDP — comfortably beyond any realistic surplus) is generous yet keeps
    # log(Sf+1) well-scaled around the benchmark.
    def _lower():
        if trade_closure != "flexible_balance":
            return None
        lo = np.full(MO.n_unknowns(cal), 1e-9)
        lo[-1] = -1.0  # Sf signed, well-scaled floor (a surplus is Sf<0)
        return lo

    def _solve_year(cc, theta=None):
        sol = solve(
            lambda z: MO.residuals(
                cal,
                z,
                carbon_cost=cc,
                recycling=recycling,
                inv_closure=inv_closure,
                gov_closure=gov_closure,
                trade_closure=trade_closure,
                productivity=theta,
            ),
            _guess(),
            lower=_lower(),
            prefer="scipy",
        )
        # strict=True: enforce the recycling k<1 feasibility guard on the ACCEPTED equilibrium
        # (the residual evaluations inside solve() ran non-strict so the solve stays continuous).
        flexible = trade_closure == "flexible_balance"
        er = 1.0 if flexible else float(sol.x[-1])
        fs = float(sol.x[-1]) if flexible else None
        st = MO.derive_open_state(
            cal,
            sol.x[:ns],
            sol.x[ns : 2 * ns],
            sol.x[2 * ns : 2 * ns + len(factors)],
            er,
            carbon_cost=cc,
            recycling=recycling,
            strict=True,
            inv_closure=inv_closure,
            gov_closure=gov_closure,
            foreign_savings=fs,
            productivity=theta,
        )
        return sol, st

    # Per-year per-sector productivity multiplier θ (Phase 6.4 GE tier). The open economy is
    # single-region (home + ROW), so a ProductivityShock is matched by SECTOR — same as the closed
    # variant. None when no productivity shock, so the run is byte-identical to a pure carbon run.
    theta_by_year = (
        {y: _productivity_by_sector(prod_shocks, sectors, y) for y in years}
        if prod_shocks
        else None
    )

    _bsol, base = _solve_year(np.zeros(ns))  # benchmark: θ=None by construction
    # Universal post-calibration replication gate (review P1): refuse a balanced-but-unsupported SAM
    # whose benchmark does not reproduce the calibrated quantities (see _assert_open_replicates).
    _assert_open_replicates(cal, base, _bsol.x, trade_closure=trade_closure)
    records: list[dict] = []
    resid_max = _bsol.residual_norm
    backends: set[str] = {_bsol.backend}
    statuses: set[str] = {_bsol.status}
    cc_by_year: dict[int, np.ndarray] = {}
    for year in years:
        if eff_by_year is not None:
            cc = eff_by_year[year]  # already price-included (IO path); do NOT re-multiply
        else:
            tau = sum(s.price_at(year) for s in carbon_shocks)
            cc = tau * share
        cc_by_year[year] = cc
        theta = theta_by_year[year] if theta_by_year is not None else None
        sol, st = _solve_year(cc, theta)
        resid_max = max(resid_max, sol.residual_norm)
        backends.add(sol.backend)
        statuses.add(sol.status)
        # Price-free intensity for covered emissions (review P1 round 14): the IO path's unit-price
        # intensity, else the supplied share (toy path). Both are price-independent, so covered
        # emissions are emitted every year including a zero-price year.
        emission_intensity = io_intensity if io_backed else share
        _emit_open(records, cal, base, st, year, cc=cc, intensity=emission_intensity)

    # Substantive provenance: the effective per-year carbon-cost vector (hashed) + the full
    # per-sector elasticity vectors, so two runs that differ only in carbon shares or in an
    # elasticity vector produce different manifests (review P1).
    effective = {str(y): [round(float(v), 12) for v in cc.tolist()] for y, cc in cc_by_year.items()}
    emissions_inputs: list = []
    if any(any(v != 0.0 for v in row) for row in effective.values()):
        emissions_inputs.append(
            {
                "name": "EffectiveCarbonCost"
                if eff_by_year is not None
                else "EffectiveCarbonCostShare",
                "sectors": sectors,
                "content_hash": content_hash(effective),
            }
        )
    # Restore the SatelliteAccount source/version/generation identity on the IO-backed path (review
    # P1: it was dropped when _run_open_from_io delegated).
    if data.get("_emissions_provenance") is not None:
        emissions_inputs.append(data["_emissions_provenance"])
    manifest = RunManifest.build(
        engine_name=meta.name,
        engine_version=meta.version,
        data_source=data_source_id(sam.provenance),
        scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
        assumptions={
            **OPEN_ASSUMPTIONS,
            "effective_config": _effective_public_config(
                data, "open", "io" if io_backed else "supplied"
            ),
            "sectors": sectors,
            "factors": factors,
            "recycling_mode": recycling,
            "recycling_defaulted_from_none": recycling_defaulted,
            # FULL per-sector elasticity vectors, not just the first element (review P1).
            "armington_elasticity": [round(float(v), 12) for v in cal.arm_elast.tolist()],
            "cet_elasticity": [round(float(v), 12) for v in cal.cet_elast.tolist()],
            "va_elast": [round(float(v), 12) for v in cal.va_elast.tolist()],
            "value_added_nest": _va_nest_description(cal.va_elast),
            # Energy nest (Phase 5d.5): flat Leontief or the KL-E-M nest (shared helper).
            **_energy_manifest(cal, sectors),
            "solver_backends": sorted(backends),
            "solver_statuses": sorted(statuses),
            "solver_max_residual_norm": resid_max,
            "foreign_savings": float(cal.foreign_savings),
            # Trade closure (Phase 5d.7): fixed_foreign_savings (er adjusts) or flexible_balance
            # (er fixed, Sf adjusts).
            "trade_closure": trade_closure,
            # Government account (Phase 5d.1) — same keys as the closed variant's manifest.
            "government_account": _GOV_ACCOUNT if cal.has_government else "none",
            "gov_closure": gov_closure if cal.has_government else "n/a (no government)",
            "gov_benchmark_tax_share_of_factor_income": (
                round(float(cal.gov_tax_rate0), 12) if cal.has_government else 0.0
            ),
            # Savings-investment account (Phase 5d.2). With it, er·Sf routes into the investment
            # pool (investment = savings + er·Sf), not household income.
            "savings_investment_account": _SAVINV_ACCOUNT if cal.has_investment else "none",
            "inv_closure": inv_closure if cal.has_investment else "n/a (no investment)",
            "benchmark_savings_rate_of_disposable_income": (
                round(float(cal.sav_rate0), 12) if cal.has_investment else 0.0
            ),
            "emissions_priced": emissions_priced,
            "benchmark_gdp_normalised": cal.gdp0,
            # SAM credibility surface when the open SAM was built from an IOSystem (None when a SAM
            # was supplied directly and validated separately).
            "sam_quality": (
                {"worst": _q.worst.value, "summary": _q.summary()}
                if (_q := data.get("_sam_quality")) is not None
                else "supplied directly (validated: aligned, finite, non-negative, balanced)"
            ),
            "inputs": [
                input_identity("SAM", sam.provenance, content=_sam_fingerprint(sam)),
                *emissions_inputs,
            ],
        },
    )
    return ResultSet.from_records(records, manifest)


def _emit_open(records, cal, base, st, year: int, cc=None, intensity=None) -> None:
    """Emit open-economy results: activity output (volume), domestic/import/export volumes, the
    composite price, factor prices, exchange rate, real GDP, welfare and carbon revenue. ``cc`` is
    the year's per-sector carbon cost; ``intensity`` the price-free emission intensity."""
    for i, sector in enumerate(cal.sectors):
        records.append(_rec("price_change", sector, year, st.pq[i] / base.pq[i] - 1.0))
        records.append(_rec("volume_change", sector, year, st.Z[i] / base.Z[i] - 1.0))
        records.append(_rec("import_change", sector, year, _ratio(st.M[i], base.M[i])))
        records.append(_rec("export_change", sector, year, _ratio(st.E[i], base.E[i])))
        records.append(
            _rec("gva_change", sector, year, _gva_income(st, i) / _gva_income(base, i) - 1.0)
        )
        records.append(
            _rec("gva_income_change", sector, year, _gva_income(st, i) / _gva_income(base, i) - 1.0)
        )
        records.append(
            _rec("gva_volume_change", sector, year, _gva_volume(st, i) / _gva_volume(base, i) - 1.0)
        )
    for f_idx, factor in enumerate(cal.factors):
        records.append(_rec("factor_price_change", factor, year, st.w[f_idx] / base.w[f_idx] - 1.0))
        named = {"LAB": "wage_change", "CAP": "capital_return_change"}.get(factor)
        if named is not None:
            records.append(_rec(named, factor, year, st.w[f_idx] / base.w[f_idx] - 1.0))
        emp = float(st.F[f_idx, :].sum())
        emp_base = float(base.F[f_idx, :].sum())
        var = "employment_change" if factor == "LAB" else f"factor_use_change_{factor}"
        records.append(_rec(var, factor, year, emp / emp_base - 1.0))
    records.append(_rec("exchange_rate_change", "__economy__", year, st.er / base.er - 1.0))
    # Foreign savings / current account (Phase 5d.7): under flexible_balance Sf is endogenous, so
    # report the current account adjusting instead of the exchange rate. Sf is economically SIGNED
    # (a surplus is Sf<0) and is already GDP-normalised, so it is reported as a LEVEL (a share of
    # benchmark GDP) and a percentage-POINT change in that share — NOT a percentage ratio (review
    # P1, 2026-07-26): a ratio is undefined/misleading when the benchmark is non-positive or the
    # balance crosses zero (_ratio returns 0 for a non-positive base). Emitted only when Sf actually
    # moved, so fixed-Sf runs stay byte-identical.
    if abs(st.foreign_savings - base.foreign_savings) > 1e-12:
        records.append(
            _rec("foreign_savings_gdp_share", "__economy__", year, float(st.foreign_savings))
        )
        records.append(
            _rec(
                "foreign_savings_change_pp",
                "__economy__",
                year,
                float(st.foreign_savings - base.foreign_savings),
            )
        )

    # Real GDP — expenditure-side: consumption + net exports (review P1: Σ pq_i·FD_i alone is
    # household CONSUMPTION, not GDP; it only coincides with GDP when the current account is zero.
    # With non-zero foreign savings — which this model explicitly supports via the ROW capital
    # transfer er·Sf — GDP = C + (X − M), valued at their respective CPI-numéraire prices: exports
    # and imports both trade at the world price in domestic currency, er·pworld = er (pworld=1), so
    # X = er·Σ E_i and M = er·Σ M_i. Prices are in CPI units (Π pq^γ=1 pinned), so this expenditure
    # aggregate is already real. Reproduced and verified: a balanced-deficit SAM's consumption-only
    # figure diverges from the correct C+X−M change (the prior code silently used the special case
    # Sf=0 as if it were the general identity — the existing GDP==welfare test only exercised a
    # zero-current-account fixture).
    # GDP = C + G + I + (X − M) (review P1 round 13: VOLUME at benchmark prices, consistent with
    # closed/multi). Exports/imports both trade at the world price er·pworld = er in domestic
    # currency; at benchmark er=1 and pq=1, so the volume aggregate values every component at 1.
    # Current-price GDP values them at the run's pq and er; the deflator is current/volume.
    def _open_gdp(state, price_er, price_pq):
        absorption = float(np.dot(price_pq, state.FD + state.GD + state.ID))
        net_exports = price_er * float(state.E.sum() - state.M.sum())
        return absorption + net_exports

    base_pq = base.pq
    gdp_vol = _open_gdp(st, base.er, base_pq)  # quantities at benchmark prices
    gdp_vol_base = _open_gdp(base, base.er, base_pq)
    gdp_cur = _open_gdp(st, st.er, st.pq)  # current prices
    gdp_cur_base = gdp_vol_base  # benchmark current == benchmark volume
    vol_ratio = gdp_vol / gdp_vol_base
    cur_ratio = gdp_cur / gdp_cur_base
    records.append(_rec("gdp_change_real", "__economy__", year, vol_ratio - 1.0))
    records.append(_rec("gdp_change_nominal", "__economy__", year, cur_ratio - 1.0))
    records.append(_rec("gdp_deflator_change", "__economy__", year, cur_ratio / vol_ratio - 1.0))
    u = float(np.prod(np.power(st.FD, cal.gamma)))
    u_base = float(np.prod(np.power(base.FD, cal.gamma)))
    records.append(_rec("welfare_change", "__economy__", year, u / u_base - 1.0))
    records.append(_rec("carbon_revenue", "__economy__", year, st.carbon_revenue / cal.gdp0))
    # Household consumption VOLUME (benchmark prices), same convention as GDP.
    cons_vol = float(np.dot(base_pq, st.FD))
    cons_vol_base = float(np.dot(base_pq, base.FD))
    records.append(_rec("consumption_change", "__economy__", year, cons_vol / cons_vol_base - 1.0))
    _emit_covered_emissions(records, intensity, base.Z, st.Z, year)
    if cal.has_energy_nest:
        eidx = cal.energy_nest.energy_idx
        eu, eu_base = float(st.Z[eidx].sum()), float(base.Z[eidx].sum())
        records.append(
            _rec("energy_sector_output_volume_change", "__economy__", year, eu / eu_base - 1.0)
        )
    # Government account (Phase 5d.1): same emission convention as the closed variant — only when
    # a government exists, so no-government output stays byte-identical to pre-5d.1.
    if cal.has_government:
        records.append(_rec("fiscal_balance", "__economy__", year, st.fiscal_balance / cal.gdp0))
        gov_spend = float(np.dot(st.pq, st.GD))
        records.append(_rec("gov_spending", "__economy__", year, gov_spend / cal.gdp0))
    # Savings-investment account (Phase 5d.2): nominal investment and household savings, shares of
    # benchmark GDP. Investment = savings + er·Sf under savings_driven (the open S-I identity —
    # they differ by exactly the foreign-savings inflow, unlike the closed variant's S=I).
    if cal.has_investment:
        inv_spend = float(np.dot(st.pq, st.ID))
        records.append(_rec("investment", "__economy__", year, inv_spend / cal.gdp0))
        records.append(_rec("savings", "__economy__", year, st.savings / cal.gdp0))


def _ratio(x: float, x0: float) -> float:
    return float(x / x0 - 1.0) if x0 > 0 else 0.0


# ---------------------------------------------------------------------------
# Multi-region variant (Phase 5.4 — true bilateral trade)
# ---------------------------------------------------------------------------


def _is_multi_region_sam(sam: SAM) -> bool:
    """A multi-region SAM has several ``HOH_<r>`` households and region-prefixed ``a_<r>_<s>``
    activities (distinguishing it from the single-region open SAM's single ``HOH`` + ``ROW``)."""
    households = [a for a in sam.accounts if a.startswith("HOH_")]
    region_activities = [a for a in sam.accounts if a.startswith("a_") and a.count("_") >= 2]
    return len(households) >= 2 and len(region_activities) > 0


def _infer_multi_axes(sam: SAM) -> tuple[list[str], list[str]]:
    """Infer (regions, sectors) from a supplied multi-region SAM UNAMBIGUOUSLY (review P2,
    2026-07-27). Regions are the full suffix of each ``HOH_<r>`` account (strip only the ``HOH_``
    prefix — never split on a further ``_``, so a region like ``Rest_of_World`` survives intact).
    Sectors are what remains of each ``a_<r>_<s>`` activity after its known ``a_<r>_`` prefix.

    Raises if an activity does not start with any known ``a_<r>_`` prefix, or if the region names
    are prefix-ambiguous (one region name is a prefix of another, so ``a_<r>_<s>`` could split two
    ways). The IO path avoids all of this by passing the builder's exact axes."""
    regions = sorted({a[len("HOH_") :] for a in sam.accounts if a.startswith("HOH_")})
    if len(regions) < 2:
        raise ValueError(f"multi-region SAM needs ≥2 HOH_<r> households; found regions {regions}")
    # Prefix-ambiguity guard: if region A's name is a prefix of region B's, then a_<B>_<s> could be
    # mis-read as region A with sector "<B-suffix>_<s>". Require the caller to pass explicit axes.
    for ri in regions:
        for rj in regions:
            if ri != rj and rj.startswith(ri + "_"):
                raise ValueError(
                    f"region names are prefix-ambiguous ({ri!r} is a prefix of {rj!r}), so "
                    "a_<r>_<s> account names cannot be split uniquely; pass explicit 'regions' and "
                    "'sectors' in data."
                )
    sectors: set[str] = set()
    for a in sam.accounts:
        if not a.startswith("a_"):
            continue
        matched = [r for r in regions if a.startswith(f"a_{r}_")]
        if not matched:
            raise ValueError(
                f"activity account {a!r} does not start with any known 'a_<r>_' prefix for regions "
                f"{regions}; the SAM's activity/household naming is inconsistent."
            )
        # Longest matching region prefix wins (defensive; prefix-ambiguity already rejected above).
        r = max(matched, key=len)
        sectors.add(a[len(f"a_{r}_") :])
    return regions, sorted(sectors)


def _rec_r(variable: str, sector: str, region: str, year: int, value: float) -> dict:
    return {
        "variable": variable,
        "sector": sector,
        "region": region,
        "year": year,
        "scenario": "central",
        "value": float(value),
    }


def _validate_multi_sam(sam: SAM, regions: list[str], sectors: list[str], factors: list[str]):
    """Structural gate for a supplied multi-region SAM: the a_<r>_<s>/c_<r>_<s> accounts, per-region
    factors and households must exist, axes unique/aligned, cells finite non-negative, and the
    matrix globally balanced."""
    from cge.data.sam.balance import imbalance, is_balanced

    need = []
    for r in regions:
        need += [f"a_{r}_{s}" for s in sectors] + [f"c_{r}_{s}" for s in sectors]
        need += [f"{f}_{r}" for f in factors] + [f"HOH_{r}"]
    missing = [x for x in need if x not in sam.accounts]
    if missing:
        raise ValueError(f"multi-region SAM is missing required accounts: {missing[:6]}...")
    m = sam.matrix
    idx, cols = list(m.index), list(m.columns)
    # Duplicate-label guard (review P2, 2026-07-27): set-equality alone silently tolerates a
    # repeated axis label (two rows both named 'HOH_N'), which then makes .loc reads ambiguous and
    # the balance check meaningless. Check LENGTH and membership so a duplicated label is caught.
    for axis_name, axis in (("row", idx), ("column", cols)):
        dups = sorted({a for a in axis if axis.count(a) > 1})
        if dups:
            raise ValueError(f"multi-region SAM has duplicate {axis_name} labels: {dups[:6]}")
    if set(idx) != set(sam.accounts) or set(cols) != set(sam.accounts):
        raise ValueError("multi-region SAM matrix axes must equal the declared accounts exactly")
    # Declared-accounts uniqueness (the SAM contract lists accounts as a plain list).
    acc_dups = sorted({a for a in sam.accounts if sam.accounts.count(a) > 1})
    if acc_dups:
        raise ValueError(f"multi-region SAM has duplicate declared accounts: {acc_dups[:6]}")
    arr = m.to_numpy(dtype=float)
    if not np.isfinite(arr).all() or float(arr.min()) < -1e-9:
        raise ValueError("multi-region SAM has non-finite or negative cells")
    if not is_balanced(m, tol=1e-6):
        worst = float(imbalance(m).abs().max())
        raise ValueError(f"multi-region SAM is not balanced (max |row−col| = {worst:.3e} > 1e-6)")


def _run_multi_from_io(meta, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
    """Multi-region CGE on a **real build** (Phase 5.1b): construct an R-region SAM from the given
    ``IOSystem`` via ``build_multi_sam`` (bilateral trade among ALL build regions, topology- and
    quality-gated), derive the per-(region, sector) effective carbon cost from the satellite the
    SAME way Engine 1 / the open IO path does, then delegate to ``_run_multi`` with the built SAM +
    effective cost injected.

    The multi analogue of ``_run_open_from_io``. ``data`` keys: ``IOSystem``, ``SatelliteAccount``,
    plus optional ``capital_share`` and ``multi_materiality`` (dust threshold). Unlike the open path
    there is no ``home_region`` — every build region is modelled as a genuine region (the global
    economy is closed with no external rest-of-world)."""
    from cge.data.sam import build_multi_sam

    io: IOSystem = data["IOSystem"]
    sat = data.get("SatelliteAccount")
    kwargs = {}
    if "capital_share" in data:
        kwargs["capital_share"] = data["capital_share"]
    if "multi_materiality" in data:
        kwargs["materiality"] = data["multi_materiality"]
    sam, quality, regions, sectors = build_multi_sam(io, **kwargs)
    if quality.worst.value == "fail":
        raise ValueError(
            f"multi-region SAM from {io.provenance.build_id} failed quality gates: "
            f"{quality.summary()}"
        )

    carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
    # Same missing-satellite gate as the closed/open IO paths: a positive price with no satellite
    # would silently be a zero-impact run.
    positive_price = any(s.price_at(y) > 0 for s in carbon_shocks for y in years)
    if positive_price and sat is None:
        raise ValueError(
            "a positive carbon price on an IO-backed multi-region CGE run requires a "
            "'SatelliteAccount' (emission intensities); none supplied — the run would be "
            "silently zero-impact."
        )
    if carbon_shocks:
        from cge.engines.io_price.engine import _assert_coverage_labels

        _assert_coverage_labels(carbon_shocks, io)
    cc_by_year = _multi_effective_cc_from_io(io, sat, carbon_shocks, regions, sectors, years)

    inner = {k: v for k, v in data.items() if k not in ("IOSystem", "SatelliteAccount")}
    inner["SAM"] = sam
    inner["_sam_quality"] = quality
    inner["_io_backed"] = True
    # Pass the builder's regions/sectors EXPLICITLY (review P2, 2026-07-27): a_<r>_<s> account names
    # cannot be uniquely re-split when a region or sector name itself contains '_' (e.g.
    # 'Rest_of_World'), so the engine must not string-split them back out — it takes the exact axes
    # the builder used to construct the SAM.
    inner["regions"] = list(regions)
    inner["sectors"] = list(sectors)
    if cc_by_year is not None:
        inner["_effective_cc_by_year"] = cc_by_year  # per-year [nr, ns] cost, consumed as-is
    # Price-FREE per-(region,sector) intensity (review P1 round 14): unit-price effective cost, so
    # covered emissions are emitted every year including zero-price.
    if sat is not None and carbon_shocks:
        unit = [s.model_copy(update={"price": 1.0, "path": None}) for s in carbon_shocks]
        iens = _multi_effective_cc_from_io(io, sat, unit, regions, sectors, [0])
        if iens is not None:
            inner["_emission_intensity"] = iens[0]
    if sat is not None:
        inner["_emissions_provenance"] = _sat_identity(sat)
    return _run_multi(meta, inner, shocks, years)


def _multi_effective_cc_from_io(io, sat, carbon_shocks, regions, sectors, years):
    """Effective per-year carbon-cost matrix ``{year: cc[nr, ns]}`` (or None if not priced) for the
    multi-region build. Reuses Engine 1's ``carbon_cost_vector`` (units/gases/coverage/paths/1e-6
    scaling) PER YEAR over every region's labels, aggregated to (region, sector) output-weighted —
    the per-(region,sector) analogue of the open path's home-only ``_open_effective_cc_from_io``.
    The result already includes the carbon price, so the caller must not re-multiply it (see P0)."""
    positive = any(s.price_at(y) > 0 for s in carbon_shocks for y in years)
    if not positive or sat is None:
        return None
    from cge.engines.io_price.engine import carbon_cost_vector

    _assert_cge_units(io, sat)
    labels = list(io.A.columns)
    A = io.A.to_numpy(dtype=float)
    fd = io.final_demand.sum(axis=1).reindex(labels).fillna(0.0).to_numpy(dtype=float)
    x = np.linalg.solve(np.eye(A.shape[0]) - A, fd)
    r_index = {r: k for k, r in enumerate(regions)}
    s_index = {s: k for k, s in enumerate(sectors)}
    nr, ns = len(regions), len(sectors)
    out: dict[int, np.ndarray] = {}
    for year in years:
        cost, _descs = carbon_cost_vector(carbon_shocks, sat, labels, year)  # price-included
        num = np.zeros((nr, ns))
        den = np.zeros((nr, ns))
        for lb, c_i, xi in zip(labels, cost, x, strict=True):
            r, s = lb.split(":", 1)
            if r not in r_index or s not in s_index:
                continue
            ri, si = r_index[r], s_index[s]
            num[ri, si] += c_i * xi
            den[ri, si] += xi
        out[year] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out if any(np.any(v != 0.0) for v in out.values()) else None


def _run_multi(meta, data: dict, shocks: list[Shock], years: list[int]) -> ResultSet:
    """Multi-region (bilateral Armington/CET) CGE run. The SAM has ``a_<r>_<s>``/``c_<r>_<s>``
    accounts, per-region factors ``<f>_<r>`` and households ``HOH_<r>``. Carbon cost is a supplied
    per-(region, sector) dimensionless ``carbon_cost_share`` (a nested dict
    ``{region: {sector: v}}`` or a full [nr, ns] array)."""
    from cge.engines.cge_static import model_multi as MM
    from cge.engines.cge_static.calibrate_multi import calibrate_multi

    sam: SAM = data["SAM"]
    factors = list(_DEFAULT_FACTORS)
    # Region/sector axes. Prefer EXPLICIT axes (the IO path passes the builder's exact regions and
    # sectors; a supplied-SAM caller may pass them too) — a_<r>_<s> names cannot be uniquely
    # re-split when a name contains '_' (review P2, 2026-07-27). Without explicit axes, infer
    # UNAMBIGUOUSLY: regions are the whole suffix of HOH_<r> (strip only the 'HOH_' prefix, never
    # split on further '_'), then sectors are what remains after the known 'a_<r>_' prefix.
    if data.get("regions") is not None and data.get("sectors") is not None:
        regions, sectors = list(data["regions"]), list(data["sectors"])
    else:
        regions, sectors = _infer_multi_axes(sam)
    _validate_multi_sam(sam, regions, sectors, factors)
    nr, ns = len(regions), len(sectors)

    # IO-backed path (Phase 5.1b): _run_multi_from_io built the SAM and the EFFECTIVE per-year
    # carbon-cost matrix (already price × intensity × 1e-6, honouring gases/coverage) upstream, so
    # this run consumes it directly rather than re-multiplying by the price, exactly as the open
    # IO path does (review P0: re-applying the price double-counts).
    io_backed = bool(data.get("_io_backed"))
    eff_by_year = data.get("_effective_cc_by_year")

    carbon_shocks = [s for s in shocks if isinstance(s, CarbonPrice)]
    prod_shocks = [s for s in shocks if isinstance(s, ProductivityShock)]
    for s in carbon_shocks:
        # On the supplied-SAM path the multi CGE cannot select gases/coverage (the share is a bare
        # per-(region,sector) number); the IO path HAS honoured them while building eff_by_year.
        if not io_backed and (s.gases != ["CO2"] or s.coverage_sectors or s.coverage_regions):
            raise ValueError(
                "the multi-region CGE applies a per-(region,sector) carbon_cost_share and cannot "
                "select gases or apply coverage; express coverage via carbon_cost_share values."
            )
    modes = {s.revenue_recycling for s in carbon_shocks} or {"none"}
    if len(modes) > 1:
        raise ValueError(f"cge_static needs a single recycling mode; got {sorted(modes)}.")
    requested_recycling = modes.pop()

    share = _multi_carbon_share(data, regions, sectors)
    positive = any(s.price_at(y) > 0 for s in carbon_shocks for y in years)
    if positive and share is None and not io_backed:
        raise ValueError(
            "a positive carbon price on the multi-region CGE requires carbon_cost_share"
        )
    # Price-free emission intensity for covered emissions (review P1 round 14): the IO path's
    # unit-price intensity, else the supplied share — captured BEFORE ``share`` is zeroed below, so
    # the fallback is not lost (the earlier code replaced a None share with zeros first, so the
    # IO-path fallback never fired and covered emissions never emitted).
    emission_intensity = data.get("_emission_intensity") if io_backed else share
    share = share if share is not None else np.zeros((nr, ns))

    # Each region has one household, so labour_tax_cut ≡ lump_sum within a region (same aggregate
    # income). 'none' does not close (revenue would vanish), so it defaults to lump_sum — but only
    # when a positive-revenue scenario actually reaches here (review P2: the flag was always set,
    # and — round-9 follow-up — the switch itself must also only fire then, or a zero-impact run's
    # manifest can report recycling_mode="lump_sum" next to recycling_defaulted_from_none=False).
    io_priced = bool(
        eff_by_year is not None and any(np.any(v != 0.0) for v in eff_by_year.values())
    )
    emissions_priced = bool(positive and (io_priced if io_backed else np.any(share != 0.0)))
    recycling = requested_recycling
    recycling_defaulted = requested_recycling == "none" and emissions_priced
    if recycling_defaulted:
        recycling = "lump_sum"

    # Government (Phase 5d.1) and savings-investment (Phase 5d.2) accounts: GOV_<r>/SAVINV_<r>
    # per region, by the same naming convention as HOH_<r>. calibrate_multi enforces the
    # all-regions-or-none rule and the flow restrictions (incl. capital transfers routing between
    # the SAVINV accounts, not between households).
    has_gov_accounts = any(a.startswith("GOV_") for a in sam.accounts)
    has_savinv_accounts = any(a.startswith("SAVINV_") for a in sam.accounts)
    inv_closure = data.get("inv_closure", "savings_driven")
    if inv_closure not in ("savings_driven", "fixed_real"):
        raise ValueError(
            f"unsupported inv_closure {inv_closure!r}; use 'savings_driven' or 'fixed_real'."
        )
    if inv_closure != "savings_driven" and not has_savinv_accounts:
        raise ValueError(
            f"inv_closure={inv_closure!r} needs SAVINV_<r> accounts in the SAM; this SAM has "
            "none, so the closure choice would silently do nothing."
        )
    cal = calibrate_multi(
        sam,
        regions=regions,
        sectors=sectors,
        factors=factors,
        arm_elast=data.get("armington_elast", 2.0),
        cet_elast=data.get("cet_elast", 2.0),
        va_elast=data.get("va_elast", 1.0),
        government=has_gov_accounts,
        savings_investment=has_savinv_accounts,
        energy_sectors=data.get("energy_sectors"),  # Phase 5d.5 (opt-in KL-E-M nest per region)
        energy_elasticities=data.get("energy_elasticities"),
    )

    def _solve_year(cc, theta=None):
        sol = solve(
            lambda z: MM.residuals(
                cal,
                z,
                carbon_cost=cc,
                recycling=recycling,
                inv_closure=inv_closure,
                productivity=theta,
            ),
            MM.initial_guess(cal),
            prefer="scipy",
        )
        # strict=True enforces the recycling k<1 feasibility guard on the ACCEPTED equilibrium.
        st = MM.unpack_state(
            cal,
            sol.x,
            carbon_cost=cc,
            recycling=recycling,
            strict=True,
            inv_closure=inv_closure,
            productivity=theta,
        )
        return sol, st

    # Per-year per-(region,sector) productivity multiplier θ (Phase 6.4 GE tier). The multi CGE has
    # a region dimension, so a ProductivityShock's region coverage IS honoured (a nature
    # degradation built for one region hits only its sectors). None ⇒ byte-identical to a pure run.
    theta_by_year = (
        {y: _productivity_by_region_sector(prod_shocks, regions, sectors, y) for y in years}
        if prod_shocks
        else None
    )

    _bsol, base = _solve_year(np.zeros((nr, ns)))  # benchmark: θ=None
    # Universal post-calibration replication gate (review P1): refuse a balanced-but-unsupported SAM
    # whose benchmark does not reproduce the calibrated quantities.
    _assert_multi_replicates(cal, base, _bsol.x)
    records: list[dict] = []
    resid_max = _bsol.residual_norm
    backends, statuses = {_bsol.backend}, {_bsol.status}
    cc_by_year: dict[int, np.ndarray] = {}
    for year in years:
        if io_backed and eff_by_year is not None:
            # Effective cost is already price-included per year (may be absent for a year the
            # coverage excludes → zero); consume as-is, do NOT multiply by the price again.
            cc = eff_by_year.get(year, np.zeros((nr, ns)))
        else:
            tau = sum(s.price_at(year) for s in carbon_shocks)
            cc = tau * share
        cc_by_year[year] = cc
        theta = theta_by_year[year] if theta_by_year is not None else None
        sol, st = _solve_year(cc, theta)
        resid_max = max(resid_max, sol.residual_norm)
        backends.add(sol.backend)
        statuses.add(sol.status)
        # Price-free per-(region,sector) intensity for covered emissions (review P1 round 14) —
        # captured above, so covered emissions emit every year including zero-price.
        _emit_multi(records, cal, base, st, year, cc=cc, intensity=emission_intensity)

    # Substantive provenance: the effective per-year carbon-cost matrix (hashed) so two runs that
    # differ only in the carbon shares get different manifests (review P1).
    effective = {
        str(y): [[round(float(v), 12) for v in row] for row in cc.tolist()]
        for y, cc in cc_by_year.items()
    }
    emissions_inputs = (
        [
            {
                "name": "EffectiveCarbonCostMatrix",
                "regions": regions,
                "sectors": sectors,
                "content_hash": content_hash(effective),
            }
        ]
        if emissions_priced
        else []
    )
    # IO-backed path (Phase 5.1b): record the satellite identity actually consulted, like the open
    # path (surfaced even when the effective cost is zero — a real satellite was read).
    if data.get("_emissions_provenance") is not None:
        emissions_inputs.append(data["_emissions_provenance"])
    manifest = RunManifest.build(
        engine_name=meta.name,
        engine_version=meta.version,
        data_source=data_source_id(sam.provenance),
        scenario={"shocks": [s.model_dump(mode="json") for s in shocks], "years": years},
        assumptions={
            **MULTI_ASSUMPTIONS,
            "effective_config": _effective_public_config(
                data, "multi", "io" if io_backed else "supplied"
            ),
            "regions": regions,
            "sectors": sectors,
            "factors": factors,
            "recycling_mode": recycling,
            "recycling_defaulted_from_none": recycling_defaulted,
            "armington_elasticity": float(cal.arm_elast[0, 0]),
            "cet_elasticity": float(cal.cet_elast[0, 0]),
            "va_elast": float(cal.va_elast[0, 0]),
            "value_added_nest": _va_nest_description(cal.va_elast),
            # Energy nest (Phase 5d.5): flat Leontief or the per-region KL-E-M nest (shared helper).
            **_energy_manifest(cal, sectors),
            "solver_backends": sorted(backends),
            "solver_statuses": sorted(statuses),
            "solver_max_residual_norm": resid_max,
            "foreign_savings_by_region": [
                round(float(v), 12) for v in cal.foreign_savings.tolist()
            ],
            # Government accounts (Phase 5d.1) — same convention as closed/open, per region.
            "government_account": "GOV_<r> per region" if cal.has_government else "none",
            "gov_closure": "balanced_budget" if cal.has_government else "n/a (no government)",
            "gov_benchmark_tax_share_of_factor_income_by_region": (
                [round(float(v), 12) for v in cal.gov_tax_rate0.tolist()]
                if cal.has_government
                else []
            ),
            # Savings-investment accounts (Phase 5d.2): Sf_r routes into each region's investment
            # pool (investment_r = savings_r + Sf_r), not household income.
            "savings_investment_account": (
                "SAVINV_<r> per region" if cal.has_investment else "none"
            ),
            "inv_closure": inv_closure if cal.has_investment else "n/a (no investment)",
            "benchmark_savings_rate_of_disposable_income_by_region": (
                [round(float(v), 12) for v in cal.sav_rate0.tolist()] if cal.has_investment else []
            ),
            "emissions_priced": emissions_priced,
            "benchmark_gdp_normalised": cal.gdp0,
            # SAM credibility surface when the multi SAM was built from an IOSystem (Phase 5.1b);
            # None when a SAM was supplied directly and validated separately.
            "sam_quality": (
                {"worst": _q.worst.value, "summary": _q.summary()}
                if (_q := data.get("_sam_quality")) is not None
                else "supplied directly (validated: aligned, finite, non-negative, balanced)"
            ),
            "inputs": [
                input_identity("SAM", sam.provenance, content=_sam_fingerprint(sam)),
                *emissions_inputs,
            ],
        },
    )
    return ResultSet.from_records(records, manifest)


def _emit_multi(records, cal, base, st, year: int, cc=None, intensity=None) -> None:
    """Emit multi-region results, region-tagged: per (region, sector) price/volume/import/export/GVA
    change, per-region named factor prices + employment, real GDP and real_consumption_change,
    welfare, carbon revenue and emissions (as shares of / relative to that region's OWN benchmark).
    ``cc`` is the year's per-(region,sector) carbon cost for the emissions-change output."""
    for ri, region in enumerate(cal.regions):
        for si, sector in enumerate(cal.sectors):
            records.append(
                _rec_r("price_change", sector, region, year, st.pq[ri, si] / base.pq[ri, si] - 1.0)
            )
            records.append(
                _rec_r("volume_change", sector, region, year, st.Z[ri, si] / base.Z[ri, si] - 1.0)
            )
            records.append(
                _rec_r(
                    "import_change",
                    sector,
                    region,
                    year,
                    _ratio(st.M[ri, si, :].sum(), base.M[ri, si, :].sum()),
                )
            )
            records.append(
                _rec_r(
                    "export_change",
                    sector,
                    region,
                    year,
                    _ratio(st.EX[ri, si, :].sum(), base.EX[ri, si, :].sum()),
                )
            )
            # Sector GVA per region — INCOME (current factor payments Σ_f w·F) AND VOLUME (factor
            # quantity Σ_f F at benchmark prices), named separately (review P1 round 13).
            gva_inc = float(np.dot(st.w[:, ri], st.F[:, ri, si]))
            gva_inc_b = float(np.dot(base.w[:, ri], base.F[:, ri, si]))
            gva_vol = float(st.F[:, ri, si].sum())
            gva_vol_b = float(base.F[:, ri, si].sum())
            records.append(_rec_r("gva_change", sector, region, year, gva_inc / gva_inc_b - 1.0))
            records.append(
                _rec_r("gva_income_change", sector, region, year, gva_inc / gva_inc_b - 1.0)
            )
            records.append(
                _rec_r("gva_volume_change", sector, region, year, gva_vol / gva_vol_b - 1.0)
            )
        for fi, factor in enumerate(cal.factors):
            records.append(
                _rec_r(
                    "factor_price_change",
                    factor,
                    region,
                    year,
                    st.w[fi, ri] / base.w[fi, ri] - 1.0,
                )
            )
            named = {"LAB": "wage_change", "CAP": "capital_return_change"}.get(factor)
            if named is not None:
                records.append(
                    _rec_r(named, factor, region, year, st.w[fi, ri] / base.w[fi, ri] - 1.0)
                )
            emp = float(st.F[fi, ri, :].sum())
            emp_base = float(base.F[fi, ri, :].sum())
            var = "employment_change" if factor == "LAB" else f"factor_use_change_{factor}"
            records.append(_rec_r(var, factor, region, year, emp / emp_base - 1.0))
        # Real household consumption per region, as a **base-price (Laspeyres) quantity index**:
        # benchmark prices are 1, so Σ FD valued at benchmark prices = Σ FD. Note the reason pq·FD
        # would be unsuitable here is NOT that other regions' prices are somehow "unpinned" — in
        # this connected system the single global numéraire (region 0's CPI = 1) fixes the common
        # nominal scale for every region's prices, including pq[r] for r≠0, which are fully
        # determined at the solved equilibrium (review P2: an earlier comment claimed otherwise).
        # The actual reason is that pq[r]·FD[r] is CURRENT-PRICE nominal expenditure — it moves
        # with both the quantity change AND the composite price change, so summing it conflates
        # the two rather than isolating the real (quantity) effect. Valuing FD at BASE prices (all
        # 1 at benchmark) strips out the price move and gives a genuine real quantity index. NB
        # this is disposable **consumption**, NOT production-side real GDP: a true real-GDP series
        # needs a base-price value-added measure with trade + tax accounting (review P1). Renamed
        # accordingly so it is not read as GDP.
        cons = float(np.dot(np.ones(cal.ns), st.FD[ri]))
        cons_base = float(np.dot(np.ones(cal.ns), base.FD[ri]))
        records.append(
            _rec_r("real_consumption_change", "__economy__", region, year, cons / cons_base - 1.0)
        )
        # Real GDP per region (review P2, 2026-07-27): the expenditure-side aggregate at BASE prices
        # (all 1 at benchmark), so it is a genuine real quantity index like the closed/open
        # gdp_change_real — consumption + government + investment + net exports (exports − imports),
        # all base-priced. This is production-side real GDP's expenditure counterpart, distinct from
        # real_consumption_change (household consumption alone). GD/INV are zero without the
        # accounts, so the identity degrades gracefully.
        gd = st.GD[ri] if cal.has_government else np.zeros(cal.ns)
        gd_b = base.GD[ri] if cal.has_government else np.zeros(cal.ns)
        idv = st.ID[ri] if cal.has_investment else np.zeros(cal.ns)
        idv_b = base.ID[ri] if cal.has_investment else np.zeros(cal.ns)
        # VOLUME GDP: expenditure aggregate C+G+I+(X−M) at BENCHMARK prices (all 1), a Laspeyres
        # quantity index. Net exports at benchmark prices: Σ EX[ri] − Σ M[ri].
        net_exports_vol = float(st.EX[ri].sum() - st.M[ri].sum())
        gdp_r = cons + float(gd.sum()) + float(idv.sum()) + net_exports_vol
        gdp_r_base = (
            cons_base
            + float(gd_b.sum())
            + float(idv_b.sum())
            + float(base.EX[ri].sum() - base.M[ri].sum())
        )
        vol_ratio = gdp_r / gdp_r_base
        records.append(_rec_r("gdp_change_real", "__economy__", region, year, vol_ratio - 1.0))
        # CURRENT-PRICE GDP: the SAME basket at the run's prices — absorption at composite prices
        # pq[ri], and net exports at CURRENT BILATERAL ROUTE prices pe for BOTH exports and imports
        # (review P1 round 14 — the earlier version priced net exports at benchmark, understating
        # the move and the deflator). Exports from ri: pe[ri,s,d]·EX[ri,s,d]; imports of ri from o:
        # pe[o,s,ri]·M[ri,s,o] (the o→ri route price). Deflator = current-price GDP / volume GDP.
        exports_cur = float((st.pe[ri, :, :] * st.EX[ri, :, :]).sum())
        imports_cur = float((st.pe[:, :, ri].transpose(1, 0) * st.M[ri, :, :]).sum())
        absorb_cur = float(np.dot(st.pq[ri], st.FD[ri] + gd + idv))
        gdp_cur = absorb_cur + exports_cur - imports_cur
        cur_ratio = gdp_cur / gdp_r_base if gdp_r_base != 0 else 1.0
        records.append(_rec_r("gdp_change_nominal", "__economy__", region, year, cur_ratio - 1.0))
        # deflator = (1+nominal) / (1+volume) − 1: the GDP basket's price move vs the CPI numéraire.
        records.append(
            _rec_r("gdp_deflator_change", "__economy__", region, year, cur_ratio / vol_ratio - 1.0)
        )
        u = float(np.prod(np.power(st.FD[ri], cal.gamma[ri])))
        u_base = float(np.prod(np.power(base.FD[ri], cal.gamma[ri])))
        records.append(_rec_r("welfare_change", "__economy__", region, year, u / u_base - 1.0))
        # carbon_revenue is a SHARE OF THIS REGION'S OWN benchmark GDP (review P2: dividing by
        # cal.gdp0 — GLOBAL benchmark GDP — understates the share for any region smaller than the
        # whole economy; e.g. on the toy fixture North has 53.16% of global GDP, so a "5.00%"
        # figure divided by global GDP is actually 9.41% of North's own GDP). Regional benchmark
        # GDP = that region's factor income at benchmark, cal.F0[:, ri, :].sum() (F0 is [f,r,s]).
        regional_gdp0 = float(cal.F0[:, ri, :].sum())
        records.append(
            _rec_r(
                "carbon_revenue", "__economy__", region, year, st.carbon_revenue[ri] / regional_gdp0
            )
        )
        # Covered physical emissions per region (review P1 round 13): change in Σ intensity·Z[r],
        # weighted by the PRICE-FREE per-(region,sector) intensity, so it is well-defined every year
        # (including zero-price). Falls back to cc[r] if no separate intensity was passed.
        inten_r = None
        if intensity is not None:
            inten_r = intensity[ri]
        elif cc is not None:
            inten_r = cc[ri]
        _emit_covered_emissions(records, inten_r, base.Z[ri], st.Z[ri], year, region=region)
        # Energy-sector output volume per region (renamed from energy_use_change — review P2 round
        # 13): total real output of the region's energy sectors, only with that region's nest.
        if cal.has_energy_nest:
            eidx = cal.energy_nests[ri].energy_idx
            eu, eu_base = float(st.Z[ri, eidx].sum()), float(base.Z[ri, eidx].sum())
            records.append(
                _rec_r(
                    "energy_sector_output_volume_change",
                    "__economy__",
                    region,
                    year,
                    eu / eu_base - 1.0,
                )
            )
        # Government accounts (Phase 5d.1): per-region fiscal balance (≡0 under balanced_budget)
        # and government spending, as shares of the region's OWN benchmark GDP (consistent with
        # carbon_revenue above). Emitted only when governments exist, so no-government output
        # stays byte-identical to pre-5d.1.
        if cal.has_government:
            records.append(
                _rec_r(
                    "fiscal_balance",
                    "__economy__",
                    region,
                    year,
                    st.fiscal_balance[ri] / regional_gdp0,
                )
            )
            gov_spend = float(np.dot(st.pq[ri], st.GD[ri]))
            records.append(
                _rec_r("gov_spending", "__economy__", region, year, gov_spend / regional_gdp0)
            )
        # Savings-investment accounts (Phase 5d.2): per-region nominal investment and household
        # savings, shares of the region's OWN benchmark GDP. investment_r = savings_r + Sf_r under
        # savings_driven — they differ by exactly the region's capital-account inflow.
        if cal.has_investment:
            inv_spend = float(np.dot(st.pq[ri], st.ID[ri]))
            records.append(
                _rec_r("investment", "__economy__", region, year, inv_spend / regional_gdp0)
            )
            records.append(
                _rec_r("savings", "__economy__", region, year, st.savings[ri] / regional_gdp0)
            )


def _multi_carbon_share(data: dict, regions: list[str], sectors: list[str]):
    """Per-(region, sector) dimensionless carbon cost share from ``data['carbon_cost_share']``.

    Accepts a nested dict ``{region: {sector: value}}`` or a full ``[nr, ns]`` array. Missing
    entries default to 0. Values must be finite and non-negative (a negative share is an
    undocumented subsidy)."""
    ei = data.get("carbon_cost_share")
    if ei is None:
        return None
    nr, ns = len(regions), len(sectors)
    arr = np.zeros((nr, ns))
    if isinstance(ei, dict):
        unknown_r = [r for r in ei if r not in regions]
        if unknown_r:
            raise ValueError(f"carbon_cost_share regions not in the SAM {regions}: {unknown_r}")
        for ri, r in enumerate(regions):
            row = ei.get(r, {})
            unknown_s = [s for s in row if s not in sectors]
            if unknown_s:
                raise ValueError(f"carbon_cost_share sectors not in the SAM {sectors}: {unknown_s}")
            for si, s in enumerate(sectors):
                arr[ri, si] = float(row.get(s, 0.0))
    else:
        arr = np.asarray(ei, dtype=float)
        if arr.shape != (nr, ns):
            raise ValueError(f"carbon_cost_share array must be shape ({nr}, {ns}); got {arr.shape}")
    if not np.isfinite(arr).all() or float(arr.min()) < 0.0:
        raise ValueError("carbon_cost_share values must be finite and non-negative")
    return arr


@dataclass(frozen=True)
class SweepResult:
    """A provenance-complete Armington elasticity sweep (review P2: the bare DataFrame discarded
    everything needed to identify the sweep once exported). ``bands`` is the tidy envelope
    DataFrame (``sector, variable, low, central, high``); the rest pins the exact inputs so an
    exported sweep is reproducible and cannot be confused with any other ordered triple."""

    bands: object  # pandas.DataFrame — the low/central/high envelope
    elasticities: tuple[float, float, float]  # the exact (low, central, high) values swept
    swept_parameter: str  # which parameter was varied (here 'armington_elast')
    year: int
    engine_version: str
    scenario_hash: str  # hash of the shocks + year (shared across bands)
    sam_identity: dict  # canonical SAM fingerprint (shared across bands)
    manifests: dict  # {'low'|'central'|'high': the full RunManifest of that band's run}


def armington_sensitivity_sweep(
    data: dict,
    shocks: list[Shock],
    year: int = 2020,
    *,
    elasticities: tuple[float, float, float] = (1.5, 2.0, 4.0),
) -> SweepResult:
    """Run the open-economy CGE across low/central/high **Armington** elasticities and return a
    provenance-complete :class:`SweepResult` (Phase 5.3 sensitivity sweep). Volume responses are
    elasticity-sensitive, so — as with Engine 2's demand bands — the band is a first-class output,
    and (review P2) it carries the exact elasticity values, the engine version, the scenario hash,
    the SAM identity, and each band's full manifest, so nothing is lost on export. ``.bands`` is the
    tidy envelope DataFrame (``sector, variable, low, central, high``). Requires an open SAM."""
    import pandas as pd

    sam = data.get("SAM")
    if sam is None or "ROW" not in sam.accounts:
        raise ValueError("armington_sensitivity_sweep needs an open SAM (with a ROW account)")

    lo, ce, hi = (float(e) for e in elasticities)
    if not all(np.isfinite([lo, ce, hi])) or lo <= 0:
        raise ValueError(f"sweep elasticities must be finite and positive; got {elasticities}.")
    if not (lo < ce < hi):
        raise ValueError(
            f"sweep elasticities must be strictly ordered low < central < high; got "
            f"(low={lo}, central={ce}, high={hi}). An unordered band would mislabel the envelope."
        )
    bands = {"low": lo, "central": ce, "high": hi}
    per_band: dict[str, pd.Series] = {}
    manifests: dict = {}
    for band, elast in bands.items():
        res = CGEStaticEngine().run(
            data={**data, "armington_elast": elast}, shocks=shocks, years=[year]
        )
        manifests[band] = res.manifest
        d = res.data
        d = d[d["variable"].isin(("volume_change", "import_change", "export_change"))]
        per_band[band] = d.set_index(["sector", "variable"])["value"]

    out = pd.DataFrame(per_band).reset_index().sort_values(["variable", "sector"])
    out = out.reset_index(drop=True)
    central = manifests["central"]
    return SweepResult(
        bands=out,
        elasticities=(lo, ce, hi),
        swept_parameter="armington_elast",
        year=year,
        engine_version=CGEStaticEngine.meta.version,
        scenario_hash=central.scenario_hash,
        sam_identity=_sam_fingerprint(sam),
        manifests=manifests,
    )


registry.register(CGEStaticEngine())
