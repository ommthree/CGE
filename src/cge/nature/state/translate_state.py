"""Translate a physical-state **pathway** into the Phase-6.4 ``NatureStress`` vocabulary (Phase
6b.3).

This is the seam between Phase 6b (physical ecosystem-service state) and Phase 6 (ENCORE exposure).
A scenario specifies a physical degradation/restoration pathway for a channel (Phase 6b.2); here we
apply the channel's documented state→severity response (Phase 6b.1/6b.4) per year and emit a
``NatureStress`` — with a per-year severity **path** — for each ENCORE service the channel drives.
Everything downstream (``build_nature_shocks`` → exposure → engine) is unchanged: 6b produces the
``severity`` that a scenario author previously had to assert by hand.

One channel may drive several ENCORE services (e.g. water availability → *Water supply* + *Water
flow
regulation* + *Water purification*). Each becomes its own ``NatureStress`` carrying the SAME derived
severity path, so the exposure layer applies each service to the goods that depend on it. Coverage
(``coverage_sectors`` / ``coverage_regions``) is passed through so a scenario can restrict a
physical
pathway to, say, one agricultural region.
"""

from __future__ import annotations

from cge.contracts.shocks import NatureStress
from cge.nature.state.channels import ServiceStateChannel
from cge.nature.state.pathways import StatePathway


def state_severity_path(
    channel: ServiceStateChannel,
    pathway: StatePathway,
    years: list[int],
) -> dict[int, float]:
    """The per-year service-degradation **severity** (in [0, 1]) implied by running ``pathway``
    through ``channel``'s state→severity response over ``years`` (Phase 6b.3).

    The effective state path (recovery hysteresis applied, Phase 6b.4) is mapped through the
    channel's ``severity`` response. Years where the state is at/above baseline give severity 0."""
    state = pathway.state_path(years)
    return {y: channel.severity(state[y]) for y in sorted(state)}


def state_to_nature_stresses(
    channel: ServiceStateChannel,
    pathway: StatePathway,
    years: list[int],
    *,
    coverage_sectors: list[str] | None = None,
    coverage_regions: list[str] | None = None,
) -> list[NatureStress]:
    """Emit one ``NatureStress`` per ENCORE service ``channel`` drives, each carrying the derived
    per-year severity path (Phase 6b.3). Returns ``[]`` if the pathway never degrades the service
    (all severities below a tiny epsilon) — a byte-identical no-op, matching Phase 6's convention.

    ``pathway.channel_id`` must match ``channel.channel_id`` (a mismatched pair is a scenario
    error).
    """
    if pathway.channel_id != channel.channel_id:
        raise ValueError(
            f"pathway is for channel {pathway.channel_id!r} but the channel is "
            f"{channel.channel_id!r}; pass the matching pair."
        )
    # The pathway's baseline (the index the physical trajectory departs from) must AGREE with the
    # channel's response baseline (the index at which severity = 0) — otherwise a pathway anchored
    # at
    # 200 fed through a channel whose response baseline is 100 silently starts at 50% shortfall, a
    # nonsensical severity at the reference condition (review P2 2026-08-23). Both default to 100,
    # so
    # the common case is unaffected; a deliberate custom baseline must be set consistently on BOTH.
    if abs(pathway.baseline - channel.response.baseline) > 1e-9:
        raise ValueError(
            f"pathway baseline {pathway.baseline} disagrees with channel {channel.channel_id!r} "
            f"response baseline {channel.response.baseline}: the physical trajectory's reference "
            "level and the state→severity response's zero-severity level must be the same index "
            "(both default to 100). Set them consistently."
        )
    sev_path = state_severity_path(channel, pathway, years)
    if not sev_path or max(sev_path.values()) <= 1e-12:
        return []  # never degrades → no shock (identical to an unstressed run)

    # Resource-specific sector restriction (review P1, 2026-08-23): an EXPLICIT scenario
    # coverage_sectors always wins; otherwise fall back to the channel's default_sectors, so a
    # forestry- or fisheries-stock shock is confined to the forestry/fishing sectors rather than
    # leaking through the shared "Biomass provisioning" service into every dependent sector (crops,
    # the other resource, …). An economy-wide channel (empty default_sectors) is unchanged.
    effective_sectors = (
        list(coverage_sectors) if coverage_sectors else list(channel.default_sectors)
    )

    # The scalar `severity` is the peak of the path (NatureStress requires a scalar; the path drives
    # the per-year values). Round-trip-safe: a flat path reduces to that scalar.
    peak = max(sev_path.values())
    stresses: list[NatureStress] = []
    for service in channel.services:
        stresses.append(
            NatureStress(
                service=service,
                severity=peak,
                path=dict(sev_path),
                coverage_sectors=effective_sectors,
                coverage_regions=list(coverage_regions or []),
            )
        )
    return stresses


def build_state_scenario(
    items: list[tuple[ServiceStateChannel, StatePathway]],
    years: list[int],
    *,
    coverage_sectors: list[str] | None = None,
    coverage_regions: list[str] | None = None,
) -> list[NatureStress]:
    """Compose several (channel, pathway) pairs into the full ``NatureStress`` list a scenario feeds
    to ``build_nature_shocks`` (Phase 6b.3). Coverage is applied to every emitted stress.

    Two pathways driving the SAME ENCORE service would each emit a stress on it; the Phase-6
    translate
    layer already rejects two stresses that both reach the same good on one service, so a scenario
    combining overlapping physical channels is caught there with a clear message."""
    out: list[NatureStress] = []
    for channel, pathway in items:
        out.extend(
            state_to_nature_stresses(
                channel,
                pathway,
                years,
                coverage_sectors=coverage_sectors,
                coverage_regions=coverage_regions,
            )
        )
    return out
