"""Shared physical constants used across layers (data adapters and engines).

Kept here — not in the data layer — so engines can use them without importing the data
layer (respecting the contract boundary in ADR-0002).
"""

# Global Warming Potentials, 100-year, IPCC AR5. Used to combine gases into CO2e and to
# weight a scenario's selected gases. Source: IPCC AR5 (2013), Table 8.7.
GWP100_AR5 = {
    "CO2": 1.0,
    "CH4": 28.0,
    "N2O": 265.0,
}
