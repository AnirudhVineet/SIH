"""
Reference tables for the release optimizer.

**Read this before quoting any number from here in a demo.** None of these
are sourced operational figures. Buffer stock levels, state absorption
capacity and NAFED/NCCF transport tariffs are not published in any feed this
project ingests (see QUESTIONS.md). Rather than block the optimizer or
silently fake its inputs, each input here is an explicit, labelled estimate
built from a real published quantity, and the UI carries that label wherever
the number is shown.

  available_stock       -> operator input in the dashboard, not a data source
  absorption capacity   -> "population-based estimate" (Census 2011 state
                           populations, the last full published census)
  transport cost        -> "distance-based estimate" (great-circle km from a
                           single assumed central depot x a flat per-tonne-km
                           rate)

Centre coordinates are genuine geographic values, so the distance term is
real distance -- it is the *rate* and the single-depot assumption that are
the estimate.
"""

from __future__ import annotations

import math

# Assumed single central depot for the transport term. Real NAFED/NCCF
# operations run many depots; one depot is the simplification.
DEPOT = {"name": "Delhi (assumed central depot)", "lat": 28.6139, "lon": 77.2090}

# Flat rate applied to great-circle distance. Indicative road-freight order of
# magnitude for bulk agri commodities; not a tendered rate.
TRANSPORT_RATE_INR_PER_TONNE_KM = 3.5

# Genuine city coordinates for the 10 benchmark centres.
CENTRE_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Nagpur": (21.1458, 79.0882),
    "Lucknow": (26.8467, 80.9462),
    "Kolkata": (22.5726, 88.3639),
    "Bengaluru": (12.9716, 77.5946),
    "Patna": (25.5941, 85.1376),
    "Ahmedabad": (23.0225, 72.5714),
    "Bhopal": (23.2599, 77.4126),
    "Kurnool": (15.8281, 78.0373),
}

# Census 2011 state populations (millions) -- the last full published census.
STATE_POPULATION_MILLIONS = {
    "NCT of Delhi": 16.8,
    "Maharashtra": 112.4,
    "Uttar Pradesh": 199.8,
    "West Bengal": 91.3,
    "Karnataka": 61.1,
    "Bihar": 104.1,
    "Gujarat": 60.4,
    "Madhya Pradesh": 72.6,
    "Andhra Pradesh": 49.4,
}

# Tonnes per million people that a state's mandis could plausibly absorb in a
# single release window without simply displacing normal trade. The scaling
# constant is the assumption; population is the published quantity. Set so
# total national capacity lands in the low hundreds of thousands of tonnes,
# the order of magnitude of real PSF onion/pulse release operations -- which
# makes a stock of a few tens of thousands of tonnes a genuinely binding
# constraint the optimizer has to allocate under, rather than a trivial one.
ABSORPTION_TONNES_PER_MILLION = 300.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def distance_from_depot_km(centre: str) -> float | None:
    coords = CENTRE_COORDS.get(centre)
    if coords is None:
        return None
    return haversine_km(DEPOT["lat"], DEPOT["lon"], coords[0], coords[1])


def transport_cost_per_tonne(centre: str) -> float | None:
    """Distance-based estimate, INR per tonne moved to this centre."""
    km = distance_from_depot_km(centre)
    if km is None:
        return None
    return km * TRANSPORT_RATE_INR_PER_TONNE_KM


def absorption_capacity_tonnes(state: str | None) -> float:
    """Population-based estimate of how much this state can absorb."""
    pop = STATE_POPULATION_MILLIONS.get(state or "")
    if pop is None:
        # Unknown state -> smallest tracked state's capacity, so an unmapped
        # centre can still receive stock but is never favoured by the LP.
        pop = min(STATE_POPULATION_MILLIONS.values())
    return pop * ABSORPTION_TONNES_PER_MILLION


# Labels the UI must show wherever these numbers appear. Kept ASCII-only:
# these strings are also printed into the PDF brief, and the standard PDF
# base-14 fonts have no rupee glyph (it renders as a black box).
LABELS = {
    "available_stock": "operator input - not a sourced figure",
    "capacity": "population-based estimate (Census 2011)",
    "transport": (
        "distance-based estimate (great-circle km x Rs "
        f"{TRANSPORT_RATE_INR_PER_TONNE_KM}/tonne-km)"
    ),
}
