"""
Illustrative what-if price-path preview for the Action screen.

CLAUDE.md's what-if simulator asks for sliders on quantity, state/centre and
timing, "show[ing] the counterfactual price path bending against the
do-nothing baseline." That is the one Phase 4 piece that cannot be built
from a real fitted model: no source this project ingests contains a
historical log of past buffer-stock releases, so there is nothing to fit a
price-elasticity-of-release against.

This is therefore an explicit, labelled **assumption**, not a calibrated
estimate -- same convention as decide/reference_data.py's capacity and
transport-cost estimates. It exists to show the *shape* of the intended
effect (releasing more, or releasing into a thinner market, bends the price
down more; the effect ramps in after the dispatch date rather than landing
instantly) so an officer can reason about timing and quantity. It is not a
rupee forecast and must not be read as one.

    impact_pct(t) = MAX_IMPACT_PCT * min(release_tonnes / capacity_tonnes, 1)
                    * ramp(t - release_day)

    ramp(x) = clip(x / RAMP_DAYS, 0, 1)   -- 0 before dispatch, full strength
                                              RAMP_DAYS after it lands

    price_with_release(t) = price_baseline(t) * (1 - impact_pct(t) / 100)

The do-nothing baseline itself is not illustrative -- it is a straight-line
interpolation through the three real forecast points this system already
produces (today's price, the 7-day P50, the 14-day P50).
"""

from __future__ import annotations

from dataclasses import dataclass

# Assumed maximum price effect of a release equal to 100% of a centre's
# absorption capacity, fully ramped in. Order-of-magnitude placeholder only
# (real PSF releases are reported to shave low-single- to low-double-digit
# percentages off a spiking price, depending on market depth) -- not fitted
# to any data in this project.
MAX_IMPACT_PCT = 6.0

# Days for the effect to ramp from 0 to full strength after the release date.
RAMP_DAYS = 5

HORIZON_DAYS = 14


@dataclass
class PricePath:
    days: list[int]
    baseline: list[float]
    with_release: list[float]


def _ramp(days_since_release: float) -> float:
    if days_since_release <= 0:
        return 0.0
    return min(days_since_release / RAMP_DAYS, 1.0)


def price_path(
    *,
    current_price: float,
    p50_7d: float,
    p50_14d: float,
    release_tonnes: float,
    capacity_tonnes: float,
    release_day: int,
) -> PricePath:
    """Do-nothing baseline (piecewise-linear through the real day-0/7/14
    forecast points) plus an illustrative with-release path bent down by the
    impact model above. `capacity_tonnes` <= 0 degrades to a flat (no
    absorption headroom) with-release path rather than dividing by zero."""
    days = list(range(0, HORIZON_DAYS + 1))

    def baseline_at(t: int) -> float:
        if t <= 7:
            return current_price + (p50_7d - current_price) * (t / 7)
        return p50_7d + (p50_14d - p50_7d) * ((t - 7) / 7)

    fill_fraction = (
        min(release_tonnes / capacity_tonnes, 1.0) if capacity_tonnes > 0 else 0.0
    )

    baseline = [baseline_at(t) for t in days]
    with_release = [
        b * (1 - MAX_IMPACT_PCT * fill_fraction * _ramp(t - release_day) / 100)
        for t, b in zip(days, baseline)
    ]
    return PricePath(days=days, baseline=baseline, with_release=with_release)
