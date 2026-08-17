"""
Buffer stock release optimizer (CLAUDE.md Phase 4).

A small linear program in PuLP. Decides how much of a limited buffer stock to
release into each centre, favouring the centres under most price stress while
keeping transport cost down.

    maximise   sum_i [ (stress_i - lambda * cost_i / cost_max) * release_i ]

    subject to sum_i release_i <= available_stock
               0 <= release_i  <= absorption_capacity_i

Each tonne sent to centre i is worth that centre's stress score, discounted by
how far it has to travel. `lambda` is in stress-points: it is how much stress
relief you would give up to avoid the longest haul in the candidate set. The
capacity cap is what stops the whole allocation landing in one market, so the
LP fills the most stressed centres first, up to what each can absorb.

An earlier version scored relief as `stress_i * (release_i / capacity_i)` --
"fraction of this centre's capacity filled". That looked reasonable and was
wrong: it makes the value of a tonne inversely proportional to the receiving
state's capacity, so small states dominate and the genuinely most-stressed
large-state centres are starved. It allocated nothing at all to the two
highest-stress centres in the live data. Per-tonne stress weighting has no
such pathology.

**Honesty note.** The objective is real (stress comes from fitted models), but
two constraint inputs are labelled estimates, not sourced figures: absorption
capacity is population-derived and transport cost is distance-derived. See
decide/reference_data.py. The UI shows those labels alongside the output; this
table is a prioritisation aid, not a procurement order.
"""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from reference_data import (
    LABELS,
    absorption_capacity_tonnes,
    distance_from_depot_km,
    transport_cost_per_tonne,
)

# Weight on the transport term, in **stress points**. Both terms are
# normalised to the same scale before this is applied (stress relief 0-100
# per saturated centre; transport as a 0-1 fraction of the furthest centre's
# cost, also scaled to 0-100), so lambda reads directly as: "how many stress
# points is the full haul to the most distant centre worth giving up?"
#
# Getting this scaling right mattered. With raw rupee costs the transport
# term (0-6,000) swamped a benefit term of 0.02-0.08 per tonne by five orders
# of magnitude, and the LP degenerated to shipping only into the depot city
# itself. Normalising both sides makes the trade-off explicit and tunable.
DEFAULT_TRANSPORT_WEIGHT = 12.0

# Below this stress score a centre is not a release candidate at all; shipping
# buffer stock into a calm market displaces normal trade for no benefit.
MIN_STRESS_TO_QUALIFY = 30.0


@dataclass
class ReleasePlan:
    rows: list[dict]
    status: str
    total_released: float
    total_cost: float
    total_relief: float
    available_stock: float
    n_candidates: int


def build_candidates(stress_rows: list[dict]) -> list[dict]:
    """stress rows (commodity, centre, state, stress, ...) -> candidate
    centres with their capacity and transport cost attached.

    Absorption capacity is a *state* quantity, so when a state has more than
    one benchmark centre (Maharashtra has Mumbai and Nagpur) it is split
    between them. Giving each centre the full state figure would let the LP
    allocate that state's capacity twice over.
    """
    qualifying = [
        r
        for r in stress_rows
        if r.get("stress") is not None
        and r["stress"] >= MIN_STRESS_TO_QUALIFY
        and transport_cost_per_tonne(r["centre"]) is not None
    ]

    centres_per_state: dict[str | None, int] = {}
    for r in qualifying:
        centres_per_state[r.get("state")] = centres_per_state.get(r.get("state"), 0) + 1

    candidates = []
    for r in qualifying:
        share = centres_per_state[r.get("state")]
        candidates.append(
            {
                **r,
                "capacity_tonnes": absorption_capacity_tonnes(r.get("state")) / share,
                "centres_sharing_state_capacity": share,
                "transport_cost_per_tonne": transport_cost_per_tonne(r["centre"]),
                "distance_km": distance_from_depot_km(r["centre"]),
            }
        )
    return candidates


def optimise_release(
    stress_rows: list[dict],
    available_stock: float,
    transport_weight: float = DEFAULT_TRANSPORT_WEIGHT,
) -> ReleasePlan:
    """Solves the LP and returns a signable-shaped table.

    `stress_rows` should already be filtered to one commodity -- buffer stock
    is commodity-specific (you cannot release tur into an onion shortage), so
    mixing commodities in one stock pool would be wrong.
    """
    candidates = build_candidates(stress_rows)
    if not candidates or available_stock <= 0:
        return ReleasePlan([], "No candidates", 0.0, 0.0, 0.0, available_stock, 0)

    problem = pulp.LpProblem("buffer_stock_release", pulp.LpMaximize)
    release = {
        c["centre"]: pulp.LpVariable(
            f"release_{c['centre']}", lowBound=0, upBound=c["capacity_tonnes"]
        )
        for c in candidates
    }

    # Each tonne is worth its centre's stress, discounted by distance. Both
    # sides are in stress-points, so lambda is directly interpretable.
    max_cost = max(c["transport_cost_per_tonne"] for c in candidates) or 1.0
    for c in candidates:
        c["score"] = c["stress"] - transport_weight * (
            c["transport_cost_per_tonne"] / max_cost
        )

    problem += pulp.lpSum(c["score"] * release[c["centre"]] for c in candidates)
    problem += pulp.lpSum(release.values()) <= available_stock, "stock_limit"

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[problem.status]

    rows = []
    for c in candidates:
        qty = release[c["centre"]].value() or 0.0
        if qty < 1.0:  # ignore rounding dust
            continue
        rows.append(
            {
                "centre": c["centre"],
                "state": c.get("state"),
                "commodity": c["commodity"],
                "stress": c["stress"],
                "priority_score": round(c["score"], 1),
                "release_tonnes": round(qty, 1),
                "capacity_tonnes": round(c["capacity_tonnes"], 1),
                "pct_of_capacity": round(100 * qty / c["capacity_tonnes"], 1),
                "distance_km": round(c["distance_km"], 0),
                "transport_cost_inr": round(qty * c["transport_cost_per_tonne"], 0),
                # Stress-tonnes delivered: the objective's own units, so the
                # column and the solved objective agree.
                "stress_tonnes": round(c["stress"] * qty, 0),
            }
        )
    rows.sort(key=lambda r: r["release_tonnes"], reverse=True)

    return ReleasePlan(
        rows=rows,
        status=status,
        total_released=round(sum(r["release_tonnes"] for r in rows), 1),
        total_cost=round(sum(r["transport_cost_inr"] for r in rows), 0),
        total_relief=round(sum(r["stress_tonnes"] for r in rows), 0),
        available_stock=available_stock,
        n_candidates=len(candidates),
    )


INPUT_LABELS = LABELS
