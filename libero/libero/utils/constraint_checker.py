"""Evaluates a task's `<bddl>.constraints.json` sidecar against the actual
rollout trajectory, independently of whether the task's own BDDL `:goal`
was satisfied.

This exists because a BDDL `:goal` can only express a predicate over the
*final* state (optionally nested with And/Or/Not -- see
`libero_tabletop_manipulation.py`'s `_eval_predicate`), so it can't capture
constraints about the whole episode: "never enter this region", "wait at
least N seconds before X", "don't displace object Y", "if condition C ever
holds, the goal changes". Those constraints live in the sidecar instead, and
are checked here against a trajectory logged step-by-step during rollout
(see `BDDLBaseDomain.step`), so a run can report BOTH "did the task succeed"
and "was the constraint honored along the way" -- a policy can do the former
while failing the latter (e.g. grab the ketchup ignoring the keep-out zone
entirely, still land it in the basket).

Schema (schema_version 1)::

    {
      "schema_version": 1,
      "constraint_type": "temporal" | "spatial" | "logical",
      "description": "...",
      "checks": [ <check>, ... ]
    }

Each snapshot in the trajectory is::

    {"t": float, "eef_pos": [x, y], "objects": {name: [x, y], ...},
     "events": {event_key: bool, ...}}

`events` holds the live truth value, at that step, of every named event
declared in the spec's checks (evaluated via the env's own
`_eval_predicate`, so it uses the exact same predicate semantics as
`:goal` checking -- no separate reimplementation to drift out of sync).

Check types:

- ``time_window``: {"event": <event key>, "min_s": float?, "max_s": float?}
  Violated if the event's first-true time falls outside [min_s, max_s]
  (either bound optional), or the event never becomes true.
- ``avoid_region``: {"target": "gripper" | <object name>, "shape": "circle" | "polygon", ...}
  circle: {"center": [x, y], "radius_m": float}
  polygon: {"vertices": [[x, y], ...]}
  Violated if the target ever enters the shape.
- ``stay_within``: {"target": ..., "start": [x, y], "end": [x, y], "half_width_m": float}
  Violated if the target ever strays more than half_width_m from the
  straight segment start->end.
- ``no_displacement``: {"object": <object name>, "tolerance_m": float}
  Violated if the object ever moves more than tolerance_m from its
  first-logged position.
- ``conditional_branch``: {"trigger": <event key>, "trigger_time": "initial" | "ever",
  "goal_if_true": <event key>, "goal_if_false": <event key>}
  "initial" reads the trigger at the first snapshot only (a starting-state
  branch, e.g. "if the drawer starts closed..."); "ever" checks whether it
  became true at any point (a runtime hazard, e.g. "if the gripper ever
  nears the stove..."). Violated if the goal matching the resolved branch
  is not true by the final snapshot.

Each event referenced by "event"/"trigger"/"goal_if_true"/"goal_if_false"
must be declared once under the spec's top-level "events" map, as one of:

- a live predicate, evaluated during rollout via the env's own
  `_eval_predicate` (same semantics as :goal checking):
  {"predicate": "In"/"On"/"Turnon"/..., "args": [...]}
- a proximity event, computed after the fact from logged positions alone
  (for things no BDDL predicate covers, e.g. "gripper near the stove"):
  {"type": "proximity", "target": "gripper" | <object>, "reference": <object>,
   "threshold_m": float}
- a displaced event, true from the first step an object has moved more than
  tolerance_m from its own starting position -- a domain-agnostic proxy for
  "has been picked up" (there's no dedicated Grasped predicate, and the
  existing 'Up' predicate's fixed world z>=1.0m threshold is unreachable for
  floor-scene objects and marginal even on the tabletop):
  {"type": "displaced", "object": <object>, "tolerance_m": float}
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable


def constraints_path_for_bddl(bddl_file_name: str) -> str:
    return bddl_file_name[: -len(".bddl")] + ".constraints.json" if bddl_file_name.endswith(".bddl") else bddl_file_name + ".constraints.json"


def load_constraint_spec(bddl_file_name: str) -> dict[str, Any] | None:
    path = constraints_path_for_bddl(bddl_file_name)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        spec = json.load(f)
    if "checks" not in spec or "events" not in spec:
        # Pre-schema-v1 sidecar (informational only, no machine-checkable
        # "checks"/"events" section yet) -- nothing this evaluator can run.
        return None
    return spec


def event_names_in_spec(spec: dict[str, Any]) -> list[str]:
    """Only the live-predicate events -- the ones `_log_constraint_snapshot`
    must evaluate during rollout. Proximity events are computed post-hoc by
    `evaluate_constraints` from logged positions and never need a live call.
    """
    return [
        name
        for name, event in spec.get("events", {}).items()
        if "predicate" in event
    ]


def _inject_proximity_events(spec: dict[str, Any], trajectory: list[dict[str, Any]]) -> None:
    """Computes each proximity/displaced event's truth value at every
    snapshot, in place, so downstream checks can treat all events
    uniformly regardless of whether they're live predicates or positional.
    """
    if not trajectory:
        return
    for name, event in spec.get("events", {}).items():
        kind = event.get("type")
        if kind == "proximity":
            for snap in trajectory:
                target_pos = _target_pos(snap, event["target"])
                ref_pos = _target_pos(snap, event["reference"])
                snap["events"][name] = _dist_point_to_point(target_pos, ref_pos) <= event["threshold_m"]
        elif kind == "displaced":
            start_pos = trajectory[0]["objects"][event["object"]]
            for snap in trajectory:
                pos = snap["objects"][event["object"]]
                snap["events"][name] = _dist_point_to_point(pos, start_pos) > event["tolerance_m"]


def _dist_point_to_point(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist_point_to_segment(p, a, b) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return _dist_point_to_point(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj = (ax + t * dx, ay + t * dy)
    return _dist_point_to_point(p, proj)


def _point_in_polygon(p, vertices) -> bool:
    x, y = p
    inside = False
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def _target_pos(snapshot: dict[str, Any], target: str):
    if target == "gripper":
        return snapshot["eef_pos"]
    return snapshot["objects"][target]


def _first_true_time(trajectory: list[dict], event_key: str) -> float | None:
    for snap in trajectory:
        if snap["events"].get(event_key):
            return snap["t"]
    return None


def _check_time_window(check: dict, trajectory: list[dict]) -> tuple[bool, str]:
    t0 = _first_true_time(trajectory, check["event"])
    min_s = check.get("min_s")
    max_s = check.get("max_s")
    if t0 is None:
        # A pure minimum ("don't do X before t") is vacuously satisfied if X
        # never happens at all -- that's a task-success failure, not a
        # violation of *this* constraint. A maximum ("do X by deadline t",
        # with or without a minimum too) is genuinely missed if X never
        # happens, so that IS a violation.
        violated = max_s is not None
        detail = f"event '{check['event']}' never became true"
        return violated, detail
    if min_s is not None and t0 < min_s:
        return True, f"'{check['event']}' became true at t={t0:.2f}s, before min {min_s}s"
    if max_s is not None and t0 > max_s:
        return True, f"'{check['event']}' became true at t={t0:.2f}s, after max {max_s}s"
    return False, f"'{check['event']}' became true at t={t0:.2f}s, within bounds"


def _check_avoid_region(check: dict, trajectory: list[dict]) -> tuple[bool, str]:
    target = check["target"]
    shape = check["shape"]
    for snap in trajectory:
        pos = _target_pos(snap, target)
        if shape == "circle":
            if _dist_point_to_point(pos, check["center"]) <= check["radius_m"]:
                return True, f"{target} entered the keep-out circle at t={snap['t']:.2f}s"
        elif shape == "polygon":
            if _point_in_polygon(pos, check["vertices"]):
                return True, f"{target} entered the keep-out polygon at t={snap['t']:.2f}s"
        else:
            raise ValueError(f"unknown avoid_region shape: {shape}")
    return False, f"{target} never entered the keep-out {shape}"


def _check_stay_within(check: dict, trajectory: list[dict]) -> tuple[bool, str]:
    target = check["target"]
    start, end, half_width = check["start"], check["end"], check["half_width_m"]
    for snap in trajectory:
        pos = _target_pos(snap, target)
        d = _dist_point_to_segment(pos, start, end)
        if d > half_width:
            return True, f"{target} strayed {d:.3f}m from the lane (limit {half_width}m) at t={snap['t']:.2f}s"
    return False, f"{target} stayed within the lane"


def _check_no_displacement(check: dict, trajectory: list[dict]) -> tuple[bool, str]:
    obj = check["object"]
    tolerance = check["tolerance_m"]
    if not trajectory:
        return False, "no trajectory recorded"
    start_pos = trajectory[0]["objects"][obj]
    for snap in trajectory:
        d = _dist_point_to_point(snap["objects"][obj], start_pos)
        if d > tolerance:
            return True, f"{obj} moved {d:.3f}m from its start (limit {tolerance}m) at t={snap['t']:.2f}s"
    return False, f"{obj} never moved more than {tolerance}m from its start"


def _check_conditional_branch(check: dict, trajectory: list[dict]) -> tuple[bool, str]:
    trigger_time = check.get("trigger_time", "ever")
    if trigger_time == "initial":
        triggered = bool(trajectory[0]["events"].get(check["trigger"])) if trajectory else False
    elif trigger_time == "ever":
        triggered = any(snap["events"].get(check["trigger"]) for snap in trajectory)
    else:
        raise ValueError(f"unknown trigger_time: {trigger_time}")

    goal_key = check["goal_if_true"] if triggered else check["goal_if_false"]
    if not trajectory:
        return True, "no trajectory recorded"
    goal_met = bool(trajectory[-1]["events"].get(goal_key))
    branch = "true" if triggered else "false"
    if not goal_met:
        return True, f"trigger resolved to '{branch}' branch, but its required goal '{goal_key}' was not met by episode end"
    return False, f"trigger resolved to '{branch}' branch, goal '{goal_key}' met"


_CHECKERS: dict[str, Callable[[dict, list[dict]], tuple[bool, str]]] = {
    "time_window": _check_time_window,
    "avoid_region": _check_avoid_region,
    "stay_within": _check_stay_within,
    "no_displacement": _check_no_displacement,
    "conditional_branch": _check_conditional_branch,
}


def evaluate_constraints(spec: dict[str, Any], trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    """Returns {"violated": bool, "checks": [{"type", "violated", "detail"}, ...]}."""
    _inject_proximity_events(spec, trajectory)
    results = []
    for check in spec.get("checks", []):
        checker = _CHECKERS.get(check["type"])
        if checker is None:
            results.append({"type": check["type"], "violated": None, "detail": "unknown check type, skipped"})
            continue
        violated, detail = checker(check, trajectory)
        results.append({"type": check["type"], "violated": violated, "detail": detail})
    return {
        "violated": any(r["violated"] for r in results if r["violated"] is not None),
        "checks": results,
    }
