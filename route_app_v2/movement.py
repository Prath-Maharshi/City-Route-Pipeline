"""
movement.py — Movement-aware (turn-restricted) routing.

Identical algorithm to movement_router.py but:
  • Topology fallback uses adj_from_node dict (plain dict lookup, no NetworkX).
  • No networkx import at all — pure Python Dijkstra + movements.pkl.
"""
from __future__ import annotations

import heapq
import logging
import pickle
from pathlib import Path
from typing import Optional

log = logging.getLogger("movement")

ROAD_PENALTY = {
    "motorway": 0.6, "trunk": 0.7, "trunk_link": 0.7,
    "primary": 0.85, "secondary": 1.0, "secondary_link": 1.0,
    "tertiary": 1.2, "tertiary_link": 1.2,
    "unclassified": 1.4, "residential": 1.6,
}


class MovementRouter:
    def __init__(self, movements_pkl: str, adj_from_node: dict, fallback_penalty_s: float = 5.0):
        self.movements_pkl      = movements_pkl
        self.adj_from_node      = adj_from_node   # node_str → [eid, ...] dict from AppState
        self.fallback_penalty_s = fallback_penalty_s
        self._movements:   dict = {}
        self._successors:  dict = {}
        self._avg_turn_by_hour: dict = {}
        self._loaded = False

    def load(self) -> None:
        path = Path(self.movements_pkl)
        if not path.exists():
            log.warning("movements.pkl not found: %s", path)
            self._loaded = True
            return

        log.info("Loading movements from %s ...", path)
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self._movements  = payload.get("movements",       {})
        self._successors = payload.get("edge_successors", {})

        from collections import defaultdict
        T = 24
        incoming: dict = defaultdict(lambda: [[] for _ in range(T)])
        for (_, t), mv in self._movements.items():
            hourly = mv.get("turn_time_by_hour")
            base   = mv.get("base_turn_s", mv.get("turn_time_s", self.fallback_penalty_s))
            for h in range(T):
                cost = hourly[h] if hourly is not None else base
                incoming[t][h].append(cost)
        fp = self.fallback_penalty_s
        self._avg_turn_by_hour = {
            eid: [(sum(v[h]) / len(v[h])) if v[h] else fp for h in range(T)]
            for eid, v in incoming.items()
        }

        self._loaded = True
        log.info("MovementRouter: %d movements, %d edges with avg turn",
                 len(self._movements), len(self._avg_turn_by_hour))

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def has_movements(self) -> bool:
        return bool(self._movements)

    def get_turn_time(self, from_edge: str, to_edge: str, hour: Optional[int] = None) -> float:
        mv = self._movements.get((from_edge, to_edge))
        if mv is None:
            return self.fallback_penalty_s
        if hour is not None:
            hourly = mv.get("turn_time_by_hour")
            if hourly is not None:
                return hourly[min(hour, len(hourly) - 1)]
        return mv.get("base_turn_s", mv.get("turn_time_s", self.fallback_penalty_s))

    def get_turn_info(self, from_edge: str, to_edge: str) -> Optional[dict]:
        mv = self._movements.get((from_edge, to_edge))
        return dict(mv) if mv is not None else None

    def valid_successors(self, from_edge: str) -> list[str]:
        if self._successors:
            return self._successors.get(from_edge, [])
        return []

    def route_via_line_graph(
        self,
        edge_lookup: dict,
        start_edge:  str,
        end_edge:    str,
        hour:        int,
        blocked_edges: set | None = None,
        tt_override:   dict | None = None,
    ) -> Optional[dict]:
        """
        Turn-cost Dijkstra on the line graph.

        Nodes = SUMO edge IDs.  Edge (A → B) = valid turn from road A to road B.
        Weight = travel_time(A) + turn_time(A → B).

        Uses self.adj_from_node (plain dict) for topology fallback instead of
        NetworkX — eliminates networkx overhead entirely.
        """
        if not self._loaded:
            return None

        start = edge_lookup.get(start_edge)
        end   = edge_lookup.get(end_edge)
        if start is None or end is None:
            return None

        if blocked_edges is None:
            blocked_edges = set()

        def _edge_cost(eid: str) -> float:
            d = edge_lookup.get(eid, {})
            if not d:
                return 30.0
            type_m = ROAD_PENALTY.get(d.get("road_type", "unclassified"), 1.5)
            conf_m = 1.0 + (1.0 - d.get("confidence", 0.1)) * 0.2
            if tt_override and eid in tt_override:
                return tt_override[eid] * type_m * conf_m
            tt_arr = d.get("tt", [30.0] * 24)
            base_t = tt_arr[hour] if hour < len(tt_arr) else tt_arr[-1]
            return base_t * type_m * conf_m

        dist: dict[str, float]          = {}
        prev: dict[str, Optional[str]]  = {}
        heap = [(0.0, start_edge)]
        dist[start_edge] = 0.0
        prev[start_edge] = None

        while heap:
            d, cur = heapq.heappop(heap)
            if d > dist.get(cur, float("inf")):
                continue
            if cur == end_edge:
                break

            seen: set[str] = set()

            # 1. Movement-map successors
            if self._successors:
                for succ in self._successors.get(cur, []):
                    if succ in blocked_edges:
                        continue
                    seen.add(succ)
                    turn_t = self.get_turn_time(cur, succ, hour)
                    nd = d + _edge_cost(succ) + turn_t
                    if nd < dist.get(succ, float("inf")):
                        dist[succ] = nd
                        prev[succ] = cur
                        heapq.heappush(heap, (nd, succ))

            # 2. Topology fallback (plain dict, no NetworkX)
            cur_data = edge_lookup.get(cur, {})
            cur_v    = cur_data.get("to_node") or cur_data.get("v")
            if cur_v:
                for succ in self.adj_from_node.get(cur_v, []):
                    if succ in blocked_edges or succ in seen:
                        continue
                    nd = d + _edge_cost(succ) + self.fallback_penalty_s
                    if nd < dist.get(succ, float("inf")):
                        dist[succ] = nd
                        prev[succ] = cur
                        heapq.heappush(heap, (nd, succ))

        if end_edge not in dist:
            return None

        # Reconstruct path
        path: list[str] = []
        cur = end_edge
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()

        # Build detailed metrics
        total_tt_base   = 0.0
        total_turn_time = 0.0
        path_detail     = []

        for idx, eid in enumerate(path):
            d_edge = edge_lookup.get(eid, {})
            tt_arr = d_edge.get("tt", [30.0] * 24)
            base_t = (tt_override[eid]
                      if tt_override and eid in tt_override
                      else (tt_arr[hour] if hour < len(tt_arr) else tt_arr[-1]))
            low_t  = d_edge.get("tt_low",  [base_t] * 24)
            high_t = d_edge.get("tt_high", [base_t] * 24)
            low    = low_t[hour]  if hour < len(low_t)  else low_t[-1]
            high   = high_t[hour] if hour < len(high_t) else high_t[-1]
            low    = max(low,  base_t) if tt_override and eid in tt_override else low
            high   = max(high, base_t) if tt_override and eid in tt_override else high

            turn_t = via_len = 0.0
            turn_info = None
            if idx > 0:
                prev_eid = path[idx - 1]
                mv = self._movements.get((prev_eid, eid))
                if mv:
                    turn_t    = self.get_turn_time(prev_eid, eid, hour)
                    turn_info = dict(mv)
                    via_len   = mv.get("via_length_m", 15.0)
                else:
                    turn_t  = self.fallback_penalty_s
                    via_len = 15.0

            total_tt_base   += base_t
            total_turn_time += turn_t
            path_detail.append({
                "edge_id":    eid,
                "tt_base_s":  round(base_t, 2),
                "tt_low_s":   round(low,    2),
                "tt_high_s":  round(high,   2),
                "turn_s":     round(turn_t, 2),
                "via_length_m": round(via_len, 1),
                "turn_info":  turn_info,
                "length_m":   d_edge.get("length", 0.0),
                "road_type":  d_edge.get("road_type", ""),
                "dom_dir":    d_edge.get("dom_dir", "unknown"),
                "confidence": d_edge.get("confidence", 0.1),
            })

        total_low  = sum(s["tt_low_s"]  for s in path_detail) + total_turn_time
        total_high = sum(s["tt_high_s"] for s in path_detail) + total_turn_time
        total_exp  = sum(
            _adjusted_expected(s["tt_base_s"], s["tt_low_s"], s["tt_high_s"], s["dom_dir"])
            for s in path_detail
        ) + total_turn_time
        via_dist = sum(s["via_length_m"] for s in path_detail)

        return {
            "edges":             path,
            "total_time_s":      round(total_tt_base + total_turn_time, 2),
            "total_time_low_s":  round(total_low,  2),
            "total_time_high_s": round(total_high, 2),
            "expected_time_s":   round(total_exp,  2),
            "turn_time_s":       round(total_turn_time, 2),
            "n_turns":           len(path) - 1,
            "dist_m":            sum(s["length_m"] for s in path_detail) + via_dist,
            "path_detail":       path_detail,
        }


def _adjusted_expected(base_t, low_t, high_t, dom_dir):
    if dom_dir == "underestimate":
        adj = base_t + 0.75 * (high_t - base_t)
    elif dom_dir == "overestimate":
        adj = base_t - 0.75 * (base_t - low_t)
    else:
        adj = base_t
    return max(low_t, min(high_t, adj))
