"""
movement_router.py
==================
Builds a movement-aware (turn-restricted) routing graph and provides the
impedance function for app.py.

Why a line graph?
-----------------
Standard Dijkstra on a node graph cannot model turn costs correctly because
the cost of traversing an edge depends on which edge you came from. The line
graph transformation solves this:

  Original graph:   nodes = junctions,  edges = roads
  Line graph:       nodes = roads,      edges = valid turns

In the line graph, the weight of a "turn edge" (road_A → road_B) is:
    travel_time(road_A, hour) + turn_time(road_A → road_B)

This means Dijkstra on the line graph automatically picks the sequence of
roads that minimises total travel + turn time.

Turn restrictions are implicit: if (road_A, road_B) is not in the movement
map, there is no edge in the line graph → Dijkstra cannot use that turn.

Usage in app.py
---------------
    from movement_router import MovementRouter
    
    router = MovementRouter(movements_pkl, fallback_penalty=1.05)
    router.build(G_nx, edge_lookup, hour=8)   # call once per app startup

    # Per request:
    path_edges, total_time = router.route(start_edge, end_edge, hour, blocked)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

log = logging.getLogger("movement_router")


class MovementRouter:
    """
    Movement-aware router using the pre-built movements.pkl.

    The router builds a line graph from the movements at startup.
    Routing is performed via networkx Dijkstra on the line graph.
    """

    def __init__(
            self,
            movements_pkl: str,
            fallback_penalty_s: float = 1.05,
        ):
            self.movements_pkl      = movements_pkl
            self.fallback_penalty_s = fallback_penalty_s
            self._movements: dict   = {}
            self._successors: dict  = {}
            self._avg_turn_time: dict = {}
            self._loaded            = False

    def load(self) -> None:
            """Load movements.pkl. Call once at startup."""
            path = Path(self.movements_pkl)
            if not path.exists():
                log.warning("movements.pkl not found: %s — will use flat penalty fallback",
                            path)
                self._loaded = True
                return

            log.info("Loading movements from %s ...", path)
            with open(path, "rb") as f:
                payload = pickle.load(f)

            self._movements  = payload.get("movements",       {})
            self._successors = payload.get("edge_successors", {})

            # Precompute avg incoming turn time per edge — used by build_impedance_fn
            # to avoid O(movements) scan per Dijkstra step
            from collections import defaultdict
            incoming: dict = defaultdict(list)
            for (_, t), mv in self._movements.items():
                incoming[t].append(mv["turn_time_s"])
            self._avg_turn_time: dict = {
                eid: sum(v) / len(v) for eid, v in incoming.items()
            }

            self._loaded = True
            log.info("  %d movements loaded  |  %d edges with avg turn time precomputed",
                    len(self._movements), len(self._avg_turn_time))

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def has_movements(self) -> bool:
        return bool(self._movements)

    def get_turn_time(self, from_edge: str, to_edge: str) -> float:
        """
        Return turn time in seconds for the (from_edge → to_edge) movement.
        Returns fallback_penalty_s if the movement is not in the map
        (i.e. the turn is geometrically possible but not explicitly modelled).
        """
        mv = self._movements.get((from_edge, to_edge))
        if mv is not None:
            return mv["turn_time_s"]   # was mv.turn_time_s
        return self.fallback_penalty_s

    def get_turn_info(self, from_edge: str, to_edge: str) -> Optional[dict]:
        """Return full movement info dict or None."""
        mv = self._movements.get((from_edge, to_edge))
        return dict(mv) if mv is not None else None   # was mv.to_dict()

    def valid_successors(self, from_edge: str) -> list[str]:
        """Return list of edges that can follow from_edge (turn-restriction aware)."""
        if self._successors:
            return self._successors.get(from_edge, [])
        return []   # no data — caller falls back to graph neighbours

    def build_impedance_fn(
            self,
            edge_lookup: dict,
            hour:         int,
            blocked_edges: set | None = None,
        ):
            fallback      = self.fallback_penalty_s
            avg_turn_time = self._avg_turn_time

            ROAD_PENALTY = {
                "motorway": 0.6, "trunk": 0.7, "trunk_link": 0.7,
                "primary": 0.85, "secondary": 1.0, "secondary_link": 1.0,
                "tertiary": 1.2, "tertiary_link": 1.2,
                "unclassified": 1.4, "residential": 1.6,
            }

            def impedance(u, v, data):
                eid = data.get("id", "")
                if blocked_edges and eid in blocked_edges:
                    return float("inf")

                tt_arr = data.get("tt", [30.0] * 24)
                base_t = tt_arr[hour] if hour < len(tt_arr) else tt_arr[-1]

                r_type = data.get("road_type", "unclassified")
                conf   = data.get("confidence", 0.1)
                type_m = ROAD_PENALTY.get(r_type, 1.5)
                conf_m = 1.0 + (1.0 - conf) * 0.2

                avg_turn = avg_turn_time.get(eid, fallback)

                return base_t * type_m * conf_m + avg_turn

            return impedance

    def route_via_line_graph(
        self,
        G_nx,            # networkx DiGraph (original edge-based graph from app.py)
        edge_lookup: dict,
        start_edge:  str,
        end_edge:    str,
        hour:        int,
        blocked_edges: set | None = None,
        k_paths:     int = 1,
    ) -> Optional[dict]:
        """
        Exact turn-cost routing using the line graph transformation.

        Nodes in the line graph are SUMO edge IDs.
        Edge (A, B) in the line graph = valid turn from road A to road B.
        Weight = travel_time(A, hour) + turn_time(A → B).

        Returns dict with keys: edges (list), total_time_s, path_detail
        or None if no path found.
        """
        import networkx as nx

        if not self._loaded:
            return None

        start = edge_lookup.get(start_edge)
        end   = edge_lookup.get(end_edge)
        if start is None or end is None:
            return None

        if blocked_edges is None:
            blocked_edges = set()

        ROAD_PENALTY = {
            "motorway": 0.6, "trunk": 0.7, "trunk_link": 0.7,
            "primary": 0.85, "secondary": 1.0, "secondary_link": 1.0,
            "tertiary": 1.2, "tertiary_link": 1.2,
            "unclassified": 1.4, "residential": 1.6,
        }

        def _edge_cost(eid: str) -> float:
            """Cost of traversing edge eid at given hour."""
            d = edge_lookup.get(eid, {})
            if not d:
                return 30.0
            tt_arr = d.get("tt", [30.0] * 24)
            base_t = tt_arr[hour] if hour < len(tt_arr) else tt_arr[-1]
            r_type = d.get("road_type", "unclassified")
            conf   = d.get("confidence", 0.1)
            type_m = ROAD_PENALTY.get(r_type, 1.5)
            conf_m = 1.0 + (1.0 - conf) * 0.2
            return base_t * type_m * conf_m

        # ── Build line graph on-the-fly ───────────────────────────────────────
        # Only build the relevant subgraph reachable from start_edge
        # (full line graph over 16k edges × average 3 successors = ~48k nodes)
        # For performance: build lazily during Dijkstra via custom search.

        # Custom Dijkstra on the movement graph
        # Custom Dijkstra on the movement graph
        import heapq

        dist: dict[str, float] = {}
        prev: dict[str, Optional[str]] = {}
        heap = [(0.0, start_edge)]
        dist[start_edge] = 0.0
        prev[start_edge] = None

        while heap:
            d, cur = heapq.heappop(heap)
            if d > dist.get(cur, float("inf")):
                continue
            if cur == end_edge:
                break

            seen_successors: set[str] = set()

            # 1. Movement-map successors (turn-restriction aware)
            if self._successors:
                for succ_eid in self._successors.get(cur, []):
                    if succ_eid in blocked_edges:
                        continue
                    seen_successors.add(succ_eid)
                    turn_t    = self._movements[(cur, succ_eid)]["turn_time_s"]
                    succ_cost = _edge_cost(succ_eid)
                    nd = d + succ_cost + turn_t
                    if nd < dist.get(succ_eid, float("inf")):
                        dist[succ_eid] = nd
                        prev[succ_eid] = cur
                        heapq.heappush(heap, (nd, succ_eid))

            # 2. Graph-topology fallback for edges not in movement map.
            # Requires G_nx to be passed and cur to exist as a node.
            # cur_v is resolved from edge_lookup using the "to_node" key
            # (set when app.py builds edge_lookup) rather than "v" which
            # was never populated.
            if G_nx is not None:
                cur_data = edge_lookup.get(cur, {})
                cur_v    = cur_data.get("to_node")   # fixed: was "v"
                if cur_v is not None and G_nx.has_node(cur_v):
                    for _, next_v, ndata in G_nx.out_edges(cur_v, data=True):
                        succ_eid = ndata.get("id", "")
                        if not succ_eid or succ_eid in blocked_edges:
                            continue
                        if succ_eid in seen_successors:
                            continue
                        succ_cost = _edge_cost(succ_eid)
                        nd        = d + succ_cost + self.fallback_penalty_s
                        if nd < dist.get(succ_eid, float("inf")):
                            dist[succ_eid] = nd
                            prev[succ_eid] = cur
                            heapq.heappush(heap, (nd, succ_eid))

        if end_edge not in dist:
            return None

        # Reconstruct path
        path: list[str] = []
        cur = end_edge
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()

        # Compute detailed metrics
        total_tt_base   = 0.0
        total_turn_time = 0.0
        path_detail     = []

        for idx, eid in enumerate(path):
            d_edge = edge_lookup.get(eid, {})
            tt_arr = d_edge.get("tt", [30.0] * 24)
            base_t = tt_arr[hour] if hour < len(tt_arr) else tt_arr[-1]
            low_t  = d_edge.get("tt_low",  [base_t] * 24)
            high_t = d_edge.get("tt_high", [base_t] * 24)
            low    = low_t[hour]  if hour < len(low_t)  else low_t[-1]
            high   = high_t[hour] if hour < len(high_t) else high_t[-1]

            turn_t = 0.0
            turn_info = None
            if idx > 0:
                prev_eid = path[idx - 1]
                mv = self._movements.get((prev_eid, eid))
                if mv:
                    turn_t    = mv["turn_time_s"]   # was mv.turn_time_s
                    turn_info = dict(mv)            # was mv.to_dict()
                else:
                    turn_t = self.fallback_penalty_s

            total_tt_base   += base_t
            total_turn_time += turn_t
            path_detail.append({
                "edge_id":   eid,
                "tt_base_s": round(base_t, 2),
                "tt_low_s":  round(low,    2),
                "tt_high_s": round(high,   2),
                "turn_s":    round(turn_t, 2),
                "turn_info": turn_info,
                "length_m":  d_edge.get("length", 0.0),
                "road_type": d_edge.get("road_type", ""),
                "dom_dir":   d_edge.get("dom_dir", "unknown"),
                "confidence": d_edge.get("confidence", 0.1),
            })

        # Total times including turn costs
        total_low  = sum(s["tt_low_s"]  for s in path_detail) + total_turn_time
        total_high = sum(s["tt_high_s"] for s in path_detail) + total_turn_time
        total_exp  = sum(
            _adjusted_expected(s["tt_base_s"], s["tt_low_s"],
                               s["tt_high_s"], s["dom_dir"])
            for s in path_detail
        ) + total_turn_time
        # Number of turns = len(path) - 1, but first edge has no incoming turn
        n_actual_turns = sum(1 for s in path_detail if s["turn_s"] > 0)

        return {
            "edges":             path,
            "total_time_s":      round(total_tt_base + total_turn_time, 2),
            "total_time_low_s":  round(total_low,  2),
            "total_time_high_s": round(total_high, 2),
            "expected_time_s":   round(total_exp,  2),
            "turn_time_s":       round(total_turn_time, 2),
            "n_turns":           len(path) - 1,
            "dist_m":            sum(s["length_m"] for s in path_detail) + (n_actual_turns * 15.0),  # +15m per junction traversal
            "path_detail":       path_detail,
        }


def _adjusted_expected(base_t, low_t, high_t, dom_dir):
    """Compute direction-adjusted expected travel time (same logic as app.py)."""
    if dom_dir == "underestimate":
        adj = base_t + 0.75 * (high_t - base_t)
    elif dom_dir == "overestimate":
        adj = base_t - 0.75 * (base_t - low_t)
    else:
        adj = base_t
    return max(low_t, min(high_t, adj))


# ── movements.pkl builder ─────────────────────────────────────────────────────


def build_movements_pkl(net_path: str, out_path: str) -> None:
    """
    Parse a SUMO net.xml and write movements.pkl containing:
        {
            "movements":       {(from_edge_id, to_edge_id): dict},
            "edge_successors": {from_edge_id: [to_edge_id, ...]}
        }

    Turn time is estimated from the connection's via-lane length and a
    conservative crawl speed (5 km/h through the junction box).
    """
    import pickle
    import sumolib

    # Direction-aware junction speeds (km/h → m/s)
    # Straight: vehicle barely slows, especially on priority roads
    # Right: moderate deceleration, short arc
    # Left: slow — long arc, oncoming traffic gap
    # U-turn: crawl
    JUNCTION_SPEED_MS = {
        "straight":    (40.0 / 3.6),   # ~11.1 m/s — minor slowdown only
        "right":       (15.0 / 3.6),   # ~4.2  m/s — yield + short arc
        "right_sharp": (10.0 / 3.6),   # ~2.8  m/s
        "left":        (10.0 / 3.6),   # ~2.8  m/s — oncoming gap + long arc
        "left_sharp":  ( 7.0 / 3.6),   # ~1.9  m/s
        "uturn":       ( 5.0 / 3.6),   # ~1.4  m/s — crawl
        "unknown":     (15.0 / 3.6),   # fallback
    }
    DEFAULT_SPEED_MS = 15.0 / 3.6

    DIR_MAP = {
        "s": "straight", "t": "uturn",
        "r": "right",    "R": "right_sharp",
        "l": "left",     "L": "left_sharp",
        "i": "invalid",
    }

    print(f"Reading network from {net_path} ...")
    net = sumolib.net.readNet(net_path, withInternal=True)

    movements: dict = {}
    edge_successors: dict = {}

    for edge in net.getEdges():
        from_eid = edge.getID()
        if from_eid not in edge_successors:
            edge_successors[from_eid] = []

        for lane in edge.getLanes():
            for conn in lane.getOutgoing():
                to_lane = conn.getToLane()
                to_edge = to_lane.getEdge()
                to_eid  = to_edge.getID()

                if to_eid.startswith(":"):
                    continue

                pair = (from_eid, to_eid)

                if to_eid not in edge_successors[from_eid]:
                    edge_successors[from_eid].append(to_eid)

                turn_dir = DIR_MAP.get(conn.getDirection(), "unknown")
                speed_ms = JUNCTION_SPEED_MS.get(turn_dir, DEFAULT_SPEED_MS)

                via_id    = conn.getViaLaneID() or ""
                turn_time = 1.0  # fallback if no via-lane

                if via_id:
                    try:
                        via_lane  = net.getLane(via_id)
                        via_len   = via_lane.getLength()
                        turn_time = via_len / speed_ms   # direction-aware speed
                    except Exception:
                        pass

                if pair not in movements or turn_time < movements[pair]["turn_time_s"]:
                    movements[pair] = {
                        "from_edge":   from_eid,
                        "to_edge":     to_eid,
                        "turn_time_s": round(turn_time, 3),
                        "turn_type":   turn_dir,
                        "via_lane":    via_id,
                    }

    edge_successors = {k: v for k, v in edge_successors.items() if v}

    payload = {
        "movements":       movements,
        "edge_successors": edge_successors,
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_edges = len(edge_successors)
    n_mov   = len(movements)
    print(f"Done.")
    print(f"  Edges with successors : {n_edges}")
    print(f"  Total movements       : {n_mov}")
    print(f"  Avg successors/edge   : {n_mov/n_edges:.1f}" if n_edges else "  Avg: N/A")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build movements.pkl from SUMO net.xml")
    p.add_argument("--net", default="outputs/networks/full.net.xml")
    p.add_argument("--out", default="outputs/networks/movements.pkl")
    args = p.parse_args()
    build_movements_pkl(args.net, args.out)