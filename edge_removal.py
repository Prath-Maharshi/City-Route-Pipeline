"""
edge_removal.py
===============
Traffic flow redistribution under edge (road) removal.

Algorithm (per edge e, per hour t):
    1. Remove edge e from the routing graph
    2. Take q_e(t) as the displaced flow
    3. Find k=3 shortest alternative paths u→v using Dijkstra on
       the pruned graph with travel_time[t] as weights
    4. Distribute q_e(t) across paths ∝ exp(-θ × path_travel_time)
    5. Add redistributed flow to each edge on each path
    6. Recompute travel times via BPR on updated flows
    7. Report delta_flows, congestion flags, total delay delta

Designed to be plug-and-play with the web app:
    - NetworkState is loaded once and cached
    - remove_edge() is a pure function (returns new state, never mutates)
    - All results are JSON-serialisable via .to_dict()

Usage (standalone):
    python edge_removal.py \\
        --pkl  outputs/graph_reconstruction/gurugram_traffic_arrays.pkl \\
        --net  outputs/networks/full.net.xml \\
        --edge "-123456789#0" \\
        --hour 8

Usage (as library):
    from edge_removal import NetworkState, remove_edge

    state  = NetworkState.load(pkl_path, net_path)
    result = remove_edge(state, edge_id="-123456789#0", hour=8)
    print(result.summary())
    result.to_dict()   # JSON-ready for web app
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

log = logging.getLogger("edge_removal")

# ── BPR parameters (calibrated for Indian mixed traffic) ─────────────────────
BPR_ALPHA = 0.15
BPR_BETA  = 4.0

# Logit assignment temperature — controls how sharply flow prefers
# the shortest path. Higher θ → more concentrated on best route.
THETA_DEFAULT = 0.05   # 1/minute — calibrated for urban Gurugram

# Number of alternative paths to consider
K_PATHS = 3

# Maximum path search depth (hops) — prevents runaway Dijkstra on large graphs
MAX_PATH_HOPS = 60

# Capacity threshold above which an edge is considered congested
CONGESTION_VC_THRESHOLD = 0.85


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK STATE  (loaded once, shared across requests)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NetworkState:
    """
    Immutable network state loaded from the reconstruction pickle.

    Attributes
    ----------
    edge_ids      : list[str]         Edge IDs in canonical order
    edge_index    : dict[str, int]    ID → index
    flows         : ndarray(T, N)     Reconstructed flows (veh/h)
    travel_time   : ndarray(T, N)     Travel times (s)
    capacity      : ndarray(N,)       Capacity (veh/h)
    length_m      : ndarray(N,)       Edge lengths (m)
    speed_free    : ndarray(N,)       Free-flow speed (km/h)
    within_reach  : ndarray(N, bool)  True if within 10 hops of a sensor
    data_source   : list[str]         "sensor"|"reconstructed"|"prior" per edge
    T             : int               Number of time steps
    N             : int               Number of edges
    adj           : dict[int, list[tuple[int,float,int]]]
                    Adjacency list: node_id → [(node_id, edge_idx, direction)]
    nodes_from    : ndarray(N,)       From-node index per edge
    nodes_to      : ndarray(N,)       To-node index per edge
    node_index    : dict[str, int]    Node ID → node index
    road_type     : list[str]         Road type per edge
    """
    edge_ids:     list
    edge_index:   dict
    flows:        np.ndarray
    travel_time:  np.ndarray
    capacity:     np.ndarray
    length_m:     np.ndarray
    speed_free:   np.ndarray
    within_reach: np.ndarray
    data_source:  list
    road_type:    list
    T:            int
    N:            int
    # Graph topology (built at load time)
    adj:          dict = field(default_factory=dict)   # node → [(node, edge_idx)]
    nodes_from:   np.ndarray = field(default=None)
    nodes_to:     np.ndarray = field(default=None)
    node_ids:     list = field(default_factory=list)   # node_index → node_id_str
    node_index:   dict = field(default_factory=dict)   # node_id_str → int

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, pkl_path: str, net_path: str) -> "NetworkState":
        """
        Load network state from reconstruction pickle + SUMO net.xml.
        This is the expensive step — call once and cache the result.
        """
        t0 = time.time()
        log.info("Loading NetworkState from %s ...", pkl_path)

        with open(pkl_path, "rb") as f:
            arrays = pickle.load(f)

        edge_ids   = arrays["edge_ids"]
        N          = len(edge_ids)
        edge_index = {eid: i for i, eid in enumerate(edge_ids)}
        flows      = arrays["flows"].astype(np.float64)
        tt         = arrays["travel_time"].astype(np.float64)
        capacity   = arrays["capacity"].astype(np.float64)
        T          = flows.shape[0]

        # Length from travel_time and speed: L = tt * speed / 3.6
        speed_arr  = arrays["speed"].astype(np.float64)
        length_m   = np.array([
            float(tt[:, i].mean() * speed_arr[:, i].mean() / 3.6)
            for i in range(N)
        ], dtype=np.float64)
        length_m   = np.maximum(length_m, 1.0)

        speed_free = speed_arr.max(axis=0)   # free-flow ≈ max observed speed
        speed_free = np.maximum(speed_free, 5.0)

        within_reach = arrays.get("within_reach", np.ones(N, dtype=bool))

        # data_source: infer if not stored
        if "prior_only" in arrays:
            data_source = ["prior" if arrays["prior_only"][i]
                           else ("sensor" if arrays["conf_observed"][i] > 0
                                 else "reconstructed")
                           for i in range(N)]
        else:
            conf_obs = arrays.get("conf_observed", np.zeros(N))
            data_source = ["sensor" if conf_obs[i] > 0
                           else "reconstructed" if within_reach[i]
                           else "prior"
                           for i in range(N)]

        # Build topology from SUMO net.xml
        adj, nodes_from, nodes_to, node_ids, node_index, road_type = \
            _build_topology(net_path, edge_ids, edge_index, N)

        state = cls(
            edge_ids    = edge_ids,
            edge_index  = edge_index,
            flows       = flows,
            travel_time = tt,
            capacity    = capacity,
            length_m    = length_m,
            speed_free  = speed_free,
            within_reach = within_reach,
            data_source  = data_source,
            road_type    = road_type,
            T           = T,
            N           = N,
            adj         = adj,
            nodes_from  = nodes_from,
            nodes_to    = nodes_to,
            node_ids    = node_ids,
            node_index  = node_index,
        )

        log.info("  NetworkState ready: N=%d edges, T=%d hours, "
                 "%d nodes  (%.1fs)", N, T, len(node_ids), time.time() - t0)
        return state

    def edge_id_to_idx(self, edge_id: str) -> Optional[int]:
        return self.edge_index.get(edge_id)

    def idx_to_edge_id(self, idx: int) -> str:
        return self.edge_ids[idx]


def _build_topology(
    net_path: str,
    edge_ids: list,
    edge_index: dict,
    N: int,
) -> tuple:
    """
    Build a node-based adjacency list from SUMO net.xml.

    Returns
    -------
    adj        : dict[int, list[(to_node_int, edge_idx)]]
    nodes_from : ndarray(N,)   from-node index per edge
    nodes_to   : ndarray(N,)   to-node index per edge
    node_ids   : list[str]     node_index → node_id string
    node_index : dict[str,int] node_id string → node_index
    road_type  : list[str]     road type per edge
    """
    import sumolib

    log.info("  Building topology from %s ...", net_path)
    net = sumolib.net.readNet(net_path)

    # Build node index
    node_ids   = []
    node_index = {}
    for node in net.getNodes():
        nid = node.getID()
        node_index[nid] = len(node_ids)
        node_ids.append(nid)

    nodes_from = np.full(N, -1, dtype=np.int32)
    nodes_to   = np.full(N, -1, dtype=np.int32)
    road_type  = ["unknown"] * N

    # Adjacency: node → [(to_node, edge_idx)]
    adj = {k: [] for k in range(len(node_ids))}

    for edge in net.getEdges():
        eid = edge.getID()
        if eid not in edge_index:
            continue
        i = edge_index[eid]
        u = node_index.get(edge.getFromNode().getID(), -1)
        v = node_index.get(edge.getToNode().getID(),   -1)
        if u < 0 or v < 0:
            continue
        nodes_from[i] = u
        nodes_to[i]   = v
        rt = edge.getType() or "unknown"
        road_type[i] = rt.split(".")[-1]

        adj[u].append((v, i))   # directed: u → v via edge i

    n_connected = sum(1 for i in range(N) if nodes_from[i] >= 0)
    log.info("  Topology: %d nodes, %d/%d edges connected",
             len(node_ids), n_connected, N)

    return adj, nodes_from, nodes_to, node_ids, node_index, road_type


# ═══════════════════════════════════════════════════════════════════════════════
# REMOVAL RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RemovalResult:
    """
    Result of removing one edge from the network at a given hour.

    All arrays are length N (one value per edge).
    """
    edge_id:       str
    hour:          int

    # Flow deltas (positive = gained flow, negative = lost flow)
    # delta_flows[i] is the change in flow on edge i after removal
    delta_flows:   np.ndarray    # (N,) veh/h

    # Updated absolute flows after redistribution
    flows_updated: np.ndarray    # (N,) veh/h

    # Updated travel times after BPR recalculation
    tt_updated:    np.ndarray    # (N,) seconds

    # Edges that are now congested (v/c > threshold) AND weren't before
    newly_congested: np.ndarray  # (N,) bool

    # Edges that were congested before but are now relieved
    newly_relieved:  np.ndarray  # (N,) bool

    # Total displaced flow (sum of flow that was on removed edge)
    displaced_flow:  float       # veh/h

    # How much of displaced flow was successfully rerouted
    rerouted_flow:   float       # veh/h

    # Network-wide total travel time before and after
    total_tt_before: float       # veh·s
    total_tt_after:  float       # veh·s
    total_delay:     float       # veh·s extra

    # Alternative paths found
    paths:           list        # list of {path, travel_time, flow_assigned}

    # Any warning (e.g. no alternative path found)
    warning:         Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"Edge removal: {self.edge_id}  hour={self.hour:02d}:00",
            f"  Displaced flow   : {self.displaced_flow:.1f} veh/h",
            f"  Rerouted flow    : {self.rerouted_flow:.1f} veh/h "
            f"({self.rerouted_flow/max(self.displaced_flow,1)*100:.1f}%)",
            f"  Alternative paths: {len(self.paths)}",
            f"  Newly congested  : {int(self.newly_congested.sum())} edges",
            f"  Newly relieved   : {int(self.newly_relieved.sum())} edges",
            f"  Total extra delay: {self.total_delay/3600:.2f} veh·h",
        ]
        if self.warning:
            lines.append(f"  WARNING: {self.warning}")
        for k, p in enumerate(self.paths):
            lines.append(
                f"  Path {k+1}: {len(p['edges'])} edges  "
                f"tt={p['travel_time_s']:.1f}s  "
                f"flow={p['flow_assigned']:.1f} veh/h  "
                f"weight={p['weight']:.3f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable dict for the web app."""
        top_gained = _top_k_edges_by_delta(self.delta_flows, k=20, positive=True)
        top_lost   = _top_k_edges_by_delta(self.delta_flows, k=20, positive=False)

        return {
            "edge_id":         self.edge_id,
            "hour":            self.hour,
            "displaced_flow":  round(self.displaced_flow, 2),
            "rerouted_flow":   round(self.rerouted_flow, 2),
            "reroute_pct":     round(self.rerouted_flow / max(self.displaced_flow, 1) * 100, 1),
            "total_delay_veh_h": round(self.total_delay / 3600, 3),
            "n_newly_congested": int(self.newly_congested.sum()),
            "n_newly_relieved":  int(self.newly_relieved.sum()),
            "paths": [
                {
                    "edges":          p["edges"],
                    "travel_time_s":  round(p["travel_time_s"], 2),
                    "flow_assigned":  round(p["flow_assigned"], 2),
                    "weight":         round(p["weight"], 4),
                }
                for p in self.paths
            ],
            "top_flow_gained":  top_gained,
            "top_flow_lost":    top_lost,
            "newly_congested_edges": [
                i for i in range(len(self.newly_congested))
                if self.newly_congested[i]
            ],
            "warning": self.warning,
        }

    def to_geojson_delta(self, state: NetworkState, net) -> dict:
        """
        GeoJSON FeatureCollection of affected edges with flow deltas.
        Only edges with |delta| > threshold are included.
        net: sumolib network object for coordinates.
        """
        threshold = max(1.0, self.displaced_flow * 0.01)
        features  = []

        for i in range(state.N):
            delta = float(self.delta_flows[i])
            if abs(delta) < threshold:
                continue

            eid = state.edge_ids[i]
            try:
                edge = net.getEdge(eid)
                shape = edge.getShape()
                coords = [[round(lon, 6), round(lat, 6)]
                          for lon, lat in [net.convertXY2LonLat(x, y)
                                           for x, y in shape]]
            except Exception:
                continue

            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "edge_id":      eid,
                    "delta_flow":   round(delta, 2),
                    "flow_before":  round(float(state.flows[self.hour, i]), 2),
                    "flow_after":   round(float(self.flows_updated[i]), 2),
                    "tt_before":    round(float(state.travel_time[self.hour, i]), 2),
                    "tt_after":     round(float(self.tt_updated[i]), 2),
                    "congested":    bool(self.newly_congested[i]),
                    "relieved":     bool(self.newly_relieved[i]),
                    "data_source":  state.data_source[i],
                },
            })

        return {
            "type": "FeatureCollection",
            "metadata": {
                "removed_edge":   self.edge_id,
                "hour":           self.hour,
                "n_affected":     len(features),
                "total_delay_veh_h": round(self.total_delay / 3600, 3),
            },
            "features": features,
        }


def _top_k_edges_by_delta(delta_flows, k=20, positive=True):
    if positive:
        idx = np.argsort(-delta_flows)
        idx = idx[delta_flows[idx] > 0]
    else:
        idx = np.argsort(delta_flows)
        idx = idx[delta_flows[idx] < 0]
    idx = idx[:k]
    return [{"edge_idx": int(i), "delta": round(float(delta_flows[i]), 2)}
            for i in idx]


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ALGORITHM: remove_edge()
# ═══════════════════════════════════════════════════════════════════════════════

def remove_edge(
    state:    NetworkState,
    edge_id:  str,
    hour:     int,
    k:        int        = K_PATHS,
    theta:    float      = THETA_DEFAULT,
    nx_graph  = None,
) -> RemovalResult:
    import networkx as nx

    hour = max(0, min(state.T - 1, hour))
    idx  = state.edge_index.get(edge_id)

    if idx is None:
        return RemovalResult(
            edge_id=edge_id, hour=hour,
            delta_flows=np.zeros(state.N),
            flows_updated=state.flows[hour].copy(),
            tt_updated=state.travel_time[hour].copy(),
            newly_congested=np.zeros(state.N, dtype=bool),
            newly_relieved=np.zeros(state.N, dtype=bool),
            displaced_flow=0, rerouted_flow=0,
            total_tt_before=0, total_tt_after=0, total_delay=0,
            paths=[], warning=f"Edge '{edge_id}' not found in network",
        )

    tt_current    = state.travel_time[hour].copy()
    flows_current = state.flows[hour].copy()

    # ── Step 0: Stranded flow (simplified — removed edge only) ───────────────
    stranded_mask, displaced_flow = _find_stranded_flow(
        state, idx, hour, nx_graph
    )

    # ── Sanity cap: displaced flow <= edge capacity × 1.5 ────────────────────
    capacity_i     = float(state.capacity[idx])
    displaced_flow = min(displaced_flow, capacity_i * 1.5)
    displaced_flow = max(displaced_flow, 0.0)

    # ── Step 1: Find k alternative paths ─────────────────────────────────────
    paths_raw = []

    if nx_graph is not None:
        if not nx_graph.has_edge(*(
            edge_data := next(
                ((u, v) for u, v, d in nx_graph.edges(data=True) if d.get("id") == edge_id),
                (None, None)
            )
        )):
            pass
        else:
            u_node, v_node = edge_data

            def weight_fn(u, v, d):
                eid_d = d.get("id", "")
                i_d   = state.edge_index.get(eid_d)
                if i_d is None:
                    return float(d.get("tt", [30.0] * 24)[hour])
                return float(tt_current[i_d])

            G_pruned = nx_graph.copy()
            G_pruned.remove_edge(u_node, v_node)

            try:
                path_gen = nx.shortest_simple_paths(
                    G_pruned, u_node, v_node, weight=weight_fn
                )
                for path_nodes in path_gen:
                    if len(paths_raw) >= k:
                        break
                    edge_indices = []
                    path_tt = 0.0
                    valid = True
                    for i_n in range(len(path_nodes) - 1):
                        pn_u = path_nodes[i_n]
                        pn_v = path_nodes[i_n + 1]
                        if not G_pruned.has_edge(pn_u, pn_v):
                            valid = False; break
                        seg_eid = G_pruned[pn_u][pn_v].get("id", "")
                        seg_idx = state.edge_index.get(seg_eid)
                        if seg_idx is None:
                            seg_tt = float(G_pruned[pn_u][pn_v].get(
                                "tt", [30.0] * 24)[hour])
                        else:
                            seg_idx_int = int(seg_idx)
                            edge_indices.append(seg_idx_int)
                            seg_tt = float(tt_current[seg_idx_int])
                        path_tt += seg_tt
                    if valid and edge_indices:
                        paths_raw.append({
                            "edges": edge_indices,
                            "travel_time_s": path_tt,
                        })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

    if not paths_raw and nx_graph is None:
        u_int = int(state.nodes_from[idx])
        v_int = int(state.nodes_to[idx])
        if u_int >= 0 and v_int >= 0:
            paths_raw = _k_shortest_paths(
                adj=state.adj, nodes_from=state.nodes_from,
                nodes_to=state.nodes_to, tt=tt_current,
                source=u_int, target=v_int,
                removed_edge=idx, k=k, max_hops=MAX_PATH_HOPS,
            )

    # ── No path found ─────────────────────────────────────────────────────────
    if not paths_raw:
        flows_updated = flows_current.copy()
        flows_updated[idx] = 0.0

        tt_bpr_before = _bpr_travel_times(flows_current, state.capacity, state.length_m, state.speed_free)
        tt_bpr_after  = _bpr_travel_times(flows_updated, state.capacity, state.length_m, state.speed_free)
        tt_updated    = np.maximum(tt_current + (tt_bpr_after - tt_bpr_before), 0.1)

        total_before = float(np.sum(flows_current * tt_current))
        total_after  = float(np.sum(flows_updated * tt_updated))
        return RemovalResult(
            edge_id=edge_id, hour=hour,
            delta_flows=flows_updated - flows_current,
            flows_updated=flows_updated, tt_updated=tt_updated,
            newly_congested=_congestion_delta(flows_current, flows_updated,
                                               state.capacity, new=True),
            newly_relieved=_congestion_delta(flows_current, flows_updated,
                                              state.capacity, new=False),
            displaced_flow=displaced_flow, rerouted_flow=0.0,
            total_tt_before=total_before, total_tt_after=total_after,
            total_delay=total_after - total_before,
            paths=[],
            warning="No alternative path found — network may be disconnected at this edge",
        )

    # ── Step 2: Logit assignment ──────────────────────────────────────────────
    path_tts = np.array([p["travel_time_s"] for p in paths_raw])
    log_w    = -theta * path_tts
    log_w   -= log_w.max()
    weights  = np.exp(log_w)
    weights /= weights.sum()
    flow_assignments = displaced_flow * weights
    paths_annotated  = []

    # ── Step 3: Delta flows ───────────────────────────────────────────────────
    delta_flows = np.zeros(state.N, dtype=np.float64)

    # Only zero out the single removed edge — not collateral stranded edges
    delta_flows[idx] = -float(flows_current[idx])

    # Distribute rerouted flow onto alternative paths
    for p, w, fa in zip(paths_raw, weights, flow_assignments):
        for edge_i in p["edges"]:
            delta_flows[edge_i] += fa
        paths_annotated.append({
            "edges":         [state.edge_ids[ei] for ei in p["edges"]],
            "edge_indices":  p["edges"],
            "travel_time_s": float(p["travel_time_s"]),
            "flow_assigned": float(fa),
            "weight":        float(w),
            "n_hops":        len(p["edges"]),
        })

    # ── Step 4: Updated flows + BPR ──────────────────────────────────────────
    flows_updated = np.maximum(flows_current + delta_flows, 0.0)

    tt_bpr_before = _bpr_travel_times(flows_current, state.capacity, state.length_m, state.speed_free)
    tt_bpr_after  = _bpr_travel_times(flows_updated, state.capacity, state.length_m, state.speed_free)
    tt_updated    = np.maximum(tt_current + (tt_bpr_after - tt_bpr_before), 0.1)

    # ── Step 5: Congestion analysis ───────────────────────────────────────────
    newly_congested = _congestion_delta(flows_current, flows_updated,
                                         state.capacity, new=True)
    newly_relieved  = _congestion_delta(flows_current, flows_updated,
                                         state.capacity, new=False)

    total_before = float(np.sum(flows_current * tt_current))
    total_after  = float(np.sum(flows_updated * tt_updated))
    rerouted     = float(sum(p["flow_assigned"] for p in paths_annotated))

    log.info("Removed %s  hour=%02d  displaced=%.1f  rerouted=%.0f%%  "
             "paths=%d  congested=%d  delay=%.2f veh·h",
             edge_id, hour, displaced_flow,
             rerouted / max(displaced_flow, 1) * 100,
             len(paths_annotated), int(newly_congested.sum()),
             (total_after - total_before) / 3600)

    return RemovalResult(
        edge_id=edge_id, hour=hour,
        delta_flows=delta_flows,
        flows_updated=flows_updated, tt_updated=tt_updated,
        newly_congested=newly_congested, newly_relieved=newly_relieved,
        displaced_flow=displaced_flow, rerouted_flow=rerouted,
        total_tt_before=total_before, total_tt_after=total_after,
        total_delay=total_after - total_before,
        paths=paths_annotated,
    )

def _find_stranded_flow(
    state:        NetworkState,
    removed_idx:  int,
    hour:         int,
    nx_graph      = None,
) -> tuple[np.ndarray, float]:
    """
    Simplified stranding: only the removed edge loses its flow.
    Cascade propagation is disabled — it over-fires on dense urban graphs,
    causing phantom multi-hop displacement that inflates displaced_flow 10-100x.

    Re-enable cascade only after validating on isolated corridors.
    """
    stranded = np.zeros(state.N, dtype=bool)
    stranded[removed_idx] = True
    stranded_flow = float(state.flows[hour, removed_idx])

    log.info("  Stranded edges: 1  displaced_flow=%.1f veh/h", stranded_flow)
    return stranded, stranded_flow


def remove_edge_all_hours(
    state:    NetworkState,
    edge_id:  str,
    k:        int   = K_PATHS,
    theta:    float = THETA_DEFAULT,
    nx_graph  = None,
) -> list:
    """Run remove_edge for all T hours. Returns list of T RemovalResults."""
    return [remove_edge(state, edge_id, t, k=k, theta=theta, nx_graph=nx_graph)
            for t in range(state.T)]


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

def _k_shortest_paths(
    adj:          dict,
    nodes_from:   np.ndarray,
    nodes_to:     np.ndarray,
    tt:           np.ndarray,
    source:       int,
    target:       int,
    removed_edge: int,
    k:            int,
    max_hops:     int,
) -> list[dict]:
    """
    Find up to k shortest paths from source to target (in node space)
    using Yen's algorithm with the removed edge excluded.

    Returns list of dicts:
        {"edges": [edge_idx, ...], "travel_time_s": float}
    Paths are node-disjoint at intermediate nodes where possible.
    """
    import heapq

    def dijkstra(adj, nodes_from, nodes_to, tt, src, tgt,
                  blocked_edges: frozenset, blocked_nodes: frozenset) -> Optional[dict]:
        """
        Standard Dijkstra returning the cheapest path.
        blocked_edges: edge indices to skip
        blocked_nodes: intermediate node indices to skip
        """
        dist = {}
        prev = {}  # node → (prev_node, edge_idx)
        heap = [(0.0, src)]
        dist[src] = 0.0

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if u == tgt:
                # Reconstruct path
                path_edges = []
                cur = tgt
                hops = 0
                while cur != src:
                    pn, ei = prev[cur]
                    path_edges.append(ei)
                    cur = pn
                    hops += 1
                    if hops > max_hops:
                        return None
                path_edges.reverse()
                return {"edges": path_edges, "travel_time_s": d}

            if adj.get(u) is None:
                continue

            for (vn, ei) in adj[u]:
                if ei in blocked_edges:
                    continue
                if vn != tgt and vn in blocked_nodes:
                    continue
                nd = d + float(tt[ei])
                if nd < dist.get(vn, float("inf")):
                    dist[vn] = nd
                    prev[vn] = (u, ei)
                    heapq.heappush(heap, (nd, vn))

        return None  # no path

    # ── Yen's algorithm ────────────────────────────────────────────────────────
    blocked_base = frozenset([removed_edge])
    first = dijkstra(adj, nodes_from, nodes_to, tt,
                     source, target, blocked_base, frozenset())
    if first is None:
        return []

    A = [first]   # accepted paths
    B = []        # candidate paths (heap)

    for _ in range(k - 1):
        last = A[-1]
        for spur_idx in range(len(last["edges"])):
            spur_node  = source if spur_idx == 0 \
                         else int(nodes_to[last["edges"][spur_idx - 1]])
            root_edges = last["edges"][:spur_idx]
            root_tt    = sum(float(tt[e]) for e in root_edges)

            # Block edges that are used by existing accepted paths
            # at the same root path prefix
            blocked_e = set(blocked_base)
            blocked_n = set()
            for p in A:
                if (len(p["edges"]) > spur_idx and
                        p["edges"][:spur_idx] == root_edges):
                    blocked_e.add(p["edges"][spur_idx])

            # Block root path nodes (except spur_node) to prevent loops
            cur = source
            for ei in root_edges:
                if cur != spur_node:
                    blocked_n.add(cur)
                cur = int(nodes_to[ei])

            spur = dijkstra(adj, nodes_from, nodes_to, tt,
                            spur_node, target,
                            frozenset(blocked_e), frozenset(blocked_n))
            if spur is None:
                continue

            # Candidate = root + spur
            cand_edges = root_edges + spur["edges"]
            cand_tt    = root_tt    + spur["travel_time_s"]
            cand       = {"edges": cand_edges, "travel_time_s": cand_tt}

            # Dedup: skip if already in A or B
            if not any(c["edges"] == cand_edges for c in A) and \
               not any(c["edges"] == cand_edges for _, c in B):
                heapq.heappush(B, (cand_tt, cand))

        if not B:
            break
        _, best = heapq.heappop(B)
        A.append(best)

    return A


# ═══════════════════════════════════════════════════════════════════════════════
# BPR SPEED / TRAVEL TIME UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

def _bpr_travel_times(
    flows:      np.ndarray,
    capacity:   np.ndarray,
    length_m:   np.ndarray,
    speed_free: np.ndarray,
) -> np.ndarray:
    """
    Compute per-edge travel times (s) using BPR speed model.
    v(i) = v_free[i] / (1 + α (q/C)^β)
    t(i) = length[i] × 3.6 / v(i)
    """
    vc_clip   = np.minimum(flows / np.maximum(capacity, 1.0), 10.0)
    speed_bpr = speed_free / (1.0 + BPR_ALPHA * vc_clip ** BPR_BETA)
    speed_bpr = np.maximum(speed_bpr, 0.5)   # floor 0.5 km/h
    return length_m * 3.6 / speed_bpr


def _congestion_delta(
    flows_before: np.ndarray,
    flows_after:  np.ndarray,
    capacity:     np.ndarray,
    new:          bool,
) -> np.ndarray:
    """
    Returns boolean mask of edges that:
      new=True  → newly congested (was fine, now v/c > threshold)
      new=False → newly relieved  (was congested, now v/c < threshold)
    """
    thr = CONGESTION_VC_THRESHOLD
    vc_before = flows_before / np.maximum(capacity, 1.0)
    vc_after  = flows_after  / np.maximum(capacity, 1.0)
    if new:
        return (vc_before <= thr) & (vc_after > thr)
    else:
        return (vc_before >  thr) & (vc_after <= thr)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH ANALYSIS: edge criticality scoring
# ═══════════════════════════════════════════════════════════════════════════════

def criticality_scores(
    state:    NetworkState,
    hour:     int = 8,
    edge_ids: Optional[list] = None,
    k:        int = K_PATHS,
    theta:    float = THETA_DEFAULT,
) -> list[dict]:
    """
    Compute a criticality score for each edge = extra delay caused by removal.

    Scores are normalised to [0, 1] across the evaluated edges.
    Only sensor and reconstructed edges are evaluated (prior-only skipped).

    Returns list of dicts sorted by criticality descending.
    """
    if edge_ids is None:
        # Evaluate sensor + reconstructed edges only
        edge_ids = [
            state.edge_ids[i] for i in range(state.N)
            if state.data_source[i] in ("sensor", "reconstructed")
            and state.nodes_from[i] >= 0
            and state.flows[hour, i] > 5.0   # skip trivially empty edges
        ]

    log.info("Computing criticality for %d edges at hour %02d:00 ...",
             len(edge_ids), hour)

    scores = []
    for eid in edge_ids:
        r = remove_edge(state, eid, hour, k=k, theta=theta)
        scores.append({
            "edge_id":          eid,
            "displaced_flow":   r.displaced_flow,
            "rerouted_pct":     r.rerouted_flow / max(r.displaced_flow, 1) * 100,
            "total_delay_veh_h": r.total_delay / 3600,
            "n_congested":      int(r.newly_congested.sum()),
            "warning":          r.warning,
        })

    # Normalise delay to [0, 1]
    delays = np.array([s["total_delay_veh_h"] for s in scores])
    if delays.max() > 0:
        delays_norm = delays / delays.max()
    else:
        delays_norm = np.zeros_like(delays)

    for s, cn in zip(scores, delays_norm):
        s["criticality"] = round(float(cn), 4)

    scores.sort(key=lambda s: -s["criticality"])
    log.info("  Top edge: %s  criticality=%.4f  delay=%.2f veh·h",
             scores[0]["edge_id"] if scores else "—",
             scores[0]["criticality"] if scores else 0,
             scores[0]["total_delay_veh_h"] if scores else 0)
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# WEB APP INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeRemovalService:
    """
    Stateful service class for the web app.

    Usage:
        service = EdgeRemovalService(pkl_path, net_path)
        service.load()              # call once at startup (background thread)
        service.set_nx_graph(G)     # inject the already-built NetworkX DiGraph
                                    # from app.py — avoids topology rebuild

        # Per-request:
        result_dict = service.simulate(edge_id="-123#0", hour=8)
        geojson     = service.simulate_geojson(edge_id="-123#0", hour=8)
    """

    def __init__(self, pkl_path: str, net_path: str):
        self.pkl_path  = pkl_path
        self.net_path  = net_path
        self._state:  Optional[NetworkState] = None
        self._net     = None   # sumolib net for GeoJSON export
        self._G       = None   # NetworkX DiGraph injected from app.py
        self._loaded  = False

    def set_nx_graph(self, G) -> None:
        """
        Inject the NetworkX DiGraph built by app.py.

        This is the authoritative routing graph — it was built from the
        same GeoJSON + sumolib edges the router uses, so it's guaranteed
        to be connected correctly. Call this after load() completes.
        """
        self._G = G
        log.info("EdgeRemovalService: NetworkX graph injected "
                 "(%d nodes, %d edges)", G.number_of_nodes(), G.number_of_edges())

    def load(self) -> None:
        self._state  = NetworkState.load(self.pkl_path, self.net_path)
        try:
            import sumolib
            self._net = sumolib.net.readNet(self.net_path)
        except Exception as ex:
            log.warning("Could not load sumolib net for GeoJSON: %s", ex)
        self._loaded = True
        log.info("EdgeRemovalService ready.")

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def n_edges(self) -> int:
        return self._state.N if self._state else 0

    def simulate(
        self,
        edge_id: str,
        hour:    int   = 8,
        k:       int   = K_PATHS,
        theta:   float = THETA_DEFAULT,
    ) -> dict:
        """Returns JSON-serialisable result dict."""
        if not self._loaded:
            return {"error": "Service not loaded"}
        r = remove_edge(self._state, edge_id, hour, k=k, theta=theta,
                        nx_graph=self._G)
        return r.to_dict()

    def simulate_all_hours(
        self,
        edge_id: str,
        k:       int   = K_PATHS,
        theta:   float = THETA_DEFAULT,
    ) -> list[dict]:
        """Returns list of result dicts for all 24 hours."""
        if not self._loaded:
            return [{"error": "Service not loaded"}]
        results = remove_edge_all_hours(self._state, edge_id, k=k, theta=theta,
                                         nx_graph=self._G)
        return [r.to_dict() for r in results]

    def simulate_geojson(
        self,
        edge_id: str,
        hour:    int   = 8,
        k:       int   = K_PATHS,
        theta:   float = THETA_DEFAULT,
    ) -> dict:
        """Returns GeoJSON FeatureCollection of affected edges."""
        if not self._loaded:
            return {"error": "Service not loaded"}
        if self._net is None:
            return {"error": "Network geometry not available"}
        r = remove_edge(self._state, edge_id, hour, k=k, theta=theta,
                        nx_graph=self._G)
        return r.to_geojson_delta(self._state, self._net)

    def get_edge_info(self, edge_id: str, hour: int = 8) -> Optional[dict]:
        """Get current state of a single edge (before removal)."""
        if not self._loaded:
            return None
        idx = self._state.edge_index.get(edge_id)
        if idx is None:
            return None
        return {
            "edge_id":     edge_id,
            "road_type":   self._state.road_type[idx],
            "capacity":    round(float(self._state.capacity[idx]), 1),
            "length_m":    round(float(self._state.length_m[idx]), 1),
            "flow_veh_h":  round(float(self._state.flows[hour, idx]), 1),
            "vc_ratio":    round(float(self._state.flows[hour, idx] /
                               max(self._state.capacity[idx], 1)), 4),
            "tt_s":        round(float(self._state.travel_time[hour, idx]), 1),
            "data_source": self._state.data_source[idx],
            "within_reach": bool(self._state.within_reach[idx]),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "pkl": "outputs/graph_reconstruction/gurugram_traffic_arrays.pkl",
    "net": "outputs/networks/full.net.xml",
}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Edge removal & flow redistribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pkl",   default=DEFAULTS["pkl"])
    p.add_argument("--net",   default=DEFAULTS["net"])
    p.add_argument("--edge",  required=True,  help="SUMO edge ID to remove")
    p.add_argument("--hour",  type=int, default=8, help="Hour of day (0-23)")
    p.add_argument("--k",     type=int, default=K_PATHS,
                   help="Number of alternative paths")
    p.add_argument("--theta", type=float, default=THETA_DEFAULT,
                   help="Logit temperature (1/s)")
    p.add_argument("--all-hours", action="store_true",
                   help="Run for all 24 hours")
    p.add_argument("--json", action="store_true",
                   help="Output JSON instead of summary text")
    args = p.parse_args()

    for path, label in [(args.pkl, "--pkl"), (args.net, "--net")]:
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}")
            raise SystemExit(1)

    state = NetworkState.load(args.pkl, args.net)

    if args.all_hours:
        results = remove_edge_all_hours(state, args.edge,
                                         k=args.k, theta=args.theta)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            for r in results:
                print(r.summary())
                print()
    else:
        result = remove_edge(state, args.edge, args.hour,
                              k=args.k, theta=args.theta)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.summary())


if __name__ == "__main__":
    main()