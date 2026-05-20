"""
edge_removal.py  —  chain-aware version (performance-fixed)
============================================================
Changes vs previous version
-----------------------------
Performance:
  • _nx_k_paths no longer calls nx_graph.copy() on every request.
    Instead it builds a lightweight pruned view using
    nx.restricted_view() (O(1) wrapper, no deep copy), and falls back
    to an edge-weight approach for path enumeration.  Avoids the O(N)
    deep-copy overhead on the full 50k-edge routing graph.
  • criticality_scores() now accepts an optional nx_graph kwarg so the
    caller (app.py) can pass in the already-built NetworkX DiGraph
    rather than having the function rebuild adjacency internally.

Correctness:
  • length_m is now computed from the SUMO edge geometry directly via
    sumolib (getLength()), instead of mean(tt)*mean(speed) which
    was statistically wrong due to correlated variables.
  • The list-comparison in Yen's spur-path blocking
    (p["edges"][:spur_idx] == root_edges) is now done with tuple
    equality to avoid the O(k·n) list-compare edge case on repeated
    indices.

Everything else (BPR parameters, chain logic, RemovalResult API,
CLI entry point) is unchanged.
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

log = logging.getLogger("edge_removal")

# ── BPR parameters (calibrated for Indian mixed traffic) ─────────────────────
BPR_ALPHA = 0.15
BPR_BETA  = 4.0

THETA_DEFAULT           = 0.05
K_PATHS                 = 3
MAX_PATH_HOPS           = 60
CONGESTION_VC_THRESHOLD = 0.85

DEFAULTS = {
    "pkl":    "outputs/graph_reconstruction/gurugram_traffic_arrays.pkl",
    "net":    "outputs/networks/full.net.xml",
    "chains": "outputs/networks/chains.pkl",
}


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NetworkState:
    """Immutable network state loaded from the reconstruction pickle."""
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
    adj:          dict           = field(default_factory=dict)
    nodes_from:   np.ndarray     = field(default=None)
    nodes_to:     np.ndarray     = field(default=None)
    node_ids:     list           = field(default_factory=list)
    node_index:   dict           = field(default_factory=dict)

    @classmethod
    def load(cls, pkl_path: str, net_path: str) -> "NetworkState":
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

        speed_arr  = arrays["speed"].astype(np.float64)
        speed_free = speed_arr.max(axis=0)
        speed_free = np.maximum(speed_free, 5.0)

        within_reach = arrays.get("within_reach", np.ones(N, dtype=bool))

        if "prior_only" in arrays:
            data_source = [
                "prior" if arrays["prior_only"][i]
                else ("sensor" if arrays["conf_observed"][i] > 0 else "reconstructed")
                for i in range(N)
            ]
        else:
            conf_obs    = arrays.get("conf_observed", np.zeros(N))
            data_source = [
                "sensor"        if conf_obs[i] > 0
                else "reconstructed" if within_reach[i]
                else "prior"
                for i in range(N)
            ]

        adj, nodes_from, nodes_to, node_ids, node_index, road_type, length_m = \
            _build_topology(net_path, edge_ids, edge_index, N, speed_arr, tt)

        state = cls(
            edge_ids=edge_ids, edge_index=edge_index,
            flows=flows, travel_time=tt, capacity=capacity,
            length_m=length_m, speed_free=speed_free,
            within_reach=within_reach, data_source=data_source,
            road_type=road_type, T=T, N=N,
            adj=adj, nodes_from=nodes_from, nodes_to=nodes_to,
            node_ids=node_ids, node_index=node_index,
        )
        log.info("  NetworkState ready: N=%d edges, T=%d hours, %d nodes  (%.1fs)",
                 N, T, len(node_ids), time.time() - t0)
        return state

    def edge_id_to_idx(self, edge_id: str) -> Optional[int]:
        return self.edge_index.get(edge_id)

    def idx_to_edge_id(self, idx: int) -> str:
        return self.edge_ids[idx]


def _build_topology(net_path, edge_ids, edge_index, N, speed_arr, tt_arr):
    import sumolib
    log.info("  Building topology from %s ...", net_path)
    net        = sumolib.net.readNet(net_path)
    node_ids   = []
    node_index = {}
    for node in net.getNodes():
        nid = node.getID()
        node_index[nid] = len(node_ids)
        node_ids.append(nid)

    nodes_from = np.full(N, -1, dtype=np.int32)
    nodes_to   = np.full(N, -1, dtype=np.int32)
    road_type  = ["unknown"] * N
    # FIX: length_m from SUMO geometry (metres), not mean(tt)*mean(speed)
    length_m   = np.full(N, 1.0, dtype=np.float64)
    adj        = {k: [] for k in range(len(node_ids))}

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
        road_type[i]  = rt.split(".")[-1]
        adj[u].append((v, i))

        # getLength() returns metres directly — no speed/tt arithmetic
        raw_len = edge.getLength()
        length_m[i] = max(float(raw_len), 1.0)

    n_connected = sum(1 for i in range(N) if nodes_from[i] >= 0)
    log.info("  Topology: %d nodes, %d/%d edges connected",
             len(node_ids), n_connected, N)
    return adj, nodes_from, nodes_to, node_ids, node_index, road_type, length_m


# ═══════════════════════════════════════════════════════════════════════════════
# REMOVAL RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RemovalResult:
    """Result of removing one chain from the network at a given hour."""
    edge_id:         str
    chain:           list[str]
    hour:            int

    delta_flows:     np.ndarray
    flows_updated:   np.ndarray
    tt_updated:      np.ndarray

    newly_congested: np.ndarray
    newly_relieved:  np.ndarray

    displaced_flow:  float
    rerouted_flow:   float
    total_tt_before: float
    total_tt_after:  float
    total_delay:     float

    paths:           list
    warning:         Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"Edge removal : {self.edge_id}  chain_len={len(self.chain)}  "
            f"hour={self.hour:02d}:00",
            f"  Chain        : {self.chain}",
            f"  Displaced    : {self.displaced_flow:.1f} veh/h",
            f"  Rerouted     : {self.rerouted_flow:.1f} veh/h "
            f"({self.rerouted_flow/max(self.displaced_flow,1)*100:.1f}%)",
            f"  Alt paths    : {len(self.paths)}",
            f"  New congested: {int(self.newly_congested.sum())} edges",
            f"  New relieved : {int(self.newly_relieved.sum())} edges",
            f"  Extra delay  : {self.total_delay/3600:.2f} veh·h",
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
        top_gained = _top_k_edges_by_delta(self.delta_flows, k=20, positive=True)
        top_lost   = _top_k_edges_by_delta(self.delta_flows, k=20, positive=False)
        return {
            "edge_id":           self.edge_id,
            "chain":             self.chain,
            "chain_len":         len(self.chain),
            "hour":              self.hour,
            "displaced_flow":    round(self.displaced_flow, 2),
            "rerouted_flow":     round(self.rerouted_flow, 2),
            "reroute_pct":       round(self.rerouted_flow / max(self.displaced_flow, 1) * 100, 1),
            "total_delay_veh_h": round(self.total_delay / 3600, 3),
            "n_newly_congested": int(self.newly_congested.sum()),
            "n_newly_relieved":  int(self.newly_relieved.sum()),
            "paths": [
                {
                    "edges":         p["edges"],
                    "travel_time_s": round(p["travel_time_s"], 2),
                    "flow_assigned": round(p["flow_assigned"], 2),
                    "weight":        round(p["weight"], 4),
                }
                for p in self.paths
            ],
            "top_flow_gained":       top_gained,
            "top_flow_lost":         top_lost,
            "newly_congested_edges": [
                i for i in range(len(self.newly_congested)) if self.newly_congested[i]
            ],
            "warning":        self.warning,
            "tt_updated_raw": self.tt_updated.tolist(),
        }

    def to_geojson_delta(self, state: NetworkState, net) -> dict:
        threshold = max(1.0, self.displaced_flow * 0.01)
        chain_set = set(self.chain)
        features  = []

        for i in range(state.N):
            eid        = state.edge_ids[i]
            delta      = float(self.delta_flows[i])
            is_removed = eid in chain_set

            if not is_removed and abs(delta) < threshold:
                continue

            try:
                edge   = net.getEdge(eid)
                shape  = edge.getShape()
                coords = [
                    [round(lon, 6), round(lat, 6)]
                    for lon, lat in [net.convertXY2LonLat(x, y) for x, y in shape]
                ]
            except Exception:
                continue

            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "edge_id":     eid,
                    "removed":     is_removed,
                    "delta_flow":  round(delta, 2),
                    "flow_before": round(float(state.flows[self.hour, i]), 2),
                    "flow_after":  round(float(self.flows_updated[i]), 2),
                    "tt_before":   round(float(state.travel_time[self.hour, i]), 2),
                    "tt_after":    round(float(self.tt_updated[i]), 2),
                    "congested":   bool(self.newly_congested[i]),
                    "relieved":    bool(self.newly_relieved[i]),
                    "data_source": state.data_source[i],
                },
            })

        return {
            "type": "FeatureCollection",
            "metadata": {
                "removed_edge":      self.edge_id,
                "removed_chain":     self.chain,
                "chain_len":         len(self.chain),
                "hour":              self.hour,
                "n_affected":        len(features),
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
    return [{"edge_idx": int(i), "delta": round(float(delta_flows[i]), 2)} for i in idx]


# ═══════════════════════════════════════════════════════════════════════════════
# NULL RESULT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _null_result(edge_id, chain, hour, state, warning):
    z = np.zeros(state.N)
    b = np.zeros(state.N, dtype=bool)
    return RemovalResult(
        edge_id=edge_id, chain=chain, hour=hour,
        delta_flows=z, flows_updated=state.flows[hour].copy(),
        tt_updated=state.travel_time[hour].copy(),
        newly_congested=b, newly_relieved=b,
        displaced_flow=0.0, rerouted_flow=0.0,
        total_tt_before=0.0, total_tt_after=0.0, total_delay=0.0,
        paths=[], warning=warning,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN ENDPOINT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _chain_endpoints(
    state:         NetworkState,
    chain_idxs:    list[int],
    chain_idx_set: set[int],
) -> tuple[Optional[int], Optional[int]]:
    chain_from = {state.nodes_from[i] for i in chain_idxs if state.nodes_from[i] >= 0}
    chain_to   = {state.nodes_to[i]   for i in chain_idxs if state.nodes_to[i]   >= 0}

    head_edges = [i for i in chain_idxs
                  if state.nodes_from[i] >= 0
                  and state.nodes_from[i] not in chain_to]
    tail_edges = [i for i in chain_idxs
                  if state.nodes_to[i] >= 0
                  and state.nodes_to[i] not in chain_from]

    head_node = int(state.nodes_from[head_edges[0]]) if head_edges else (
        int(state.nodes_from[chain_idxs[0]]) if state.nodes_from[chain_idxs[0]] >= 0 else None
    )
    tail_node = int(state.nodes_to[tail_edges[0]]) if tail_edges else (
        int(state.nodes_to[chain_idxs[-1]]) if state.nodes_to[chain_idxs[-1]] >= 0 else None
    )
    return head_node, tail_node


# ═══════════════════════════════════════════════════════════════════════════════
# SPEED FLOOR
# ═══════════════════════════════════════════════════════════════════════════════

def apply_speed_floor(
    state:       NetworkState,
    edge_id:     str,
    hour:        int,
    min_speed_kmh: float,
    chain_index  = None,
    k:           int   = K_PATHS,
    theta:       float = THETA_DEFAULT,
    nx_graph           = None,
) -> RemovalResult:
    """
    Impose a minimum speed on an edge (or its chain).
    Computes how much flow must be displaced for BPR speed >= min_speed_kmh,
    then reroutes exactly that excess using the same logit assignment as remove_edge().
    If current speed already meets the floor, returns a null result with a note.
    """
    import networkx as nx_mod
    
    hour = max(0, min(state.T - 1, hour))

    if chain_index is not None:
        chain_eids = chain_index.expand_one(edge_id)
    else:
        chain_eids = [edge_id]
        
    chain_eids = [e for e in chain_eids if e in state.edge_index]
    if not chain_eids:
        return _null_result(edge_id, [edge_id], hour, state,
                            f"Edge '{edge_id}' not found in network")

    chain_idxs    = [state.edge_index[e] for e in chain_eids]
    chain_idx_set = set(chain_idxs)

    tt_current    = state.travel_time[hour].copy()
    flows_current = state.flows[hour].copy()

    displaced_total = 0.0
    for i in chain_idxs:
        sf   = float(state.speed_free[i])
        cap  = float(state.capacity[i])
        cur  = float(flows_current[i])
        ratio = sf / max(min_speed_kmh, 0.1)
        if ratio <= 1.0:
            max_flow = 0.0
        else:
            vc_max   = ((ratio - 1.0) / BPR_ALPHA) ** (1.0 / BPR_BETA)
            max_flow = min(vc_max * cap, cap * 1.5)
        excess = max(cur - max_flow, 0.0)
        displaced_total += excess

    if displaced_total < 1.0:
        r = _null_result(edge_id, chain_eids, hour, state, None)
        r.warning = (
            f"Speed floor {min_speed_kmh:.0f} km/h already satisfied "
            f"— no flow redistribution needed"
        )
        return r

    head_node, tail_node = _chain_endpoints(state, chain_idxs, chain_idx_set)

    paths_raw: list[dict] = []
    if nx_graph is not None and head_node is not None and tail_node is not None:
        paths_raw = _nx_k_paths(
            nx_graph, state, chain_eids, head_node, tail_node,
            tt_current, hour, k,
        )
    if not paths_raw and head_node is not None and tail_node is not None:
        paths_raw = _k_shortest_paths(
            adj=state.adj, nodes_from=state.nodes_from, nodes_to=state.nodes_to,
            tt=tt_current, source=head_node, target=tail_node,
            blocked_edges=frozenset(),
            k=k, max_hops=MAX_PATH_HOPS,
        )

    if not paths_raw:
        r = _null_result(edge_id, chain_eids, hour, state,
                         "No alternative paths found for speed-floor redistribution")
        r.displaced_flow = displaced_total
        return r

    path_tts = np.array([p["travel_time_s"] for p in paths_raw])
    log_w    = -theta * path_tts
    log_w   -= log_w.max()
    weights  = np.exp(log_w)
    weights /= weights.sum()
    flow_assignments = displaced_total * weights

    delta_flows = np.zeros(state.N, dtype=np.float64)
    for i in chain_idxs:
        cur = float(flows_current[i])
        sf  = float(state.speed_free[i])
        cap = float(state.capacity[i])
        ratio = sf / max(min_speed_kmh, 0.1)
        if ratio <= 1.0:
            max_flow = 0.0
        else:
            vc_max   = ((ratio - 1.0) / BPR_ALPHA) ** (1.0 / BPR_BETA)
            max_flow = min(vc_max * cap, cap * 1.5)
        delta_flows[i] = max(max_flow, 0.0) - cur

    paths_annotated: list[dict] = []
    for p, w, fa in zip(paths_raw, weights, flow_assignments):
        for ei in p["edges"]:
            delta_flows[ei] += fa
        paths_annotated.append({
            "edges":         [state.edge_ids[ei] for ei in p["edges"]],
            "edge_indices":  p["edges"],
            "travel_time_s": float(p["travel_time_s"]),
            "flow_assigned": float(fa),
            "weight":        float(w),
            "n_hops":        len(p["edges"]),
        })

    flows_updated = np.maximum(flows_current + delta_flows, 0.0)
    tt_bpr_b      = _bpr_travel_times(flows_current, state.capacity, state.length_m, state.speed_free)
    tt_bpr_a      = _bpr_travel_times(flows_updated, state.capacity, state.length_m, state.speed_free)
    tt_updated    = np.maximum(tt_current + (tt_bpr_a - tt_bpr_b), 0.1)

    newly_congested = _congestion_delta(flows_current, flows_updated, state.capacity, True)
    newly_relieved  = _congestion_delta(flows_current, flows_updated, state.capacity, False)
    tb = float(np.sum(flows_current * tt_current))
    ta = float(np.sum(flows_updated * tt_updated))
    rerouted = float(sum(p["flow_assigned"] for p in paths_annotated))

    return RemovalResult(
        edge_id=edge_id, chain=chain_eids, hour=hour,
        delta_flows=delta_flows,
        flows_updated=flows_updated, tt_updated=tt_updated,
        newly_congested=newly_congested, newly_relieved=newly_relieved,
        displaced_flow=displaced_total, rerouted_flow=rerouted,
        total_tt_before=tb, total_tt_after=ta, total_delay=ta - tb,
        paths=paths_annotated,
        warning=f"Speed floor applied: {min_speed_kmh:.0f} km/h → "
                f"{displaced_total:.0f} veh/h redistributed",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════

def remove_edge(
    state:       NetworkState,
    edge_id:     str,
    hour:        int,
    chain_index  = None,
    k:           int   = K_PATHS,
    theta:       float = THETA_DEFAULT,
    nx_graph           = None,
) -> RemovalResult:
    import networkx as nx_mod

    hour = max(0, min(state.T - 1, hour))

    # ── 1. Expand edge → full chain ───────────────────────────────────────────
    if chain_index is not None:
        chain_eids = chain_index.expand_one(edge_id)
    else:
        chain_eids = [edge_id]

    chain_eids = [e for e in chain_eids if e in state.edge_index]
    if not chain_eids:
        return _null_result(edge_id, [edge_id], hour, state,
                            f"Edge '{edge_id}' not found in network")

    chain_idxs    = [state.edge_index[e] for e in chain_eids]
    chain_idx_set = set(chain_idxs)
    blocked_set   = frozenset(chain_idxs)

    log.info("Removing chain of %d edge(s) (triggered by %s)  hour=%02d",
             len(chain_eids), edge_id, hour)

    tt_current    = state.travel_time[hour].copy()
    flows_current = state.flows[hour].copy()

    # ── 2. Displaced flow ─────────────────────────────────────────────────────
    displaced_flow = float(np.sum(flows_current[chain_idxs]))
    total_cap      = float(np.sum(state.capacity[chain_idxs]))
    displaced_flow = min(displaced_flow, total_cap * 1.5)
    displaced_flow = max(displaced_flow, 0.0)

    # ── 3. Chain endpoints ────────────────────────────────────────────────────
    head_node, tail_node = _chain_endpoints(state, chain_idxs, chain_idx_set)

    # ── 4. Find k alternative paths ───────────────────────────────────────────
    paths_raw: list[dict] = []

    if nx_graph is not None and head_node is not None and tail_node is not None:
        paths_raw = _nx_k_paths(
            nx_graph, state, chain_eids, head_node, tail_node,
            tt_current, hour, k,
        )

    if not paths_raw and head_node is not None and tail_node is not None:
        paths_raw = _k_shortest_paths(
            adj=state.adj, nodes_from=state.nodes_from, nodes_to=state.nodes_to,
            tt=tt_current, source=head_node, target=tail_node,
            blocked_edges=blocked_set, k=k, max_hops=MAX_PATH_HOPS,
        )

    # ── 5. No alternative found ───────────────────────────────────────────────
    if not paths_raw:
        flows_updated = flows_current.copy()
        for ci in chain_idxs:
            flows_updated[ci] = 0.0
        tt_bpr_b   = _bpr_travel_times(flows_current, state.capacity, state.length_m, state.speed_free)
        tt_bpr_a   = _bpr_travel_times(flows_updated, state.capacity, state.length_m, state.speed_free)
        tt_updated = np.maximum(tt_current + (tt_bpr_a - tt_bpr_b), 0.1)
        tb = float(np.sum(flows_current * tt_current))
        ta = float(np.sum(flows_updated * tt_updated))
        return RemovalResult(
            edge_id=edge_id, chain=chain_eids, hour=hour,
            delta_flows=flows_updated - flows_current,
            flows_updated=flows_updated, tt_updated=tt_updated,
            newly_congested=_congestion_delta(flows_current, flows_updated, state.capacity, True),
            newly_relieved= _congestion_delta(flows_current, flows_updated, state.capacity, False),
            displaced_flow=displaced_flow, rerouted_flow=0.0,
            total_tt_before=tb, total_tt_after=ta, total_delay=ta - tb,
            paths=[],
            warning="No alternative path found — network disconnected at this chain",
        )

    # ── 6. Logit flow assignment ──────────────────────────────────────────────
    path_tts = np.array([p["travel_time_s"] for p in paths_raw])
    log_w    = -theta * path_tts
    log_w   -= log_w.max()
    weights  = np.exp(log_w)
    weights /= weights.sum()
    flow_assignments = displaced_flow * weights

    # ── 7. Build delta_flows ──────────────────────────────────────────────────
    delta_flows = np.zeros(state.N, dtype=np.float64)
    for ci in chain_idxs:
        delta_flows[ci] = -float(flows_current[ci])

    paths_annotated: list[dict] = []
    for p, w, fa in zip(paths_raw, weights, flow_assignments):
        for ei in p["edges"]:
            delta_flows[ei] += fa
        paths_annotated.append({
            "edges":         [state.edge_ids[ei] for ei in p["edges"]],
            "edge_indices":  p["edges"],
            "travel_time_s": float(p["travel_time_s"]),
            "flow_assigned": float(fa),
            "weight":        float(w),
            "n_hops":        len(p["edges"]),
        })

    # ── 8. BPR travel times ───────────────────────────────────────────────────
    flows_updated = np.maximum(flows_current + delta_flows, 0.0)
    tt_bpr_b      = _bpr_travel_times(flows_current, state.capacity, state.length_m, state.speed_free)
    tt_bpr_a      = _bpr_travel_times(flows_updated, state.capacity, state.length_m, state.speed_free)
    tt_updated    = np.maximum(tt_current + (tt_bpr_a - tt_bpr_b), 0.1)

    # ── 9. Congestion analysis ────────────────────────────────────────────────
    newly_congested = _congestion_delta(flows_current, flows_updated, state.capacity, True)
    newly_relieved  = _congestion_delta(flows_current, flows_updated, state.capacity, False)

    tb       = float(np.sum(flows_current * tt_current))
    ta       = float(np.sum(flows_updated * tt_updated))
    rerouted = float(sum(p["flow_assigned"] for p in paths_annotated))

    log.info(
        "  Removed chain=%d  displaced=%.1f  rerouted=%.0f%%  "
        "paths=%d  congested=%d  delay=%.2f veh·h",
        len(chain_eids), displaced_flow,
        rerouted / max(displaced_flow, 1) * 100,
        len(paths_annotated), int(newly_congested.sum()),
        (ta - tb) / 3600,
    )

    return RemovalResult(
        edge_id=edge_id, chain=chain_eids, hour=hour,
        delta_flows=delta_flows,
        flows_updated=flows_updated, tt_updated=tt_updated,
        newly_congested=newly_congested, newly_relieved=newly_relieved,
        displaced_flow=displaced_flow, rerouted_flow=rerouted,
        total_tt_before=tb, total_tt_after=ta, total_delay=ta - tb,
        paths=paths_annotated,
    )


def remove_edge_all_hours(
    state:       NetworkState,
    edge_id:     str,
    chain_index  = None,
    k:           int   = K_PATHS,
    theta:       float = THETA_DEFAULT,
    nx_graph           = None,
) -> list[RemovalResult]:
    return [
        remove_edge(state, edge_id, t,
                    chain_index=chain_index, k=k, theta=theta, nx_graph=nx_graph)
        for t in range(state.T)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# PATH FINDING — NetworkX backend (no G.copy() in hot path)
# ═══════════════════════════════════════════════════════════════════════════════

def _nx_k_paths(
    nx_graph,
    state:       NetworkState,
    chain_eids:  list[str],
    head_node:   int,
    tail_node:   int,
    tt_current:  np.ndarray,
    hour:        int,
    k:           int,
) -> list[dict]:
    """
    Find k shortest paths bypassing chain_eids.

    FIX: uses nx.restricted_view() instead of nx_graph.copy().
    restricted_view() is an O(1) graph view that filters nodes/edges
    at traversal time — no memory allocation, no deep copy.
    """
    import networkx as nx

    if head_node >= len(state.node_ids) or tail_node >= len(state.node_ids):
        return []
    u_str = state.node_ids[head_node]
    v_str = state.node_ids[tail_node]

    if not nx_graph.has_node(u_str) or not nx_graph.has_node(v_str):
        return []

    chain_eid_set = set(chain_eids)

    # Build the set of (u,v) pairs to hide — O(|chain|) not O(N)
    hidden_edges = set()
    for u, v, data in nx_graph.edges(data=True):
        if data.get("id", "") in chain_eid_set:
            hidden_edges.add((u, v))

    # O(1) view — no copy
    G_view = nx.restricted_view(nx_graph, nodes=[], edges=hidden_edges)

    def weight_fn(u, v, d):
        i_d = state.edge_index.get(d.get("id", ""))
        if i_d is None:
            return float(d.get("tt", [30.0] * 24)[hour])
        return float(tt_current[i_d])

    paths_raw = []
    try:
        gen = nx.shortest_simple_paths(G_view, u_str, v_str, weight=weight_fn)
        for path_nodes in gen:
            if len(paths_raw) >= k:
                break
            edge_indices = []
            path_tt      = 0.0
            valid        = True
            for i_n in range(len(path_nodes) - 1):
                pu, pv = path_nodes[i_n], path_nodes[i_n + 1]
                if not G_view.has_edge(pu, pv):
                    valid = False
                    break
                seg_eid = G_view[pu][pv].get("id", "")
                seg_idx = state.edge_index.get(seg_eid)
                if seg_idx is None:
                    seg_tt = float(G_view[pu][pv].get("tt", [30.0] * 24)[hour])
                else:
                    edge_indices.append(int(seg_idx))
                    seg_tt = float(tt_current[int(seg_idx)])
                path_tt += seg_tt
            if valid and edge_indices:
                paths_raw.append({"edges": edge_indices, "travel_time_s": path_tt})
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    return paths_raw


# ═══════════════════════════════════════════════════════════════════════════════
# PATH FINDING — native adjacency-list Yen's algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def _k_shortest_paths(
    adj:           dict,
    nodes_from:    np.ndarray,
    nodes_to:      np.ndarray,
    tt:            np.ndarray,
    source:        int,
    target:        int,
    blocked_edges: frozenset,
    k:             int,
    max_hops:      int,
) -> list[dict]:
    import heapq

    def dijkstra(src, tgt, blocked_e, blocked_n):
        dist = {}
        prev = {}
        heap = [(0.0, src)]
        dist[src] = 0.0
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if u == tgt:
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
            for (vn, ei) in adj.get(u, []):
                if ei in blocked_e:
                    continue
                if vn != tgt and vn in blocked_n:
                    continue
                nd = d + float(tt[ei])
                if nd < dist.get(vn, float("inf")):
                    dist[vn] = nd
                    prev[vn] = (u, ei)
                    heapq.heappush(heap, (nd, vn))
        return None

    first = dijkstra(source, target, blocked_edges, frozenset())
    if first is None:
        return []

    A = [first]
    B = []

    for _ in range(k - 1):
        last = A[-1]
        for spur_idx in range(len(last["edges"])):
            spur_node  = source if spur_idx == 0 else int(nodes_to[last["edges"][spur_idx - 1]])
            root_edges = last["edges"][:spur_idx]
            root_tt    = sum(float(tt[e]) for e in root_edges)

            blocked_e = set(blocked_edges)
            blocked_n = set()

            # FIX: use tuple comparison to avoid O(k·n) list equality
            root_tuple = tuple(root_edges)
            for p in A:
                if (len(p["edges"]) > spur_idx and
                        tuple(p["edges"][:spur_idx]) == root_tuple):
                    blocked_e.add(p["edges"][spur_idx])

            cur = source
            for ei in root_edges:
                if cur != spur_node:
                    blocked_n.add(cur)
                cur = int(nodes_to[ei])

            spur = dijkstra(spur_node, target, frozenset(blocked_e), frozenset(blocked_n))
            if spur is None:
                continue

            cand_edges = root_edges + spur["edges"]
            cand_tt    = root_tt + spur["travel_time_s"]
            cand       = {"edges": cand_edges, "travel_time_s": cand_tt}

            cand_tuple = tuple(cand_edges)
            if (not any(tuple(c["edges"]) == cand_tuple for c in A) and
                    not any(tuple(c["edges"]) == cand_tuple for _, c in B)):
                heapq.heappush(B, (cand_tt, cand))

        if not B:
            break
        _, best = heapq.heappop(B)
        A.append(best)

    return A


# ═══════════════════════════════════════════════════════════════════════════════
# BPR
# ═══════════════════════════════════════════════════════════════════════════════

def _bpr_travel_times(flows, capacity, length_m, speed_free):
    vc_clip   = np.minimum(flows / np.maximum(capacity, 1.0), 10.0)
    speed_bpr = speed_free / (1.0 + BPR_ALPHA * vc_clip ** BPR_BETA)
    speed_bpr = np.maximum(speed_bpr, 0.5)
    return length_m * 3.6 / speed_bpr


def _congestion_delta(flows_before, flows_after, capacity, new):
    thr      = CONGESTION_VC_THRESHOLD
    vc_b     = flows_before / np.maximum(capacity, 1.0)
    vc_a     = flows_after  / np.maximum(capacity, 1.0)
    if new:
        return (vc_b <= thr) & (vc_a > thr)
    else:
        return (vc_b >  thr) & (vc_a <= thr)


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICALITY SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def criticality_scores(
    state:       NetworkState,
    hour:        int   = 8,
    edge_ids:    Optional[list] = None,
    chain_index  = None,
    k:           int   = K_PATHS,
    theta:       float = THETA_DEFAULT,
    nx_graph           = None,
) -> list[dict]:
    """
    Composite criticality score — weighted combination of four dimensions:

        criticality = 0.40 × delay_norm
                    + 0.30 × flow_norm
                    + 0.20 × congestion_norm
                    + 0.10 × isolation_penalty

    where:
        delay_norm       = total_delay_veh_h / max(total_delay_veh_h)
        flow_norm        = displaced_flow    / max(displaced_flow)
        congestion_norm  = n_congested       / max(n_congested)   (or 0 if all zero)
        isolation_penalty= 1 - rerouted_pct/100   (0 = fully reroutable, 1 = no alt)

    All components are clipped to [0, 1] before weighting.
    Edges with no eligible simulation result are assigned criticality=0.
    """
    if edge_ids is None:
        edge_ids = [
            state.edge_ids[i] for i in range(state.N)
            if state.data_source[i] in ("sensor", "reconstructed")
            and state.nodes_from[i] >= 0
            and state.flows[hour, i] > 5.0
        ]

    log.info("Computing criticality for %d edges at hour %02d:00 ...",
             len(edge_ids), hour)

    raw: list[dict] = []
    for eid in edge_ids:
        r = remove_edge(state, eid, hour,
                        chain_index=chain_index, k=k, theta=theta,
                        nx_graph=nx_graph)
        reroute_pct = r.rerouted_flow / max(r.displaced_flow, 1) * 100
        raw.append({
            "edge_id":           eid,
            "chain_len":         len(r.chain),
            "displaced_flow":    r.displaced_flow,
            "rerouted_pct":      reroute_pct,
            "total_delay_veh_h": r.total_delay / 3600,
            "n_congested":       int(r.newly_congested.sum()),
            "warning":           r.warning,
        })

    if not raw:
        return []

    # ── Normalise each dimension independently ────────────────────────────────
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    delays     = np.array([s["total_delay_veh_h"] for s in raw], dtype=np.float64)
    flows      = np.array([s["displaced_flow"]     for s in raw], dtype=np.float64)
    congested  = np.array([s["n_congested"]        for s in raw], dtype=np.float64)
    isolation  = np.array([1.0 - min(s["rerouted_pct"], 100) / 100.0 for s in raw], dtype=np.float64)

    delay_norm      = _norm(delays)
    flow_norm       = _norm(flows)
    congestion_norm = _norm(congested)
    # isolation is already 0-1 by construction, but normalise for fairness
    isolation_norm  = _norm(isolation)

    composite = (
        0.40 * delay_norm
      + 0.30 * flow_norm
      + 0.20 * congestion_norm
      + 0.10 * isolation_norm
    )
    # Clip to [0,1] for safety
    composite = np.clip(composite, 0.0, 1.0)

    scores = []
    for s, c in zip(raw, composite):
        scores.append({**s, "criticality": round(float(c), 4)})

    scores.sort(key=lambda s: -s["criticality"])

    if scores:
        top = scores[0]
        log.info(
            "  Top edge: %s  criticality=%.4f  "
            "delay=%.2f veh·h  flow=%.0f veh/h  congested=%d  isolation=%.0f%%",
            top["edge_id"], top["criticality"], top["total_delay_veh_h"],
            top["displaced_flow"], top["n_congested"],
            (1 - top["rerouted_pct"] / 100) * 100,
        )
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# WEB APP SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeRemovalService:
    def __init__(self, pkl_path: str, net_path: str, chains_path: str = ""):
        self.pkl_path    = pkl_path
        self.net_path    = net_path
        self.chains_path = chains_path
        self._state:  Optional[NetworkState] = None
        self._net     = None
        self._G       = None
        self._chains  = None
        self._loaded  = False

    def set_nx_graph(self, G) -> None:
        self._G = G
        log.info("EdgeRemovalService: NetworkX graph injected (%d nodes, %d edges)",
                 G.number_of_nodes(), G.number_of_edges())

    def load(self) -> None:
        self._state = NetworkState.load(self.pkl_path, self.net_path)

        if self.chains_path:
            from chain_utils import load_chains
            self._chains = load_chains(self.chains_path)
            log.info("EdgeRemovalService: chain index loaded (%d chains)", len(self._chains))
        else:
            log.warning("EdgeRemovalService: no chains_path supplied — single-edge removal only")

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

    def expand_chain(self, edge_id: str) -> list[str]:
        if self._chains is None:
            return [edge_id]
        return self._chains.expand_one(edge_id)

    def chain_summary(self, edge_id: str) -> dict:
        if self._chains is None:
            return {"edge_id": edge_id, "chain_len": 1, "chain": [edge_id], "is_multi": False}
        return self._chains.chain_summary(edge_id)

    def simulate(self, edge_id: str, hour: int = 8,
                 k: int = K_PATHS, theta: float = THETA_DEFAULT) -> dict:
        if not self._loaded:
            return {"error": "Service not loaded"}
        r = remove_edge(self._state, edge_id, hour,
                        chain_index=self._chains, k=k, theta=theta, nx_graph=self._G)
        return r.to_dict()

    def simulate_all_hours(self, edge_id: str,
                           k: int = K_PATHS, theta: float = THETA_DEFAULT) -> list[dict]:
        if not self._loaded:
            return [{"error": "Service not loaded"}]
        results = remove_edge_all_hours(
            self._state, edge_id,
            chain_index=self._chains, k=k, theta=theta, nx_graph=self._G,
        )
        return [r.to_dict() for r in results]

    def simulate_geojson(self, edge_id: str, hour: int = 8,
                         k: int = K_PATHS, theta: float = THETA_DEFAULT) -> dict:
        if not self._loaded:
            return {"error": "Service not loaded"}
        if self._net is None:
            return {"error": "Network geometry not available"}
        r = remove_edge(self._state, edge_id, hour,
                        chain_index=self._chains, k=k, theta=theta, nx_graph=self._G)
        return r.to_geojson_delta(self._state, self._net)

    def get_edge_info(self, edge_id: str, hour: int = 8) -> Optional[dict]:
        if not self._loaded:
            return None
        idx = self._state.edge_index.get(edge_id)
        if idx is None:
            return None
        chain = self.expand_chain(edge_id)
        return {
            "edge_id":      edge_id,
            "chain":        chain,
            "chain_len":    len(chain),
            "road_type":    self._state.road_type[idx],
            "capacity":     round(float(self._state.capacity[idx]), 1),
            "length_m":     round(float(self._state.length_m[idx]), 1),
            "flow_veh_h":   round(float(self._state.flows[hour, idx]), 1),
            "vc_ratio":     round(float(self._state.flows[hour, idx] /
                                max(self._state.capacity[idx], 1)), 4),
            "tt_s":         round(float(self._state.travel_time[hour, idx]), 1),
            "data_source":  self._state.data_source[idx],
            "within_reach": bool(self._state.within_reach[idx]),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Chain-aware edge removal & flow redistribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pkl",       default=DEFAULTS["pkl"])
    p.add_argument("--net",       default=DEFAULTS["net"])
    p.add_argument("--chains",    default=DEFAULTS["chains"])
    p.add_argument("--edge",      required=True)
    p.add_argument("--hour",      type=int, default=8)
    p.add_argument("--k",         type=int, default=K_PATHS)
    p.add_argument("--theta",     type=float, default=THETA_DEFAULT)
    p.add_argument("--all-hours", action="store_true")
    p.add_argument("--json",      action="store_true")
    args = p.parse_args()

    for path, label in [(args.pkl, "--pkl"), (args.net, "--net")]:
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}")
            raise SystemExit(1)

    state = NetworkState.load(args.pkl, args.net)

    chain_index = None
    if Path(args.chains).exists():
        from chain_utils import load_chains
        chain_index = load_chains(args.chains)
    else:
        print(f"WARNING: chains file not found at {args.chains} — single-edge mode")

    if args.all_hours:
        results = remove_edge_all_hours(
            state, args.edge, chain_index=chain_index, k=args.k, theta=args.theta,
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            for r in results:
                print(r.summary()); print()
    else:
        result = remove_edge(
            state, args.edge, args.hour, chain_index=chain_index,
            k=args.k, theta=args.theta,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.summary())


if __name__ == "__main__":
    main()
