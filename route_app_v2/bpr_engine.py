"""
bpr_engine.py — BPR model + chain-aware edge removal.

Adapted from edge_removal.py with one key change:
  NetworkState.load_from_cache() builds topology from edge_lookup (no sumolib).

All BPR math, Yen's k-paths on chain graph, and RemovalResult are unchanged.
"""
from __future__ import annotations

import logging
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("bpr_engine")

# ── BPR parameters ────────────────────────────────────────────────────────────
BPR_ALPHA = 0.15
BPR_BETA  = 4.0
THETA_DEFAULT           = 0.05
K_PATHS                 = 3
MAX_PATH_HOPS           = 60
CONGESTION_VC_THRESHOLD = 0.85


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NetworkState:
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
    def load_from_cache(cls, pkl_path: str, edge_lookup: dict) -> "NetworkState":
        """
        Load arrays.pkl and build topology from edge_lookup — no sumolib needed.

        This is the primary entry point for v2.  The topology (nodes_from,
        nodes_to, road_type, length_m) is derived from edge_lookup which was
        already parsed from the GeoJSON at startup.  This eliminates the
        3–5 minute sumolib XML parse that dominated v1 startup time.
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
            conf_obs = arrays.get("conf_observed", np.zeros(N))
            data_source = [
                "sensor"        if conf_obs[i] > 0
                else "reconstructed" if within_reach[i]
                else "prior"
                for i in range(N)
            ]

        # Build topology from edge_lookup (O(N), no XML parse)
        adj, nodes_from, nodes_to, node_ids, node_index, road_type, length_m = \
            _topology_from_lookup(edge_lookup, edge_ids, edge_index, N)

        state = cls(
            edge_ids=edge_ids, edge_index=edge_index,
            flows=flows, travel_time=tt, capacity=capacity,
            length_m=length_m, speed_free=speed_free,
            within_reach=within_reach, data_source=data_source,
            road_type=road_type, T=T, N=N,
            adj=adj, nodes_from=nodes_from, nodes_to=nodes_to,
            node_ids=node_ids, node_index=node_index,
        )
        log.info("NetworkState ready: N=%d edges, T=%d hours, %d nodes (%.1fs)",
                 N, T, len(node_ids), time.time() - t0)
        return state


def _topology_from_lookup(
    edge_lookup: dict,
    edge_ids: list,
    edge_index: dict,
    N: int,
) -> tuple:
    """Build topology arrays from edge_lookup (no sumolib required)."""
    node_ids:   list = []
    node_index: dict = {}

    def _vid(nid: str) -> int:
        if nid not in node_index:
            node_index[nid] = len(node_ids)
            node_ids.append(nid)
        return node_index[nid]

    nodes_from = np.full(N, -1, dtype=np.int32)
    nodes_to   = np.full(N, -1, dtype=np.int32)
    road_type  = ["unknown"] * N
    length_m   = np.ones(N, dtype=np.float64)
    adj: dict  = {}

    for eid in edge_ids:
        i = edge_index.get(eid)
        if i is None:
            continue
        info = edge_lookup.get(eid)
        if not info:
            continue
        u_str = info.get("u", "")
        v_str = info.get("v", "")
        if not u_str or not v_str:
            continue
        u = _vid(u_str)
        v = _vid(v_str)
        nodes_from[i] = u
        nodes_to[i]   = v
        road_type[i]  = info.get("road_type", "unclassified")
        length_m[i]   = float(info.get("length", 1.0))
        adj.setdefault(u, []).append((v, i))

    n_connected = int((nodes_from >= 0).sum())
    log.info("Topology from cache: %d nodes, %d/%d edges connected",
             len(node_ids), n_connected, N)
    return adj, nodes_from, nodes_to, node_ids, node_index, road_type, length_m


# ══════════════════════════════════════════════════════════════════════════════
# REMOVAL RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RemovalResult:
    edge_id:         str
    chain:           list
    hour:            int
    delta_flows:     np.ndarray
    flows_updated:   np.ndarray
    tt_updated:      np.ndarray
    newly_congested: np.ndarray
    newly_relieved:  np.ndarray
    displaced_flow:  float = 0.0
    rerouted_flow:   float = 0.0
    total_delay:     float = 0.0
    warning:         str   = ""
    paths:           list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "edge_id":           self.edge_id,
            "chain":             self.chain,
            "chain_len":         len(self.chain),
            "hour":              self.hour,
            "displaced_flow":    round(float(self.displaced_flow), 2),
            "rerouted_flow":     round(float(self.rerouted_flow),  2),
            "reroute_pct":       round(self.rerouted_flow / max(self.displaced_flow, 1) * 100, 1),
            "total_delay_veh_h": round(float(self.total_delay) / 3600, 3),
            "n_newly_congested": int(self.newly_congested.sum()),
            "n_newly_relieved":  int(self.newly_relieved.sum()),
            "warning":           self.warning,
            "paths":             [{"edges": p["edges"], "tt": round(p["tt"], 2)} for p in self.paths],
            "tt_updated_raw":    [1e15 if math.isinf(v) else v for v in self.tt_updated.tolist()],
        }

    def to_geojson_delta(self, state: NetworkState, edge_lookup: dict) -> dict:
        """GeoJSON of changed edges — uses edge_lookup geometry (no sumolib)."""
        changed = np.where(self.delta_flows != 0)[0]
        features = []
        for idx in changed:
            eid  = state.edge_ids[idx]
            info = edge_lookup.get(eid, {})
            geom = info.get("geom", [])
            if not geom:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": geom},
                "properties": {
                    "edge_id":    eid,
                    "delta_flow": round(float(self.delta_flows[idx]), 2),
                    "flow_before": round(float(state.flows[self.hour, idx]), 2),
                    "flow_after":  round(float(self.flows_updated[idx]), 2),
                    "new_tt_s":   round(float(self.tt_updated[idx]) if not math.isinf(self.tt_updated[idx]) else 1e15, 2),
                    "congested":  bool(self.newly_congested[idx]),
                },
            })
        return {"type": "FeatureCollection", "features": features}


# ══════════════════════════════════════════════════════════════════════════════
# BPR CORE
# ══════════════════════════════════════════════════════════════════════════════

def _bpr_tt(flows: np.ndarray, capacity: np.ndarray, tt_free: np.ndarray) -> np.ndarray:
    vc = flows / np.maximum(capacity, 1.0)
    return tt_free * (1.0 + BPR_ALPHA * vc ** BPR_BETA)


def _tt_free(state: NetworkState) -> np.ndarray:
    return state.length_m * 3.6 / np.maximum(state.speed_free, 0.5)


def _build_accumulated_flows(state: NetworkState, session_tt: dict, hour: int) -> np.ndarray:
    """Reconstruct flow array from BPR-inverted session travel times."""
    flows    = state.flows[hour].copy()
    tt_free  = _tt_free(state)
    for idx, tt_val in session_tt.items():
        if idx >= state.N:
            continue
        ttf   = float(tt_free[idx])
        ratio = float(tt_val) / max(ttf, 0.01)
        if ratio <= 1.0:
            flows[idx] = 0.0
        else:
            vc = ((ratio - 1.0) / BPR_ALPHA) ** (1.0 / BPR_BETA)
            flows[idx] = min(vc * float(state.capacity[idx]),
                             float(state.capacity[idx]) * 2.0)
    return flows


# ══════════════════════════════════════════════════════════════════════════════
# CHAIN-AWARE K-PATHS (NetworkX on small chain graph — acceptable)
# ══════════════════════════════════════════════════════════════════════════════

def _chain_k_paths(
    chain_graph,
    state: NetworkState,
    blocked_eids: list,
    head_node: str,
    tail_node: str,
    tt_current: np.ndarray,
    k: int,
    prebuilt_chain_dg=None,
) -> list[dict]:
    import networkx as nx

    if prebuilt_chain_dg is not None:
        DG = chain_graph.block_chains_in(prebuilt_chain_dg, blocked_eids)
    else:
        DG = chain_graph.as_digraph(blocked_eids, tt_current)

    if head_node not in DG or tail_node not in DG:
        return []
    if not nx.has_path(DG, head_node, tail_node):
        return []

    paths = []
    try:
        for path_nodes in nx.shortest_simple_paths(DG, head_node, tail_node, weight="tt"):
            edges = []
            total_tt = 0.0
            ok = True
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                if not DG.has_edge(u, v):
                    ok = False
                    break
                data     = DG[u][v]
                idxs     = data.get("chain_idxs", [])
                edge_tt  = sum(float(tt_current[j]) for j in idxs) if idxs else 30.0
                total_tt += edge_tt
                edges.extend(state.edge_ids[j] for j in idxs)
            if ok and len(edges) < MAX_PATH_HOPS:
                paths.append({"edges": edges, "tt": total_tt})
            if len(paths) >= k:
                break
    except Exception as exc:
        log.debug("k-paths search failed: %s", exc)

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# LOGIT FLOW ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def _logit_assign(
    paths: list[dict],
    displaced_flow: float,
    theta: float,
) -> np.ndarray:
    """Multinomial logit over paths. Returns flow delta array (sparse).

    theta is treated as a dimensionless sensitivity: disutility per unit of
    reference travel time, not per raw second.  This keeps behaviour consistent
    regardless of network scale (short urban hops vs long arterials).
    """
    if not paths:
        return np.array([], dtype=np.float64)
    min_tt  = min(p["tt"] for p in paths)
    ref_tt  = max(min_tt, 1.0)   # normalise so theta is scale-independent
    weights = [np.exp(-theta * (p["tt"] - min_tt) / ref_tt) for p in paths]
    total_w = sum(weights)
    shares  = [w / total_w for w in weights]
    flows   = [s * displaced_flow for s in shares]
    return list(zip(paths, flows))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def remove_edge(
    state: NetworkState,
    edge_id: str,
    hour: int,
    chain_index=None,
    chain_graph=None,
    flows_override: np.ndarray = None,
    k: int = K_PATHS,
    theta: float = THETA_DEFAULT,
    prebuilt_chain_dg=None,
    **_,
) -> RemovalResult:
    """Remove edge (chain) and reroute displaced flow via BPR logit assignment."""
    idx = state.edge_index.get(edge_id)
    if idx is None:
        raise ValueError(f"Edge not found: {edge_id}")

    chain_eids = chain_index.expand_one(edge_id) if chain_index else [edge_id]
    chain_idxs = [state.edge_index[e] for e in chain_eids if e in state.edge_index]

    base_flows = flows_override if flows_override is not None else state.flows[hour].copy()
    tt_current = state.travel_time[hour].copy()
    tt_free_   = _tt_free(state)

    displaced_flow = float(base_flows[chain_idxs].sum()) if chain_idxs else 0.0

    # Find head/tail nodes for k-path search
    head_idx = state.edge_index.get(chain_eids[0])
    tail_idx = state.edge_index.get(chain_eids[-1])
    if head_idx is None or tail_idx is None:
        return _empty_result(state, edge_id, chain_eids, hour, base_flows, tt_current)

    head_node = state.node_ids[state.nodes_from[head_idx]] if state.nodes_from[head_idx] >= 0 else None
    tail_node = state.node_ids[state.nodes_to[tail_idx]]   if state.nodes_to[tail_idx]   >= 0 else None
    if head_node is None or tail_node is None:
        return _empty_result(state, edge_id, chain_eids, hour, base_flows, tt_current)

    # Block chain in flows
    flows_blocked = base_flows.copy()
    for ci in chain_idxs:
        flows_blocked[ci] = 0.0

    # Find detour paths on chain graph
    paths = []
    if chain_graph is not None:
        paths = _chain_k_paths(
            chain_graph, state, chain_eids, head_node, tail_node, tt_current, k,
            prebuilt_chain_dg=prebuilt_chain_dg,
        )

    # Assign flow to detours
    assignments = _logit_assign(paths, displaced_flow, theta)
    flows_new = flows_blocked.copy()
    rerouted  = 0.0
    for path, flow_share in (assignments or []):
        rerouted += flow_share
        for eid in path["edges"]:
            ei = state.edge_index.get(eid)
            if ei is not None:
                flows_new[ei] += flow_share

    # Recompute BPR travel times
    tt_bpr_base = _bpr_tt(base_flows,  state.capacity, tt_free_)
    tt_new      = _bpr_tt(flows_new,   state.capacity, tt_free_)
    # Keep blocked chain edges at inf travel time
    for ci in chain_idxs:
        tt_new[ci]   = float("inf")
        flows_new[ci] = 0.0

    delta_flows     = flows_new - base_flows
    newly_congested = (flows_new / np.maximum(state.capacity, 1)) > CONGESTION_VC_THRESHOLD
    was_congested   = (base_flows / np.maximum(state.capacity, 1)) > CONGESTION_VC_THRESHOLD
    newly_congested = newly_congested & ~was_congested
    newly_relieved  = (~(flows_new / np.maximum(state.capacity, 1) > CONGESTION_VC_THRESHOLD)) & was_congested

    # Total network delay change: compare BPR against BPR so model-vs-observed
    # mismatch doesn't produce a spurious baseline offset across all N edges.
    # Chain edges are excluded from "after" (inf tt_new) but included in "before".
    total_delay = float(
        np.nansum(flows_new[np.isfinite(tt_new)] * tt_new[np.isfinite(tt_new)])
        - np.nansum(base_flows * tt_bpr_base)
    )

    warning = ""
    if rerouted < displaced_flow * 0.01 and displaced_flow > 1:
        warning = "isolated"
    elif rerouted < displaced_flow * 0.5:
        warning = "partial"

    return RemovalResult(
        edge_id=edge_id, chain=chain_eids, hour=hour,
        delta_flows=delta_flows, flows_updated=flows_new, tt_updated=tt_new,
        newly_congested=newly_congested, newly_relieved=newly_relieved,
        displaced_flow=displaced_flow, rerouted_flow=rerouted,
        total_delay=total_delay, warning=warning, paths=paths,
    )


def apply_capacity_tune(
    state: NetworkState,
    edge_id: str,
    hour: int,
    capacity_factor: float = 0.5,
    chain_index=None,
    chain_graph=None,
    flows_override: np.ndarray = None,
    k: int = K_PATHS,
    theta: float = THETA_DEFAULT,
    **_,
) -> RemovalResult:
    """Reduce capacity of edge (chain) and overflow-reroute excess flow."""
    chain_eids = chain_index.expand_one(edge_id) if chain_index else [edge_id]
    chain_idxs = [state.edge_index[e] for e in chain_eids if e in state.edge_index]

    base_flows = flows_override if flows_override is not None else state.flows[hour].copy()
    tt_current = state.travel_time[hour].copy()
    tt_free_   = _tt_free(state)

    # Compute excess flow above reduced capacity
    displaced_flow = 0.0
    flows_adjusted = base_flows.copy()
    cap_adjusted   = state.capacity.copy()
    for ci in chain_idxs:
        new_cap = state.capacity[ci] * capacity_factor
        excess  = max(0.0, float(base_flows[ci]) - new_cap)
        displaced_flow    += excess
        flows_adjusted[ci] = float(base_flows[ci]) - excess
        cap_adjusted[ci]   = new_cap

    # Get head/tail nodes for detour search
    head_idx  = state.edge_index.get(chain_eids[0])
    tail_idx  = state.edge_index.get(chain_eids[-1])
    head_node = state.node_ids[state.nodes_from[head_idx]] if head_idx is not None and state.nodes_from[head_idx] >= 0 else None
    tail_node = state.node_ids[state.nodes_to[tail_idx]]   if tail_idx is not None and state.nodes_to[tail_idx]   >= 0 else None

    paths = []
    if displaced_flow > 0 and chain_graph is not None and head_node and tail_node:
        paths = _chain_k_paths(
            chain_graph, state, [], head_node, tail_node, tt_current, k,
        )

    assignments = _logit_assign(paths, displaced_flow, theta)
    flows_new = flows_adjusted.copy()
    rerouted  = 0.0
    for path, flow_share in (assignments or []):
        rerouted += flow_share
        for eid in path["edges"]:
            ei = state.edge_index.get(eid)
            if ei is not None:
                flows_new[ei] += flow_share

    tt_bpr_base = _bpr_tt(base_flows, state.capacity, tt_free_)
    tt_new      = _bpr_tt(flows_new,  cap_adjusted,   tt_free_)
    delta_flows = flows_new - base_flows

    newly_cong = (flows_new / np.maximum(state.capacity, 1)) > CONGESTION_VC_THRESHOLD
    was_cong   = (base_flows / np.maximum(state.capacity, 1)) > CONGESTION_VC_THRESHOLD

    return RemovalResult(
        edge_id=edge_id, chain=chain_eids, hour=hour,
        delta_flows=delta_flows, flows_updated=flows_new, tt_updated=tt_new,
        newly_congested=newly_cong & ~was_cong,
        newly_relieved=(~newly_cong) & was_cong,
        displaced_flow=displaced_flow, rerouted_flow=rerouted,
        total_delay=float(
            np.nansum(flows_new[np.isfinite(tt_new)] * tt_new[np.isfinite(tt_new)])
            - np.nansum(base_flows * tt_bpr_base)
        ),
        paths=paths,
    )


def apply_speed_floor(
    state: NetworkState,
    edge_id: str,
    hour: int,
    min_speed_kmh: float = 30.0,
    chain_index=None,
    chain_graph=None,
    flows_override: np.ndarray = None,
    k: int = K_PATHS,
    theta: float = THETA_DEFAULT,
    **_,
) -> RemovalResult:
    """Enforce minimum speed on edge (chain) via reduced effective capacity."""
    chain_eids = chain_index.expand_one(edge_id) if chain_index else [edge_id]
    chain_idxs = [state.edge_index[e] for e in chain_eids if e in state.edge_index]

    base_flows = flows_override if flows_override is not None else state.flows[hour].copy()
    tt_current = state.travel_time[hour].copy()
    tt_free_   = _tt_free(state)

    speed_ms    = min_speed_kmh / 3.6
    flows_adj   = base_flows.copy()
    cap_adj     = state.capacity.copy()   # unchanged — only flows are displaced
    displaced   = 0.0

    for ci in chain_idxs:
        lm     = float(state.length_m[ci])
        tt_min = lm / max(speed_ms, 0.1)
        tt_f   = float(tt_free_[ci])
        # ratio = speed_free / speed_floor; < 1 means floor > free-flow speed
        ratio  = tt_min / max(tt_f, 0.01)
        if ratio <= 1.0:
            # Floor exceeds free-flow — road can never meet the requirement;
            # displace all current flow
            excess = float(base_flows[ci])
            displaced     += excess
            flows_adj[ci]  = 0.0
        else:
            # BPR-invert: find the V/C at which BPR speed equals the floor
            vc_max = max(((ratio - 1.0) / BPR_ALPHA) ** (1.0 / BPR_BETA), 0.0)
            q_max  = vc_max * float(state.capacity[ci])
            excess = max(0.0, float(base_flows[ci]) - q_max)
            displaced     += excess
            flows_adj[ci]  = float(base_flows[ci]) - excess
        # cap_adj stays at original capacity so BPR gives tt = tt_min at q_max flow

    head_idx  = state.edge_index.get(chain_eids[0])
    tail_idx  = state.edge_index.get(chain_eids[-1])
    head_node = state.node_ids[state.nodes_from[head_idx]] if head_idx is not None and state.nodes_from[head_idx] >= 0 else None
    tail_node = state.node_ids[state.nodes_to[tail_idx]]   if tail_idx is not None and state.nodes_to[tail_idx]   >= 0 else None

    paths = []
    if displaced > 0 and chain_graph is not None and head_node and tail_node:
        paths = _chain_k_paths(chain_graph, state, [], head_node, tail_node, tt_current, k)

    assignments = _logit_assign(paths, displaced, theta)
    flows_new   = flows_adj.copy()
    rerouted    = 0.0
    for path, flow_share in (assignments or []):
        rerouted += flow_share
        for eid in path["edges"]:
            ei = state.edge_index.get(eid)
            if ei is not None:
                flows_new[ei] += flow_share

    tt_bpr_base = _bpr_tt(base_flows, state.capacity, tt_free_)
    tt_new      = _bpr_tt(flows_new,  cap_adj,        tt_free_)
    delta_flows = flows_new - base_flows

    newly_cong = (flows_new / np.maximum(state.capacity, 1)) > CONGESTION_VC_THRESHOLD
    was_cong   = (base_flows / np.maximum(state.capacity, 1)) > CONGESTION_VC_THRESHOLD

    return RemovalResult(
        edge_id=edge_id, chain=chain_eids, hour=hour,
        delta_flows=delta_flows, flows_updated=flows_new, tt_updated=tt_new,
        newly_congested=newly_cong & ~was_cong,
        newly_relieved=(~newly_cong) & was_cong,
        displaced_flow=displaced, rerouted_flow=rerouted,
        total_delay=float(
            np.nansum(flows_new[np.isfinite(tt_new)] * tt_new[np.isfinite(tt_new)])
            - np.nansum(base_flows * tt_bpr_base)
        ),
        paths=paths,
    )


def _empty_result(state, edge_id, chain_eids, hour, base_flows, tt_current):
    return RemovalResult(
        edge_id=edge_id, chain=chain_eids, hour=hour,
        delta_flows=np.zeros_like(base_flows),
        flows_updated=base_flows.copy(),
        tt_updated=tt_current.copy(),
        newly_congested=np.zeros(state.N, dtype=bool),
        newly_relieved=np.zeros(state.N, dtype=bool),
        warning="topology_incomplete",
    )


# ══════════════════════════════════════════════════════════════════════════════
# EDGE INFO
# ══════════════════════════════════════════════════════════════════════════════

def get_edge_info(state: NetworkState, edge_id: str, hour: int) -> dict | None:
    idx = state.edge_index.get(edge_id)
    if idx is None:
        return None
    tt  = float(state.travel_time[hour, idx])
    lm  = float(state.length_m[idx])
    cap = float(state.capacity[idx])
    fl  = float(state.flows[hour, idx])
    return {
        "edge_id":    edge_id,
        "road_type":  state.road_type[idx],
        "length_m":   round(lm, 1),
        "capacity":   round(cap, 1),
        "flow_veh_h": round(fl, 1),
        "vc_ratio":   round(fl / max(cap, 1), 4),
        "tt_s":       round(tt, 1),
        "speed_kmh":  round((lm / max(tt, 0.1)) * 3.6, 1),
        "data_source": state.data_source[idx],
        "within_reach": bool(state.within_reach[idx]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PER-HOUR BASE DIGRAPH CACHE (for pump_priority parallelism)
# ══════════════════════════════════════════════════════════════════════════════

import threading as _threading

_base_dg_cache: dict  = {}
_base_dg_lock         = _threading.Lock()

def get_base_dg(chain_graph, state: NetworkState, hour: int):
    if hour in _base_dg_cache:
        return _base_dg_cache[hour]
    with _base_dg_lock:
        if hour in _base_dg_cache:
            return _base_dg_cache[hour]
        try:
            dg = chain_graph.build_base_digraph(state.travel_time[hour])
            _base_dg_cache[hour] = dg
            return dg
        except Exception as exc:
            log.warning("base_dg build failed for hour %d: %s", hour, exc)
            return None
