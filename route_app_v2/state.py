"""
state.py — Application state singleton.

Loaded once at startup; shared across all requests.

Key differences from route_app.py:
  • igraph DiGraph instead of NetworkX (C-level Dijkstra, releases GIL).
  • Shapely STRtree for flood detection (O(log N), C-level GEOS).
  • Topology built from startup_cache.pkl (no sumolib XML parse at runtime).
  • Per-hour weight arrays pre-computed as numpy arrays for fast igraph calls.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import igraph as ig
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString
from shapely.strtree import STRtree

log = logging.getLogger("state")

# ── Paths (relative to City_Route_Pipeline root, one level up) ───────────────
_ROOT = Path(__file__).parent.parent   # City_Route_Pipeline/

GEOJSON_PATH    = _ROOT / "Gurugram/graph_reconstruction/traffic_state_scored.geojson"
ARRAYS_PKL      = _ROOT / "Gurugram/graph_reconstruction/gurugram_traffic_arrays.pkl"
MOVEMENTS_PKL   = _ROOT / "Gurugram/networks/movements.pkl"
CHAINS_PKL      = _ROOT / "Gurugram/graph_reconstruction/chains.pkl"
NODES_CSV       = _ROOT / "gurugram_nodes.csv"
STARTUP_CACHE   = _ROOT / "Gurugram/graph_reconstruction/startup_cache.pkl"
CRITICALITY_PKL = _ROOT / "criticality_precomputed.pkl"
GEOCODED_DIR    = _ROOT / "GEOCODED"

ROAD_PENALTY = {
    "motorway": 0.6,     "trunk": 0.7,          "trunk_link": 0.7,
    "primary": 0.85,     "secondary": 1.0,      "secondary_link": 1.0,
    "tertiary": 1.2,     "tertiary_link": 1.2,
    "unclassified": 1.4, "residential": 1.6,
}
FALLBACK_TURN_S = 5.0
_GRID_CELL      = 0.01   # degrees ≈ 1 km (fallback grid if STRtree unavailable)


class AppState:
    """All loaded data, built once at startup."""

    def __init__(self):
        # Edge data
        self.edge_lookup:    dict       = {}
        self.geojson_gz:     bytes      = b""
        self.geojson_etag:   str        = ""
        self.geojson_lite_gz:  bytes    = b""
        self.geojson_lite_etag: str     = ""

        # igraph routing graph
        self.G:              ig.Graph   = None
        self.node_to_vid:    dict       = {}   # node_str → igraph vertex id
        self.eid_to_eidx:    dict       = {}   # edge_id → igraph edge index
        self.adj_from_node:  dict       = {}   # node_str → [eid, eid, ...]

        # Pre-computed per-hour weight arrays (shape: (E,), numpy float64)
        # Indexed by igraph edge index.  Weights include road-type + confidence
        # multipliers and the fallback turn penalty.  Blocked / overridden edges
        # are handled at request time via a fast numpy copy + index update.
        self.weights_by_hour: list[np.ndarray] = []  # 24 arrays
        self.edge_eids_arr:   np.ndarray        = None  # igraph edge index → eid
        self.eid_to_eidx_arr: dict              = {}    # eid → igraph edge index

        # Flood STRtree
        self.flood_engine = None   # FloodEngine (set after edge_lookup loaded)

        # KDTree for nearest-edge
        self.kd_tree  = None
        self.kd_ids:  list[str] = []

        # BPR arrays (from arrays.pkl)
        self.bpr_state = None   # NetworkState (set in background)

        # Services (loaded in background)
        self.chains      = None   # ChainIndex
        self.chain_graph = None   # ChainGraph (NetworkX based, chain-level only)
        self.movement_router = None
        self.removal_loaded: bool = False
        self.removal_error:  str  = ""

        # Criticality cache
        self.criticality_cache: dict = {}
        self.precomputed_hours: set  = set()


def _load_startup_cache() -> dict:
    if STARTUP_CACHE.exists():
        cache_mt = STARTUP_CACHE.stat().st_mtime
        src_ok = True
        for src in (GEOJSON_PATH,):
            if src.exists() and src.stat().st_mtime > cache_mt:
                src_ok = False
                break
        if src_ok:
            print(f"Loading startup cache from {STARTUP_CACHE} ...")
            try:
                with open(STARTUP_CACHE, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                print(f"Cache load failed ({exc}), rebuilding ...")
    return _build_startup_cache()


def _build_startup_cache() -> dict:
    """
    Build edge_lookup, GeoJSON blobs, KDTree coords from GeoJSON.

    NOTE: u/v node IDs are NOT in the GeoJSON.  They are populated later
    by the original v1 startup (which uses sumolib).  If no startup_cache.pkl
    exists yet, run route_app.py once first to generate it, then start v2.
    """
    print(f"Parsing {GEOJSON_PATH} ...")
    with open(GEOJSON_PATH) as f:
        raw_data = json.load(f)

    el:        dict = {}
    kd_coords: list = []
    kd_ids:    list = []
    grid:      dict = {}

    for feat in raw_data["features"]:
        props  = feat["properties"]
        eid    = props["id"]
        coords = feat["geometry"]["coordinates"]

        tt_arr  = props.get("travel_time_s",    [1.0] * 24)
        tt_low  = props.get("travel_time_low_s",  list(tt_arr))
        tt_high = props.get("travel_time_high_s", list(tt_arr))
        if len(tt_arr)  < 24: tt_arr  += [tt_arr[-1]]  * (24 - len(tt_arr))
        if len(tt_low)  < 24: tt_low  += [tt_low[-1]]  * (24 - len(tt_low))
        if len(tt_high) < 24: tt_high += [tt_high[-1]] * (24 - len(tt_high))

        # u/v node IDs are NOT in the GeoJSON — they come from sumolib and are
        # written into the startup_cache.pkl at first build.  Leave them empty
        # here; if the cache already exists (normal case) they will be correct.
        u = props.get("from_node") or props.get("u") or ""
        v = props.get("to_node")   or props.get("v") or ""

        el[eid] = {
            "u": u, "v": v, "to_node": v,
            "geom":       coords,
            "tt":         tt_arr,
            "tt_low":     tt_low,
            "tt_high":    tt_high,
            "length":     props.get("length_m",           1.0),
            "road_type":  props.get("road_type", "unclassified"),
            "confidence": props.get("confidence",          0.1),
            "dom_dir":    props.get("dominant_direction", "unknown"),
        }
        mid = coords[len(coords) // 2]
        kd_coords.append((mid[0], mid[1]))
        kd_ids.append(eid)

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bx_min, bx_max = min(xs), max(xs)
        by_min, by_max = min(ys), max(ys)
        el[eid]["bbox"] = (bx_min, by_min, bx_max, by_max)
        for gx in range(int(bx_min / _GRID_CELL), int(bx_max / _GRID_CELL) + 1):
            for gy in range(int(by_min / _GRID_CELL), int(by_max / _GRID_CELL) + 1):
                grid.setdefault((gx, gy), []).append(eid)

    print("Building GeoJSON blobs ...")
    raw_bytes = json.dumps(raw_data, separators=(",", ":")).encode()
    etag      = '"' + hashlib.md5(raw_bytes).hexdigest() + '"'
    gz_bytes  = gzip.compress(raw_bytes, compresslevel=6)

    lite_feats = []
    for feat in raw_data["features"]:
        p       = feat["properties"]
        tt_lo   = p.get("travel_time_low_s",  []) or []
        tt_hi   = p.get("travel_time_high_s", []) or []
        spread_noon = round(tt_hi[12] - tt_lo[12], 1) if len(tt_hi) > 12 and len(tt_lo) > 12 else 0.0
        lite_feats.append({
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": {
                "id":              p["id"],
                "road_type":       p.get("road_type", "unclassified"),
                "tt_spread_noon":  spread_noon,
            },
        })
    lite_raw  = json.dumps({"type": "FeatureCollection", "features": lite_feats},
                           separators=(",", ":")).encode()
    lite_etag = '"' + hashlib.md5(lite_raw).hexdigest() + '"'
    lite_gz   = gzip.compress(lite_raw, compresslevel=6)

    payload = {
        "edge_lookup":       el,
        "kd_coords":         kd_coords,
        "kd_ids":            kd_ids,
        "geojson_gz":        gz_bytes,
        "geojson_etag":      etag,
        "edge_grid":         grid,
        "geojson_lite_gz":   lite_gz,
        "geojson_lite_etag": lite_etag,
    }
    try:
        STARTUP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(STARTUP_CACHE, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        print(f"Startup cache written → {STARTUP_CACHE}")
    except Exception as exc:
        print(f"Warning: could not write startup cache: {exc}")
    return payload


def _build_igraph(edge_lookup: dict) -> tuple[ig.Graph, dict, dict, dict, list]:
    """Build igraph DiGraph from edge_lookup.  Returns (G, node_to_vid, eid_to_eidx, adj, eids_list)."""
    print("Building igraph DiGraph ...")
    t0 = time.time()

    # Collect all unique nodes
    nodes: set[str] = set()
    for info in edge_lookup.values():
        u, v = info.get("u", ""), info.get("v", "")
        if u: nodes.add(u)
        if v: nodes.add(v)

    node_to_vid = {n: i for i, n in enumerate(sorted(nodes))}

    G = ig.Graph(directed=True)
    G.add_vertices(len(node_to_vid))
    G.vs["name"] = sorted(nodes)

    # Add edges in deterministic order
    eids_list: list[str] = []
    edge_tuples: list[tuple[int, int]] = []

    for eid, info in edge_lookup.items():
        u = info.get("u", "")
        v = info.get("v", "")
        if u not in node_to_vid or v not in node_to_vid:
            continue
        edge_tuples.append((node_to_vid[u], node_to_vid[v]))
        eids_list.append(eid)

    G.add_edges(edge_tuples)
    G.es["eid"] = eids_list

    eid_to_eidx = {eid: idx for idx, eid in enumerate(eids_list)}

    # Adjacency dict for movement router fallback: node_str → [eid, ...]
    adj: dict[str, list[str]] = {}
    for eid, info in edge_lookup.items():
        u = info.get("u", "")
        if u:
            adj.setdefault(u, []).append(eid)

    print(f"igraph: {G.vcount()} nodes, {G.ecount()} edges  ({time.time()-t0:.2f}s)")
    return G, node_to_vid, eid_to_eidx, adj, eids_list


def _build_weight_arrays(
    edge_lookup: dict,
    eids_list: list[str],
) -> list[np.ndarray]:
    """Pre-compute 24 weight arrays (one per hour) for igraph Dijkstra."""
    print("Pre-computing per-hour weight arrays ...")
    t0 = time.time()
    E = len(eids_list)

    # Road-type multipliers — static across hours
    type_mults = np.array([
        ROAD_PENALTY.get(edge_lookup[eid].get("road_type", "unclassified"), 1.5)
        for eid in eids_list
    ], dtype=np.float64)

    # Travel-time matrices: (E, 24) → transposed to (24, E)
    def _tt_mat(key: str, fallback_key: str = "tt") -> np.ndarray:
        return np.array(
            [edge_lookup[eid].get(key, edge_lookup[eid]["tt"]) for eid in eids_list],
            dtype=np.float64,
        ).T   # (24, E)

    tt_matrix  = _tt_mat("tt")
    tt_low_mat = _tt_mat("tt_low")
    tt_hi_mat  = _tt_mat("tt_high")

    # Per-hour fractional uncertainty spread: (tt_high − tt_low) / tt_base ∈ [0, 1]
    # Replaces the confidence-based static multiplier with data-driven, hour-specific penalty.
    spread_frac = np.clip(
        (tt_hi_mat - tt_low_mat) / np.maximum(tt_matrix, 1.0),
        0.0, 1.0,
    )  # shape (24, E)

    weights_by_hour: list[np.ndarray] = []
    for h in range(24):
        uncertainty_mults = 1.0 + spread_frac[h] * 0.2
        w = tt_matrix[h] * type_mults * uncertainty_mults + FALLBACK_TURN_S
        weights_by_hour.append(w)

    print(f"Weight arrays ready: 24 × {E} floats  ({time.time()-t0:.2f}s)")
    return weights_by_hour


def build_app_state() -> AppState:
    """Load all data and build AppState. Called once at startup."""
    from flood_engine import FloodEngine

    st = AppState()

    # ── 1. Startup cache (edge_lookup, GeoJSON blobs, KDTree) ────────────────
    sc = _load_startup_cache()
    st.edge_lookup       = sc["edge_lookup"]
    st.geojson_gz        = sc["geojson_gz"]
    st.geojson_etag      = sc["geojson_etag"]
    st.geojson_lite_gz   = sc.get("geojson_lite_gz", b"")
    st.geojson_lite_etag = sc.get("geojson_lite_etag", "")

    # Always rebuild lite GeoJSON from edge_lookup so the tt_spread_noon field
    # is current even when loading an older startup_cache.pkl.
    print("Building lite GeoJSON ...")
    _lf = []
    for _eid, _info in st.edge_lookup.items():
        _coords = _info.get("geom", [])
        if not _coords:
            continue
        _tt_lo  = _info.get("tt_low",  _info.get("tt", [0.0] * 24))
        _tt_hi  = _info.get("tt_high", _info.get("tt", [0.0] * 24))
        _spread_noon = round(_tt_hi[12] - _tt_lo[12], 1) if len(_tt_hi) > 12 and len(_tt_lo) > 12 else 0.0
        _lf.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": _coords},
            "properties": {
                "id":             _eid,
                "road_type":      _info.get("road_type", "unclassified"),
                "tt_spread_noon": _spread_noon,
            },
        })
    _lr = json.dumps({"type": "FeatureCollection", "features": _lf},
                     separators=(",", ":")).encode()
    st.geojson_lite_etag = '"' + hashlib.md5(_lr).hexdigest() + '"'
    st.geojson_lite_gz   = gzip.compress(_lr, compresslevel=6)

    # ── 2. igraph DiGraph ─────────────────────────────────────────────────────
    (st.G, st.node_to_vid, st.eid_to_eidx,
     st.adj_from_node, _eids_list) = _build_igraph(st.edge_lookup)

    # ── 3. Pre-computed weight arrays ─────────────────────────────────────────
    st.weights_by_hour = _build_weight_arrays(st.edge_lookup, _eids_list)
    st.edge_eids_arr   = np.array(_eids_list, dtype=object)

    # ── 4. Flood STRtree ──────────────────────────────────────────────────────
    print("Building Shapely STRtree ...")
    st.flood_engine = FloodEngine(st.edge_lookup)

    # ── 5. KDTree for nearest-edge ───────────────────────────────────────────
    if sc.get("kd_coords") and sc.get("kd_ids"):
        print("Building KDTree ...")
        st.kd_tree = KDTree(np.array(sc["kd_coords"], dtype=np.float64))
        st.kd_ids  = sc["kd_ids"]

    # ── 6. Precomputed criticality ────────────────────────────────────────────
    cp = CRITICALITY_PKL
    if Path(cp).exists():
        try:
            with open(cp, "rb") as f:
                crit_data = pickle.load(f)
            for hour, result in crit_data.items():
                if result.get("status") == "done":
                    st.criticality_cache[hour] = result
                    st.precomputed_hours.add(hour)
            log.info("Precomputed criticality: hours %s", sorted(st.criticality_cache))
        except Exception as exc:
            log.warning("Could not load criticality_precomputed.pkl: %s", exc)

    return st


def load_removal_services(st: AppState) -> None:
    """
    Background: load BPR state, chains, chain graph, movement router.
    No sumolib — topology built from edge_lookup.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))  # add root for edge_removal

    try:
        from bpr_engine import NetworkState
        from chain_utils import load_chains, ChainGraph

        print("Loading BPR arrays ...")
        bpr_state = NetworkState.load_from_cache(
            str(ARRAYS_PKL), st.edge_lookup
        )
        st.bpr_state = bpr_state

        print("Loading chains ...")
        chains = load_chains(str(CHAINS_PKL))
        st.chains = chains

        print("Building ChainGraph ...")
        st.chain_graph = ChainGraph(chains, bpr_state)

        print("Loading movements ...")
        from movement import MovementRouter
        mr = MovementRouter(str(MOVEMENTS_PKL), adj_from_node=st.adj_from_node)
        mr.load()
        st.movement_router = mr

        st.removal_loaded = True
        print("Removal services ready.")

    except Exception as exc:
        import traceback
        st.removal_error = str(exc)
        log.error("Removal service load FAILED: %s", exc, exc_info=True)


# Singleton
_state: AppState | None = None

def get_state() -> AppState:
    global _state
    if _state is None:
        raise RuntimeError("AppState not initialised — call init_state() first")
    return _state

def init_state() -> AppState:
    global _state
    _state = build_app_state()
    return _state
