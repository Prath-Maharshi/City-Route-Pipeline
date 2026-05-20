"""
app.py  —  Gurugram Traffic Routing (movement-aware, chain-aware)
=================================================================
Changes vs previous version
-----------------------------
Performance / correctness fixes:
  • updated_travel_times is now protected by a threading.Lock — no more
    concurrent-write corruption.
  • /removal runs remove_edge() exactly ONCE per request; both the dict
    result and the GeoJSON delta are derived from the same RemovalResult
    object (no double BPR run).
  • /nearest_edge uses a scipy KDTree built at startup — O(log N) per
    query instead of O(N) brute-force.
  • updated_travel_times accumulation now re-runs BPR on the *current*
    composite flow state rather than taking max(old, new) independently.
  • Session isolation: updated_travel_times is keyed by session_id
    (a cookie set on first visit). Each browser has its own closure
    state; one user's removals don't affect another's routes.

New features:
  • GET /edge_info?edge=<id>&hour=<h>
    Returns per-edge flow, v/c, travel time, speed, confidence, road
    type and data source — consumed by the hover sidebar in the UI.
  • GET /criticality?hour=<h>&limit=<n>
    Runs criticality_scores() for the top-N highest-flow edges and
    returns a ranked list with normalised criticality scores — consumed
    by the heatmap mode toggle in the UI.
  • New endpoint  GET /chain?edge=<id>  (unchanged from previous)
"""

import json
import csv
import threading
import uuid
import pickle
from pathlib import Path
import networkx as nx
from flask import Flask, render_template, request, jsonify, make_response
import sumolib
from edge_removal import EdgeRemovalService
from movement_router import MovementRouter
import logging as log
app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
GEOJSON_PATH   = "v1/graph_reconstruction/traffic_state_scored.geojson"
NET_PATH       = "outputs/networks/full.net.xml"
ARRAYS_PKL     = "v1/graph_reconstruction/gurugram_traffic_arrays.pkl"
MOVEMENTS_PKL  = "outputs/networks/movements.pkl"
CHAINS_PKL     = "v1/graph_reconstruction/chains.pkl"
NODES_CSV_PATH = "gurugram_nodes.csv"
CRITICALITY_PKL = "criticality_precomputed.pkl"
# ── Services ──────────────────────────────────────────────────────────────────
removal_service = EdgeRemovalService(ARRAYS_PKL, NET_PATH, CHAINS_PKL)
movement_router = MovementRouter(MOVEMENTS_PKL, fallback_penalty_s=5.0)


# ── Per-session travel time state ─────────────────────────────────────────────
# Structure: { session_id: { hour: { edge_idx: tt_val } } }
# Guarded by _tt_lock for all reads and writes.
_session_tt: dict[str, dict] = {}
_tt_lock = threading.Lock()

def _get_session_id(request) -> str:
    """Return existing session id from cookie, or create a new one."""
    return request.cookies.get("_gtt_sid") or str(uuid.uuid4())

def _get_session_tt(session_id: str) -> dict:
    """Return the hour→{idx:tt} map for this session (never None)."""
    with _tt_lock:
        return _session_tt.setdefault(session_id, {})

def _update_session_tt(session_id: str, hour: int, result) -> None:
    """
    Merge a new RemovalResult into the session's composite travel-time state.

    Instead of max(old, new) we re-run BPR on the *accumulated* flow
    delta so that consecutive removals compound correctly.

    result.tt_updated already reflects BPR applied to
    (base_flow + all prior deltas + this delta).  We simply overwrite
    the per-edge values with the new BPR output — which is the
    simulation's best estimate of the composite state.
    """
    raw = result.get("tt_updated_raw")
    if not raw:
        return
    with _tt_lock:
        session = _session_tt.setdefault(session_id, {})
        hour_map = session.setdefault(hour, {})
        for idx, tt_val in enumerate(raw):
            hour_map[idx] = tt_val

def _clear_session_tt(session_id: str) -> None:
    with _tt_lock:
        _session_tt.pop(session_id, None)

def _read_session_tt(session_id: str, hour: int) -> dict:
    """Return a snapshot of {idx: tt} for this session/hour (safe copy)."""
    with _tt_lock:
        return dict(_session_tt.get(session_id, {}).get(hour, {}))

# In-process criticality cache: hour → {"status","scores",...}
_criticality_cache: dict = {}
_precomputed_hours: set = set()

def _load_precomputed_criticality():
    global _precomputed_hours
    p = Path(CRITICALITY_PKL)
    if not p.exists():
        log.info("No precomputed criticality file found — will compute on demand.")
        return
    with open(p, "rb") as f:
        data = pickle.load(f)
    for hour, result in data.items():
        if result.get("status") == "done":
            _criticality_cache[hour] = result
            _precomputed_hours.add(hour)
    log.info("Loaded precomputed criticality for hours: %s",
             sorted(_criticality_cache.keys()))

_load_precomputed_criticality()

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    try:
        with open(GEOJSON_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        fallback = "outputs/graph_reconstruction/traffic_state.geojson"
        with open(fallback) as f:
            return json.load(f)

data = load_data()
print(f"Loading SUMO network from {NET_PATH}...")
net = sumolib.net.readNet(NET_PATH)

# ── Graph construction ────────────────────────────────────────────────────────
G           = nx.DiGraph()
edge_lookup = {}

print("Building graph...")
for feat in data["features"]:
    props  = feat["properties"]
    eid    = props["id"]
    coords = feat["geometry"]["coordinates"]

    try:
        sumo_edge = net.getEdge(eid)
    except KeyError:
        continue

    u = sumo_edge.getFromNode().getID()
    v = sumo_edge.getToNode().getID()

    tt_arr  = props.get("travel_time_s",    [1.0] * 24)
    tt_low  = props.get("travel_time_low_s",  tt_arr.copy())
    tt_high = props.get("travel_time_high_s", tt_arr.copy())

    if len(tt_arr)  < 24: tt_arr  += [tt_arr[-1]]  * (24 - len(tt_arr))
    if len(tt_low)  < 24: tt_low  += [tt_low[-1]]  * (24 - len(tt_low))
    if len(tt_high) < 24: tt_high += [tt_high[-1]] * (24 - len(tt_high))

    length     = props.get("length_m",           1.0)
    road_type  = props.get("road_type",  "unclassified")
    confidence = props.get("confidence",          0.1)
    dom_dir    = props.get("dominant_direction", "unknown")

    G.add_edge(u, v, id=eid, tt=tt_arr, tt_low=tt_low, tt_high=tt_high,
               length=length, geom=coords, road_type=road_type,
               confidence=confidence, dom_dir=dom_dir)

    edge_lookup[eid] = {
        "u": u, "v": v, "to_node": v,
        "geom": coords,
        "tt": tt_arr, "tt_low": tt_low, "tt_high": tt_high,
        "length": length, "road_type": road_type,
        "confidence": confidence, "dom_dir": dom_dir,
    }

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# ── Spatial index for /nearest_edge ──────────────────────────────────────────
print("Building spatial index...")
try:
    import numpy as np
    from scipy.spatial import KDTree

    _edge_coords: list[tuple[float, float]] = []
    _edge_ids_ordered: list[str]            = []

    for feat in data["features"]:
        coords = feat["geometry"]["coordinates"]
        mid    = coords[len(coords) // 2]
        _edge_coords.append((mid[0], mid[1]))          # (lng, lat)
        _edge_ids_ordered.append(feat["properties"]["id"])

    _kd_tree = KDTree(np.array(_edge_coords, dtype=np.float64))
    _USE_KDTREE = True
    print(f"KDTree built over {len(_edge_ids_ordered)} edges.")
except ImportError:
    _USE_KDTREE = False
    print("scipy not available — falling back to O(N) nearest-edge search.")

# ── Background service startup ────────────────────────────────────────────────
threading.Thread(target=movement_router.load, daemon=True).start()

def _load_removal_service():
    removal_service.load()
    removal_service.set_nx_graph(G)

threading.Thread(target=_load_removal_service, daemon=True).start()

# ── Routing constants ─────────────────────────────────────────────────────────
ROAD_PENALTY = {
    "motorway": 0.6,     "trunk": 0.7,          "trunk_link": 0.7,
    "primary": 0.85,     "secondary": 1.0,      "secondary_link": 1.0,
    "tertiary": 1.2,     "tertiary_link": 1.2,
    "unclassified": 1.4, "residential": 1.6,
}
FALLBACK_TURN_PENALTY_S = 5


def _expand_blocked(blocked_edges: set[str]) -> set[str]:
    if not removal_service.loaded:
        return blocked_edges
    expanded: set[str] = set()
    for eid in blocked_edges:
        if eid not in expanded:
            expanded.update(removal_service.expand_chain(eid))
    return expanded

# After _criticality_cache: dict = {}  add:
def _load_precomputed_criticality():
    p = Path(CRITICALITY_PKL)
    if not p.exists():
        log.info("No precomputed criticality file found — will compute on demand.")
        return
    with open(p, "rb") as f:
        data = pickle.load(f)
    for hour, result in data.items():
        if result.get("status") == "done":
            _criticality_cache[hour] = result
    log.info("Loaded precomputed criticality for hours: %s",
             sorted(_criticality_cache.keys()))

def _impedance_fallback(u, v, edge_data, hour, blocked_edges=None,
                        tt_override: dict | None = None):
    if blocked_edges and edge_data["id"] in blocked_edges:
        return float("inf")

    eid = edge_data["id"]
    if tt_override and eid in tt_override:
        base_t = tt_override[eid]
    else:
        base_t = edge_data["tt"][hour]

    type_m = ROAD_PENALTY.get(edge_data.get("road_type", "unclassified"), 1.5)
    conf_m = 1.0 + (1.0 - edge_data.get("confidence", 0.1)) * 0.2
    return base_t * type_m * conf_m + FALLBACK_TURN_PENALTY_S


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    response = make_response(render_template("index.html"))
    if not request.cookies.get("_gtt_sid"):
        response.set_cookie("_gtt_sid", str(uuid.uuid4()), samesite="Lax")
    return response

@app.route("/geojson")
def serve_geojson():
    return jsonify(data)

@app.route("/nodes")
def serve_nodes():
    nodes = []
    try:
        with open(NODES_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    nodes.append({
                        "name": row["name"].strip(),
                        "type": row.get("type", "").strip(),
                        "lat":  float(row["lat"]),
                        "lng":  float(row["lon"]),
                    })
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        return jsonify([])
    return jsonify(nodes)


@app.route("/nearest_edge")
def nearest_edge():
    try:
        lat = float(request.args.get("lat"))
        lng = float(request.args.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lng required"}), 400

    if _USE_KDTREE:
        dist, idx = _kd_tree.query([lng, lat])
        best_eid  = _edge_ids_ordered[idx]
        return jsonify({"edge_id": best_eid, "dist_deg": float(dist)})

    # O(N) fallback (scipy not installed)
    best_eid  = None
    best_dist = float("inf")
    for feat in data["features"]:
        coords = feat["geometry"]["coordinates"]
        mid    = coords[len(coords) // 2]
        dx     = mid[0] - lng
        dy     = mid[1] - lat
        d2     = dx * dx + dy * dy
        if d2 < best_dist:
            best_dist = d2
            best_eid  = feat["properties"]["id"]

    if best_eid is None:
        return jsonify({"error": "No edges in network"}), 500
    return jsonify({"edge_id": best_eid, "dist_deg": best_dist ** 0.5})


@app.route("/edge_info")
def edge_info():
    """
    GET /edge_info?edge=<edge_id>&hour=<h>

    Returns per-edge traffic state for the hover sidebar.
    Response:
        {
            "edge_id":    "...",
            "road_type":  "primary",
            "length_m":   312.4,
            "capacity":   1800.0,
            "flow_veh_h": 1150.3,
            "vc_ratio":   0.639,
            "tt_s":       42.1,
            "speed_kmh":  26.7,
            "confidence": 0.82,
            "data_source": "sensor",
            "chain_len":  3
        }
    """
    edge_id = request.args.get("edge", "").strip()
    try:
        hour = max(0, min(23, int(request.args.get("hour", 12))))
    except ValueError:
        hour = 12

    if not edge_id:
        return jsonify({"error": "edge param required"}), 400

    # Fast path: removal service has richer data
    if removal_service.loaded:
        info = removal_service.get_edge_info(edge_id, hour)
        if info:
            # Augment with geojson confidence if available
            el = edge_lookup.get(edge_id, {})
            info["confidence"] = el.get("confidence", info.get("confidence", 0.0))
            # Compute speed from tt and length
            tt  = info.get("tt_s", 1.0)
            lm  = info.get("length_m", 1.0)
            info["speed_kmh"] = round((lm / max(tt, 0.1)) * 3.6, 1)
            return jsonify(info)

    # Fallback: edge_lookup only
    el = edge_lookup.get(edge_id)
    if not el:
        return jsonify({"error": "Edge not found"}), 404

    tt  = el["tt"][hour]
    lm  = el["length"]
    return jsonify({
        "edge_id":    edge_id,
        "road_type":  el.get("road_type", "unknown"),
        "length_m":   round(lm, 1),
        "capacity":   None,
        "flow_veh_h": None,
        "vc_ratio":   None,
        "tt_s":       round(tt, 1),
        "speed_kmh":  round((lm / max(tt, 0.1)) * 3.6, 1),
        "confidence": round(el.get("confidence", 0.0), 3),
        "data_source": "geojson",
        "chain_len":   1,
    })


@app.route("/criticality")
def get_criticality():
    """
    GET /criticality?hour=<h>
    """
    try:
        hour = max(0, min(23, int(request.args.get("hour", 8))))
    except ValueError:
        hour = 8

    # ── 1. Cache hit (Instant return, even if services are booting) ───────────
    cached = _criticality_cache.get(hour)
    if cached and cached.get("status") == "done":
        return jsonify(cached)

    # ── 2. Guard background tools ─────────────────────────────────────────────
    if not removal_service.loaded:
        return jsonify({"error": "Removal service still loading"}), 503

    # ── 3. Already running ────────────────────────────────────────────────────
    running = _criticality_cache.get(hour)
    if running and running.get("status") == "running":
        return jsonify({
            "status": "running",
            "hour":   hour,
            "done":   running.get("done", 0),
            "total":  running.get("total", 0),
            "pct":    running.get("pct", 0),
        })

    # ── Start background computation ──────────────────────────────────────────
    state = removal_service._state
    flow_col = state.flows[hour]

    # ALL eligible edges — no artificial limit
    candidates = [
        state.edge_ids[i]
        for i in range(state.N)
        if state.data_source[i] in ("sensor", "reconstructed")
        and state.nodes_from[i] >= 0
        and float(flow_col[i]) > 5.0
    ]

    _criticality_cache[hour] = {
        "status": "running",
        "hour":   hour,
        "done":   0,
        "total":  len(candidates),
        "pct":    0,
    }

    def _run(hour, candidates):
        from edge_removal import remove_edge as _re
        state  = removal_service._state
        chains = removal_service._chains
        G_nx   = removal_service._G
        import numpy as np

        raw = []
        total = len(candidates)
        for i, eid in enumerate(candidates):
            try:
                r = _re(state, eid, hour, chain_index=chains, nx_graph=G_nx)
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
            except Exception:
                pass
            # Update progress every 25 edges
            if i % 25 == 0:
                pct = round(i / total * 100, 1) if total else 100
                entry = _criticality_cache.get(hour, {})
                entry["done"]  = i
                entry["pct"]   = pct
                _criticality_cache[hour] = entry

        # ── Composite score ────────────────────────────────────────────────
        if raw:
            def _norm(arr):
                import numpy as np
                import math
                # Filter out pure NaNs gracefully
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                mn, mx = arr.min(), arr.max()
                if mx - mn < 1e-9:
                    return np.zeros_like(arr)
                return (arr - mn) / (mx - mn)

            delays    = np.array([s.get("total_delay_veh_h", 0) for s in raw], dtype=np.float64)
            flows     = np.array([s.get("displaced_flow", 0)     for s in raw], dtype=np.float64)
            cong      = np.array([s.get("n_congested", 0)        for s in raw], dtype=np.float64)
            isol      = np.array([1.0 - min(s.get("rerouted_pct", 100), 100) / 100.0 for s in raw], dtype=np.float64)

            composite = np.clip(
                0.40 * _norm(delays)
              + 0.30 * _norm(flows)
              + 0.20 * _norm(cong)
              + 0.10 * _norm(isol),
                0.0, 1.0
            )
            import math
            for s, c in zip(raw, composite):
                # Ensure JSON valid primitives
                val = float(c)
                s["criticality"] = round(val, 4) if math.isfinite(val) else 0.0
            raw.sort(key=lambda s: -s["criticality"])

        _criticality_cache[hour] = {
            "status": "done",
            "hour":   hour,
            "total":  len(raw),
            "scores": raw,
        }
        log.info("Criticality done: %d edges scored at hour %02d", len(raw), hour)

    threading.Thread(target=_run, args=(hour, candidates), daemon=True).start()
    return jsonify({
        "status": "started",
        "hour":   hour,
        "total":  len(candidates),
        "pct":    0,
    })


@app.route("/chain")
def get_chain():
    edge_id = request.args.get("edge", "").strip()
    if not edge_id:
        return jsonify({"error": "edge param required"}), 400

    if not removal_service.loaded:
        return jsonify({
            "edge_id":   edge_id,
            "chain":     [edge_id],
            "chain_len": 1,
            "is_multi":  False,
            "note":      "chain service loading",
        })

    summary = removal_service.chain_summary(edge_id)
    return jsonify(summary)


@app.route("/status")
def status():
    return jsonify({
        "graph_edges":      G.number_of_edges(),
        "graph_nodes":      G.number_of_nodes(),
        "movements_loaded": movement_router.loaded,
        "movements_count":  len(movement_router._movements),
        "removal_loaded":   removal_service.loaded,
        "chains_loaded":    removal_service.loaded and removal_service._chains is not None,
        "chain_count":      len(removal_service._chains) if (
                                removal_service.loaded and removal_service._chains) else 0,
        "kdtree_active":    _USE_KDTREE,
    })


@app.route("/route")
def get_route():
    session_id = _get_session_id(request)
    start_id = request.args.get("start")
    end_id   = request.args.get("end")
    blocked_str   = request.args.get("blocked", "")
    raw_blocked   = set(filter(None, blocked_str.split(",")))
    blocked_edges = _expand_blocked(raw_blocked)

    try:
        hour = max(0, min(23, int(request.args.get("hour", 0))))
    except ValueError:
        hour = 0

    use_movements = request.args.get("movements", "true").lower() == "true"

    if start_id not in edge_lookup or end_id not in edge_lookup:
        return jsonify({"error": f"Invalid edge IDs: {start_id} or {end_id}"}), 400

    if start_id in blocked_edges:
        return jsonify({"error": "Origin edge is part of a blocked chain"}), 400
    if end_id in blocked_edges:
        return jsonify({"error": "Destination edge is part of a blocked chain"}), 400

    start_data = edge_lookup[start_id]
    end_data   = edge_lookup[end_id]

    # ── Per-session BPR travel-time override ─────────────────────────────────
    tt_override: dict[str, float] = {}
    session_tt = _read_session_tt(session_id, hour)
    if session_tt and removal_service.loaded:
        idx_map = removal_service._state.edge_index
        for eid, idx in idx_map.items():
            if idx in session_tt:
                tt_override[eid] = session_tt[idx]

    # ── Movement-aware routing ────────────────────────────────────────────────
    if use_movements and movement_router.loaded and movement_router.has_movements:
        result = movement_router.route_via_line_graph(
            G_nx          = G,
            edge_lookup   = edge_lookup,
            start_edge    = start_id,
            end_edge      = end_id,
            hour          = hour,
            blocked_edges = blocked_edges,
            tt_override   = tt_override,
        )

        if result is not None:
            route_coords = []
            for eid in result["edges"]:
                ed   = edge_lookup.get(eid, {})
                geom = ed.get("geom", [])
                if not route_coords:
                    route_coords.extend(geom)
                else:
                    route_coords.extend(geom[1:])

            response = make_response(jsonify({
                "geometry":        {"type": "LineString", "coordinates": route_coords},
                "time_range_s":    [result["total_time_low_s"], result["total_time_high_s"]],
                "time_expected_s": result["expected_time_s"],
                "time_total_s":    result["total_time_s"],
                "turn_time_s":     result["turn_time_s"],
                "n_turns":         result["n_turns"],
                "dist_m":          round(result["dist_m"], 2),
                "edges":           result["edges"],
                "routing_mode":    "movement_graph",
                "blocked_expanded": list(blocked_edges),
            }))
            if not request.cookies.get("_gtt_sid"):
                response.set_cookie("_gtt_sid", session_id, samesite="Lax")
            return response

    # ── Fallback: NetworkX Dijkstra ───────────────────────────────────────────
    try:
        if start_id == end_id:
            path_edges    = [start_data]
            route_coords  = list(start_data["geom"])
            edge_sequence = [start_id]
        else:
            path_nodes = nx.shortest_path(
                G, start_data["v"], end_data["u"],
                weight=lambda u, v, d: _impedance_fallback(
                    u, v, d, hour, blocked_edges, tt_override
                ),
            )
            path_edges    = [start_data]
            route_coords  = list(start_data["geom"])
            edge_sequence = [start_id]

            for i in range(len(path_nodes) - 1):
                u, v  = path_nodes[i], path_nodes[i + 1]
                edata = G[u][v]
                path_edges.append(edata)
                route_coords.extend(edata["geom"][1:])
                edge_sequence.append(edata["id"])

            path_edges.append(end_data)
            route_coords.extend(end_data["geom"][1:])
            edge_sequence.append(end_id)

        total_low = total_high = total_exp = total_dist = 0.0
        for e in path_edges:
            eid    = e.get("id", "")
            base_t = tt_override.get(eid, e["tt"][hour])
            low_t  = e["tt_low"][hour]
            high_t = e["tt_high"][hour]
            ddir   = e.get("dom_dir", "unknown")
            total_low  += low_t
            total_high += high_t
            total_dist += e["length"]
            if ddir == "underestimate":
                adj_t = base_t + 0.75 * (high_t - base_t)
            elif ddir == "overestimate":
                adj_t = base_t - 0.75 * (base_t - low_t)
            else:
                adj_t = base_t
            total_exp += max(low_t, min(high_t, adj_t))

        n_trans    = max(0, len(path_edges) - 1)
        turn_total = n_trans * FALLBACK_TURN_PENALTY_S
        total_low  += turn_total
        total_high += turn_total
        total_exp  += turn_total

        response = make_response(jsonify({
            "geometry":        {"type": "LineString", "coordinates": route_coords},
            "time_range_s":    [round(total_low, 2), round(total_high, 2)],
            "time_expected_s": round(total_exp, 2),
            "dist_m":          round(total_dist, 2),
            "edges":           edge_sequence,
            "routing_mode":    "nx_fallback",
            "blocked_expanded": list(blocked_edges),
        }))
        if not request.cookies.get("_gtt_sid"):
            response.set_cookie("_gtt_sid", session_id, samesite="Lax")
        return response

    except nx.NetworkXNoPath:
        return jsonify({"error": "No path exists between these edges"}), 404
    except nx.NodeNotFound:
        return jsonify({"error": "Disconnected network node"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/removal")
def get_removal():
    """
    Runs remove_edge() exactly ONCE — the same RemovalResult is used to
    produce both the summary dict and the GeoJSON delta.  No double BPR.
    """
    session_id = _get_session_id(request)
    edge_id    = request.args.get("edge")
    try:
        hour = max(0, min(23, int(request.args.get("hour", 0))))
    except ValueError:
        hour = 0

    if not edge_id or edge_id not in edge_lookup:
        return jsonify({"error": "Invalid edge ID"}), 400
    if not removal_service.loaded:
        return jsonify({"error": "Removal service loading — please wait"}), 503

    # ── Single simulation run ─────────────────────────────────────────────────
    from edge_removal import remove_edge as _remove_edge

    removal_result_obj = _remove_edge(
        removal_service._state,
        edge_id,
        hour,
        chain_index = removal_service._chains,
        nx_graph    = removal_service._G,
    )

    result  = removal_result_obj.to_dict()
    geojson = removal_result_obj.to_geojson_delta(
        removal_service._state, removal_service._net
    ) if removal_service._net else {"type": "FeatureCollection", "features": []}

    # ── Update per-session composite travel-time state ────────────────────────
    _update_session_tt(session_id, hour, result)

    result["delta_geojson"] = geojson

    response = make_response(jsonify(result))
    if not request.cookies.get("_gtt_sid"):
        response.set_cookie("_gtt_sid", session_id, samesite="Lax")
    return response

@app.route("/speed_floor")
def get_speed_floor():
    """
    GET /speed_floor?edge=<id>&hour=<h>&min_speed=<kmh>

    Imposes a minimum speed on an edge and returns the flow redistribution.
    Response shape is identical to /removal for UI reuse.
    """
    session_id  = _get_session_id(request)
    edge_id     = request.args.get("edge")
    try:
        hour      = max(0, min(23, int(request.args.get("hour", 0))))
        min_speed = float(request.args.get("min_speed", 30.0))
    except ValueError:
        return jsonify({"error": "Invalid hour or min_speed"}), 400

    if not edge_id or edge_id not in edge_lookup:
        return jsonify({"error": "Invalid edge ID"}), 400
    if not removal_service.loaded:
        return jsonify({"error": "Removal service loading — please wait"}), 503
    if min_speed <= 0:
        return jsonify({"error": "min_speed must be > 0"}), 400

    from edge_removal import apply_speed_floor as _apply_sf

    result_obj = _apply_sf(
        removal_service._state,
        edge_id,
        hour,
        min_speed_kmh = min_speed,
        chain_index   = removal_service._chains,
        nx_graph      = removal_service._G,
    )

    result  = result_obj.to_dict()
    geojson = result_obj.to_geojson_delta(
        removal_service._state, removal_service._net
    ) if removal_service._net else {"type": "FeatureCollection", "features": []}

    _update_session_tt(session_id, hour, result)
    result["delta_geojson"] = geojson
    result["simulation_mode"] = "speed_floor"
    result["min_speed_kmh"]   = min_speed

    response = make_response(jsonify(result))
    if not request.cookies.get("_gtt_sid"):
        response.set_cookie("_gtt_sid", session_id, samesite="Lax")
    return response


@app.route("/saturation_corridors")
def get_saturation_corridors():
    """
    GET /saturation_corridors?hour=<h>&vc_threshold=<float>&limit=<n>

    After one or more removals/speed-floors in this session, returns the
    detour corridors most at risk of saturation — edges whose v/c ratio
    has increased significantly above the threshold.

    Uses the session's composite travel-time / flow state, so it reflects
    all closures applied so far without re-running BPR.

    Response:
    {
      "hour": 8,
      "vc_threshold": 0.85,
      "corridors": [
        {
          "edge_id": "...",
          "road_type": "primary",
          "flow_veh_h": 1820.0,
          "capacity": 1800.0,
          "vc_ratio": 1.011,
          "vc_delta": +0.21,     # change vs. base state
          "tt_delta_s": +18.4,   # extra seconds vs. base
          "data_source": "sensor"
        },
        ...
      ],
      "n_at_risk": 12
    }
    """
    session_id = _get_session_id(request)
    try:
        hour         = max(0, min(23, int(request.args.get("hour", 0))))
        vc_threshold = float(request.args.get("vc_threshold", 0.85))
        limit        = min(int(request.args.get("limit", 30)), 200)
    except ValueError:
        return jsonify({"error": "Invalid parameter"}), 400

    if not removal_service.loaded:
        return jsonify({"error": "Removal service loading"}), 503

    state      = removal_service._state
    session_tt = _read_session_tt(session_id, hour)

    if not session_tt:
        return jsonify({
            "hour": hour, "vc_threshold": vc_threshold,
            "corridors": [], "n_at_risk": 0,
            "note": "No removals applied in this session yet",
        })

    # Rebuild updated flow from the session's BPR travel-time overrides.
    # We back-calculate flow from tt_updated using the inverse BPR formula:
    #   vc = ((tt/tt_free - 1) / alpha)^(1/beta)
    #   flow = vc * capacity
    import numpy as np
    from edge_removal import BPR_ALPHA, BPR_BETA
    
    tt_base    = state.travel_time[hour]     # shape (N,)
    cap        = state.capacity               # shape (N,)
    flow_base  = state.flows[hour]            # shape (N,)
    sf         = state.speed_free             # shape (N,)

    # tt_free = length_m * 3.6 / speed_free
    tt_free = state.length_m * 3.6 / np.maximum(sf, 0.5)

    corridors = []
    for idx, tt_val in session_tt.items():
        if idx >= state.N:
            continue
        eid = state.edge_ids[idx]

        tt_b   = float(tt_base[idx])
        tt_a   = float(tt_val)
        cap_i  = float(cap[idx])
        f_base = float(flow_base[idx])

        if cap_i < 1.0:
            continue

        # Inverse BPR for updated flow
        ttf  = float(tt_free[idx])
        if ttf < 0.01:
            continue
        ratio = tt_a / ttf
        if ratio <= 1.0:
            vc_a = 0.0
        else:
            vc_a = ((ratio - 1.0) / BPR_ALPHA) ** (1.0 / BPR_BETA)

        vc_b   = f_base / cap_i
        vc_delta = vc_a - vc_b

        # Only surface edges that crossed the threshold OR are near it with notable delta
        if vc_a < vc_threshold and vc_delta < 0.05:
            continue

        f_updated = vc_a * cap_i
        corridors.append({
            "edge_id":    eid,
            "road_type":  state.road_type[idx],
            "flow_veh_h": round(f_updated, 1),
            "capacity":   round(cap_i, 1),
            "vc_ratio":   round(vc_a, 4),
            "vc_delta":   round(vc_delta, 4),
            "tt_delta_s": round(tt_a - tt_b, 2),
            "data_source": state.data_source[idx],
        })

    corridors.sort(key=lambda c: -c["vc_ratio"])
    corridors = corridors[:limit]

    return jsonify({
        "hour":         hour,
        "vc_threshold": vc_threshold,
        "n_at_risk":    len(corridors),
        "corridors":    corridors,
    })

@app.route("/restore")
def restore():
    session_id = _get_session_id(request)
    _clear_session_tt(session_id)
    
    # Only clear hours that were computed on-demand; keep precomputed ones safely.
    for h in list(_criticality_cache.keys()):
        if h not in _precomputed_hours:
            _criticality_cache.pop(h, None)
            
    response = make_response(jsonify({"ok": True}))
    if not request.cookies.get("_gtt_sid"):
        response.set_cookie("_gtt_sid", session_id, samesite="Lax")
    return response


@app.route("/turn_info")
def get_turn_info():
    from_e = request.args.get("from")
    to_e   = request.args.get("to")
    if not from_e or not to_e:
        return jsonify({"error": "from and to params required"}), 400
    info = movement_router.get_turn_info(from_e, to_e)
    if info is None:
        return jsonify({"found": False, "fallback_s": FALLBACK_TURN_PENALTY_S})
    return jsonify({"found": True, **info})


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/route",                methods=["OPTIONS"])
@app.route("/removal",              methods=["OPTIONS"])
@app.route("/speed_floor",          methods=["OPTIONS"])
@app.route("/saturation_corridors", methods=["OPTIONS"])
@app.route("/turn_info",            methods=["OPTIONS"])
@app.route("/nodes",                methods=["OPTIONS"])
@app.route("/nearest_edge",         methods=["OPTIONS"])
@app.route("/restore",              methods=["OPTIONS"])
@app.route("/chain",                methods=["OPTIONS"])
@app.route("/edge_info",            methods=["OPTIONS"])
@app.route("/criticality",          methods=["OPTIONS"])
def preflight():
    return "", 204


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, threaded=True, use_reloader=False)
