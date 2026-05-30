"""
main.py — FastAPI v2 of the Gurugram Traffic Routing app.

Stack vs v1:
  Flask  → FastAPI + uvicorn    (async I/O, proper ASGI)
  NetworkX Dijkstra → igraph    (C-level, releases GIL)
  Grid + ray-cast   → Shapely STRtree  (O(log N), GEOS C-level)
  sumolib at startup → topology from edge_lookup  (no 3-min XML parse)
  Daemon threads    → FastAPI BackgroundTasks      (clean lifecycle)
  NetworkX topology fallback → plain adj dict     (no NX overhead)

Same API surface as v1 — frontend requires no changes.
"""
from __future__ import annotations

import csv
import gzip
import logging
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import igraph as ig
from fastapi import BackgroundTasks, FastAPI, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response as RawResponse
from fastapi.templating import Jinja2Templates

import tasks as T
import state as S
from state import ROAD_PENALTY, FALLBACK_TURN_S, GEOCODED_DIR, NODES_CSV

log = logging.getLogger("main")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BLOCKED_W = 1e15   # stand-in for ∞ in igraph Dijkstra (inf not supported)

_ROAD_CONN_WEIGHT: dict[str, float] = {
    "motorway": 1.0,  "motorway_link": 0.80,
    "trunk":    0.90, "trunk_link":    0.70,
    "primary":  0.70, "primary_link":  0.55,
    "secondary":0.40, "secondary_link":0.35,
    "tertiary": 0.20, "tertiary_link": 0.15,
}


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build AppState synchronously (it's fast now — no sumolib), then load
    BPR/chains/movements in a daemon thread so the server starts immediately."""
    st = S.init_state()
    threading.Thread(target=S.load_removal_services, args=(st,), daemon=True).start()
    yield

app = FastAPI(title="City Route Pipeline v2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_tmpl_dir = Path(__file__).parent / "templates"
templates  = Jinja2Templates(directory=str(_tmpl_dir))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sid(request: Request) -> str:
    return request.cookies.get("_gtt_sid") or str(uuid.uuid4())

def _set_sid(response: Response, request: Request, sid: str) -> None:
    if not request.cookies.get("_gtt_sid"):
        response.set_cookie("_gtt_sid", sid, samesite="lax")

def _expand_blocked(st: S.AppState, raw: set[str]) -> set[str]:
    if not st.removal_loaded or st.chains is None:
        return raw
    out: set[str] = set()
    for eid in raw:
        out.update(st.chains.expand_one(eid))
    return out

def _session_tt_override(st: S.AppState, sid: str, hour: int) -> dict[str, float]:
    """Build eid → tt map from session's per-index BPR state."""
    if not st.removal_loaded or st.bpr_state is None:
        return {}
    session_tt = T.read_session_tt(sid, hour)
    if not session_tt:
        return {}
    idx_map = st.bpr_state.edge_index
    return {eid: session_tt[idx] for eid, idx in idx_map.items() if idx in session_tt}

def _accumulated_flows(st: S.AppState, sid: str, hour: int) -> Optional[np.ndarray]:
    if not st.removal_loaded or st.bpr_state is None:
        return None
    from bpr_engine import _build_accumulated_flows
    return _build_accumulated_flows(st.bpr_state, T.read_session_tt(sid, hour), hour)


# ── igraph Dijkstra (fallback routing) ───────────────────────────────────────

def _igraph_route(
    st: S.AppState,
    source_node: str,
    target_node: str,
    hour: int,
    blocked_edges: set[str],
    tt_override: dict[str, float],
) -> Optional[list[str]]:
    """Return edge-ID path or None. Uses pre-built per-hour weight arrays."""
    G           = st.G
    nv          = st.node_to_vid
    src_vid     = nv.get(source_node)
    tgt_vid     = nv.get(target_node)
    if src_vid is None or tgt_vid is None:
        return None

    weights = st.weights_by_hour[hour].copy()   # numpy float64, O(E) C copy

    for eid in blocked_edges:
        eidx = st.eid_to_eidx.get(eid)
        if eidx is not None:
            weights[eidx] = BLOCKED_W

    if tt_override:
        for eid, tt_val in tt_override.items():
            eidx = st.eid_to_eidx.get(eid)
            if eidx is not None:
                info   = st.edge_lookup.get(eid, {})
                type_m = ROAD_PENALTY.get(info.get("road_type", "unclassified"), 1.5)
                conf_m = 1.0 + (1.0 - info.get("confidence", 0.1)) * 0.2
                weights[eidx] = tt_val * type_m * conf_m + FALLBACK_TURN_S

    try:
        path_eidxs = G.get_shortest_path(
            src_vid, tgt_vid, weights=weights.tolist(), output="epath"
        )
    except Exception:
        return None

    if not path_eidxs:
        return None

    eids_arr = st.edge_eids_arr
    return [str(eids_arr[i]) for i in path_eidxs]


def _build_route_response(
    st: S.AppState,
    edge_sequence: list[str],
    hour: int,
    tt_override: dict[str, float],
    turn_times: list[float],
    routing_mode: str,
    blocked_expanded: list[str],
    result: Optional[dict] = None,
) -> dict:
    """Assemble response dict from an ordered edge sequence."""
    el = st.edge_lookup
    route_coords: list = []
    total_low = total_high = total_exp = total_dist = 0.0
    n_turns    = max(0, len(edge_sequence) - 1)
    turn_total = sum(turn_times) if turn_times else n_turns * FALLBACK_TURN_S

    for i, eid in enumerate(edge_sequence):
        info   = el.get(eid, {})
        geom   = info.get("geom", [])
        if not route_coords:
            route_coords.extend(geom)
        else:
            route_coords.extend(geom[1:])

        tt_arr = info.get("tt",     [30.0] * 24)
        lo_arr = info.get("tt_low", tt_arr)
        hi_arr = info.get("tt_high",tt_arr)
        base_t = tt_override.get(eid, tt_arr[hour] if hour < len(tt_arr) else tt_arr[-1])
        low_t  = lo_arr[hour] if hour < len(lo_arr) else lo_arr[-1]
        high_t = hi_arr[hour] if hour < len(hi_arr) else hi_arr[-1]
        low_t  = max(low_t,  base_t) if tt_override.get(eid) else low_t
        high_t = max(high_t, base_t) if tt_override.get(eid) else high_t
        dom    = info.get("dom_dir", "unknown")
        if dom == "underestimate":
            adj_t = base_t + 0.75 * (high_t - base_t)
        elif dom == "overestimate":
            adj_t = base_t - 0.75 * (base_t - low_t)
        else:
            adj_t = base_t

        total_low  += low_t
        total_high += high_t
        total_exp  += max(low_t, min(high_t, adj_t))
        total_dist += info.get("length", 0.0)

    total_low  += turn_total
    total_high += turn_total
    total_exp  += turn_total

    out = {
        "geometry":        {"type": "LineString", "coordinates": route_coords},
        "time_range_s":    [round(total_low, 2), round(total_high, 2)],
        "time_expected_s": round(total_exp, 2),
        "dist_m":          round(total_dist, 2),
        "edges":           edge_sequence,
        "routing_mode":    routing_mode,
        "blocked_expanded": blocked_expanded,
    }
    if result:
        out.update(result)
    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request, response: Response, _gtt_sid: str = Cookie(default="")):
    sid = _gtt_sid or str(uuid.uuid4())
    _set_sid(response, request, sid)
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/geojson")
def serve_geojson(request: Request):
    st = S.get_state()
    if request.headers.get("If-None-Match") == st.geojson_etag:
        return RawResponse(status_code=304)
    use_gz = "gzip" in request.headers.get("Accept-Encoding", "")
    body   = st.geojson_gz if use_gz else gzip.decompress(st.geojson_gz)
    hdrs   = {
        "ETag": st.geojson_etag,
        "Cache-Control": "private, max-age=3600",
        "Vary": "Accept-Encoding",
        "Content-Type": "application/json",
    }
    if use_gz:
        hdrs["Content-Encoding"] = "gzip"
    return RawResponse(content=body, headers=hdrs)


@app.get("/geojson_lite")
def serve_geojson_lite(request: Request):
    st = S.get_state()
    if request.headers.get("If-None-Match") == st.geojson_lite_etag:
        return RawResponse(status_code=304)
    use_gz = "gzip" in request.headers.get("Accept-Encoding", "")
    body   = st.geojson_lite_gz if use_gz else gzip.decompress(st.geojson_lite_gz)
    hdrs   = {
        "ETag": st.geojson_lite_etag,
        "Cache-Control": "private, max-age=3600",
        "Vary": "Accept-Encoding",
        "Content-Type": "application/json",
    }
    if use_gz:
        hdrs["Content-Encoding"] = "gzip"
    return RawResponse(content=body, headers=hdrs)


@app.get("/nodes")
def serve_nodes():
    nodes = []
    try:
        with open(NODES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
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
        return JSONResponse([])
    return JSONResponse(nodes)


@app.get("/nearest_edge")
def nearest_edge(lat: float, lng: float):
    st = S.get_state()
    if st.kd_tree is not None:
        dist, idx = st.kd_tree.query([lng, lat])
        return {"edge_id": st.kd_ids[idx], "dist_deg": float(dist)}

    # O(N) fallback
    best_eid, best_d2 = None, float("inf")
    for eid, info in st.edge_lookup.items():
        coords = info.get("geom", [])
        mid    = coords[len(coords) // 2] if coords else [0, 0]
        d2     = (mid[0] - lng) ** 2 + (mid[1] - lat) ** 2
        if d2 < best_d2:
            best_d2, best_eid = d2, eid
    if best_eid is None:
        return JSONResponse({"error": "No edges"}, status_code=500)
    return {"edge_id": best_eid, "dist_deg": best_d2 ** 0.5}


@app.get("/edge_info")
def edge_info(edge: str, hour: int = 12):
    st = S.get_state()
    hour = max(0, min(23, hour))

    if st.removal_loaded and st.bpr_state is not None:
        from bpr_engine import get_edge_info
        info = get_edge_info(st.bpr_state, edge, hour)
        if info:
            el = st.edge_lookup.get(edge, {})
            info["confidence"] = el.get("confidence", info.get("confidence", 0.0))
            return info

    el = st.edge_lookup.get(edge)
    if not el:
        return JSONResponse({"error": "Edge not found"}, status_code=404)
    tt  = el["tt"][hour]
    lm  = el["length"]
    return {
        "edge_id":    edge,
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
    }


@app.get("/chain")
def get_chain(edge: str):
    st = S.get_state()
    if not st.removal_loaded or st.chains is None:
        return {"edge_id": edge, "chain": [edge], "chain_len": 1,
                "is_multi": False, "note": "chain service loading"}
    return st.chains.chain_summary(edge)


@app.get("/status")
def status():
    st = S.get_state()
    return {
        "graph_edges":      st.G.ecount() if st.G else 0,
        "graph_nodes":      st.G.vcount() if st.G else 0,
        "movements_loaded": st.movement_router is not None and st.movement_router.loaded,
        "movements_count":  len(st.movement_router._movements) if st.movement_router else 0,
        "removal_loaded":   st.removal_loaded,
        "removal_error":    st.removal_error,
        "chains_loaded":    st.chains is not None,
        "chain_count":      len(st.chains) if st.chains else 0,
        "kdtree_active":    st.kd_tree is not None,
        "strtree_active":   st.flood_engine is not None,
    }


@app.get("/route")
def get_route(
    request: Request,
    response: Response,
    start:     str,
    end:       str,
    hour:      int  = 0,
    blocked:   str  = "",
    movements: bool = True,
):
    st = S.get_state()
    hour = max(0, min(23, hour))
    sid  = _sid(request)
    _set_sid(response, request, sid)

    raw_blocked   = set(filter(None, blocked.split(",")))
    blocked_edges = _expand_blocked(st, raw_blocked)
    el            = st.edge_lookup

    if start not in el or end not in el:
        return JSONResponse({"error": f"Invalid edge IDs: {start} or {end}"}, status_code=400)
    if start in blocked_edges:
        return JSONResponse({"error": "Origin edge is part of a blocked chain"}, status_code=400)
    if end in blocked_edges:
        return JSONResponse({"error": "Destination edge is part of a blocked chain"}, status_code=400)

    start_info = el[start]
    end_info   = el[end]
    tt_override = _session_tt_override(st, sid, hour)

    # ── Movement-aware routing (primary) ─────────────────────────────────────
    if movements and st.movement_router and st.movement_router.loaded and st.movement_router.has_movements:
        result = st.movement_router.route_via_line_graph(
            edge_lookup   = el,
            start_edge    = start,
            end_edge      = end,
            hour          = hour,
            blocked_edges = blocked_edges,
            tt_override   = tt_override,
        )
        if result is not None:
            turn_times = [s["turn_s"] for s in result.get("path_detail", [])]
            return _build_route_response(
                st, result["edges"], hour, tt_override, turn_times,
                "movement_graph", list(blocked_edges),
                result={
                    "time_total_s":    result["total_time_s"],
                    "turn_time_s":     result["turn_time_s"],
                    "n_turns":         result["n_turns"],
                },
            )

    # ── igraph Dijkstra fallback ──────────────────────────────────────────────
    if start == end:
        return _build_route_response(
            st, [start], hour, tt_override, [], "igraph_fallback", list(blocked_edges)
        )

    path_eids = _igraph_route(
        st,
        source_node   = start_info.get("v", ""),
        target_node   = end_info.get("u", ""),
        hour          = hour,
        blocked_edges = blocked_edges,
        tt_override   = tt_override,
    )

    if path_eids is None:
        return JSONResponse({"error": "No path exists between these edges"}, status_code=404)

    edge_sequence = [start] + path_eids + [end]
    return _build_route_response(
        st, edge_sequence, hour, tt_override, [], "igraph_fallback", list(blocked_edges)
    )


@app.get("/removal")
def get_removal(request: Request, response: Response, edge: str, hour: int = 0):
    st  = S.get_state()
    sid = _sid(request)
    _set_sid(response, request, sid)
    hour = max(0, min(23, hour))

    if edge not in st.edge_lookup:
        return JSONResponse({"error": "Invalid edge ID"}, status_code=400)
    if not st.removal_loaded:
        return JSONResponse({"error": "Removal service loading"}, status_code=503)

    from bpr_engine import remove_edge
    r = remove_edge(
        st.bpr_state, edge, hour,
        chain_index    = st.chains,
        chain_graph    = st.chain_graph,
        flows_override = _accumulated_flows(st, sid, hour),
    )
    result  = r.to_dict()
    geojson = r.to_geojson_delta(st.bpr_state, st.edge_lookup)
    T.update_session_tt(sid, hour, result)
    result["delta_geojson"] = geojson
    return result


@app.get("/speed_floor")
def get_speed_floor(
    request: Request, response: Response,
    edge: str, hour: int = 0, min_speed: float = 30.0,
):
    st = S.get_state()
    sid = _sid(request)
    _set_sid(response, request, sid)
    hour = max(0, min(23, hour))

    if edge not in st.edge_lookup:
        return JSONResponse({"error": "Invalid edge ID"}, status_code=400)
    if not st.removal_loaded:
        return JSONResponse({"error": "Removal service loading"}, status_code=503)
    if min_speed <= 0:
        return JSONResponse({"error": "min_speed must be > 0"}, status_code=400)

    from bpr_engine import apply_speed_floor
    r = apply_speed_floor(
        st.bpr_state, edge, hour,
        min_speed_kmh  = min_speed,
        chain_index    = st.chains,
        chain_graph    = st.chain_graph,
        flows_override = _accumulated_flows(st, sid, hour),
    )
    result  = r.to_dict()
    geojson = r.to_geojson_delta(st.bpr_state, st.edge_lookup)
    T.update_session_tt(sid, hour, result)
    result["delta_geojson"]    = geojson
    result["simulation_mode"]  = "speed_floor"
    result["min_speed_kmh"]    = min_speed
    return result


@app.get("/capacity_tune")
def get_capacity_tune(
    request: Request, response: Response,
    edge: str, hour: int = 0, capacity_factor: float = 0.5,
):
    st = S.get_state()
    sid = _sid(request)
    _set_sid(response, request, sid)
    hour            = max(0, min(23, hour))
    capacity_factor = max(0.05, min(1.0, capacity_factor))

    if edge not in st.edge_lookup:
        return JSONResponse({"error": "Invalid edge ID"}, status_code=400)
    if not st.removal_loaded:
        return JSONResponse({"error": "Removal service loading"}, status_code=503)

    from bpr_engine import apply_capacity_tune
    r = apply_capacity_tune(
        st.bpr_state, edge, hour,
        capacity_factor = capacity_factor,
        chain_index     = st.chains,
        chain_graph     = st.chain_graph,
        flows_override  = _accumulated_flows(st, sid, hour),
    )
    result  = r.to_dict()
    geojson = r.to_geojson_delta(st.bpr_state, st.edge_lookup)
    T.update_session_tt(sid, hour, result)
    result["delta_geojson"]   = geojson
    result["simulation_mode"] = "capacity_tune"
    result["capacity_factor"] = capacity_factor
    return result


@app.get("/saturation_corridors")
def get_saturation_corridors(
    request: Request,
    hour: int = 0, vc_threshold: float = 0.85, limit: int = 30,
):
    st = S.get_state()
    sid = _sid(request)
    hour         = max(0, min(23, hour))
    vc_threshold = float(vc_threshold)
    limit        = min(int(limit), 200)

    if not st.removal_loaded:
        return JSONResponse({"error": "Removal service loading"}, status_code=503)

    from bpr_engine import BPR_ALPHA, BPR_BETA
    state = st.bpr_state
    session_tt = T.read_session_tt(sid, hour)
    if not session_tt:
        return {
            "hour": hour, "vc_threshold": vc_threshold,
            "corridors": [], "n_at_risk": 0,
            "note": "No removals applied in this session yet",
        }

    tt_free = state.length_m * 3.6 / np.maximum(state.speed_free, 0.5)
    cap     = state.capacity
    corridors = []
    for idx, tt_val in session_tt.items():
        if idx >= state.N:
            continue
        cap_i  = float(cap[idx])
        if cap_i < 1.0:
            continue
        ttf   = float(tt_free[idx])
        if ttf < 0.01:
            continue
        ratio = float(tt_val) / ttf
        vc_a  = 0.0 if ratio <= 1.0 else ((ratio - 1.0) / BPR_ALPHA) ** (1.0 / BPR_BETA)
        vc_b  = float(state.flows[hour, idx]) / cap_i
        vc_delta = vc_a - vc_b
        if vc_a < vc_threshold and vc_delta < 0.05:
            continue
        corridors.append({
            "edge_id":    state.edge_ids[idx],
            "road_type":  state.road_type[idx],
            "flow_veh_h": round(vc_a * cap_i, 1),
            "capacity":   round(cap_i, 1),
            "vc_ratio":   round(vc_a, 4),
            "vc_delta":   round(vc_delta, 4),
            "tt_delta_s": round(float(tt_val) - float(state.travel_time[hour, idx]), 2),
            "data_source": state.data_source[idx],
        })

    corridors.sort(key=lambda c: -c["vc_ratio"])
    return {
        "hour": hour, "vc_threshold": vc_threshold,
        "n_at_risk": len(corridors), "corridors": corridors[:limit],
    }


@app.get("/restore")
def restore(request: Request, response: Response):
    st  = S.get_state()
    sid = _sid(request)
    _set_sid(response, request, sid)
    T.clear_session(sid)
    for h in list(st.criticality_cache.keys()):
        if h not in st.precomputed_hours:
            st.criticality_cache.pop(h, None)
    return {"ok": True}


@app.get("/turn_info")
def get_turn_info(request: Request, from_: str = "", to: str = ""):
    # FastAPI uses `from_` because `from` is a Python keyword
    from_e = request.query_params.get("from", from_)
    st = S.get_state()
    if not from_e or not to:
        return JSONResponse({"error": "from and to params required"}, status_code=400)
    if st.movement_router is None:
        return {"found": False, "fallback_s": FALLBACK_TURN_S}
    info = st.movement_router.get_turn_info(from_e, to)
    if info is None:
        return {"found": False, "fallback_s": FALLBACK_TURN_S}
    return {"found": True, **info}


@app.get("/criticality")
def get_criticality(hour: int = 8, background_tasks: BackgroundTasks = None):
    st   = S.get_state()
    hour = max(0, min(23, hour))

    cached = st.criticality_cache.get(hour)
    if cached and cached.get("status") == "done":
        return cached

    if not st.removal_loaded:
        return JSONResponse({"error": "Removal service still loading"}, status_code=503)

    if cached and cached.get("status") == "running":
        return {"status": "running", "hour": hour,
                "done": cached.get("done", 0), "total": cached.get("total", 0)}

    state = st.bpr_state
    candidates = [
        state.edge_ids[i]
        for i in range(state.N)
        if state.data_source[i] in ("sensor", "reconstructed")
        and state.nodes_from[i] >= 0
        and float(state.flows[hour, i]) > 5.0
    ]
    st.criticality_cache[hour] = {"status": "running", "hour": hour,
                                   "done": 0, "total": len(candidates), "pct": 0}

    def _run():
        from bpr_engine import remove_edge
        raw = []
        total = len(candidates)
        for i, eid in enumerate(candidates):
            try:
                r = remove_edge(state, eid, hour, chain_index=st.chains,
                                chain_graph=st.chain_graph)
                rp = r.rerouted_flow / max(r.displaced_flow, 1) * 100
                raw.append({
                    "edge_id":           eid,
                    "chain_len":         len(r.chain),
                    "displaced_flow":    r.displaced_flow,
                    "rerouted_pct":      rp,
                    "total_delay_veh_h": r.total_delay / 3600,
                    "n_congested":       int(r.newly_congested.sum()),
                    "warning":           r.warning,
                })
            except Exception:
                pass
            if i % 25 == 0 and hour in st.criticality_cache:
                st.criticality_cache[hour].update(done=i, pct=round(i / total * 100, 1))

        if raw:
            def _norm(arr):
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                mn, mx = arr.min(), arr.max()
                return np.zeros_like(arr) if mx - mn < 1e-9 else (arr - mn) / (mx - mn)
            delays = np.array([s["total_delay_veh_h"] for s in raw], dtype=np.float64)
            flows  = np.array([s["displaced_flow"]     for s in raw], dtype=np.float64)
            cong   = np.array([s["n_congested"]        for s in raw], dtype=np.float64)
            isol   = np.array([1.0 - min(s["rerouted_pct"], 100) / 100 for s in raw], dtype=np.float64)
            comp   = np.clip(0.40*_norm(delays)+0.30*_norm(flows)+0.20*_norm(cong)+0.10*_norm(isol), 0, 1)
            for s, c in zip(raw, comp):
                val = float(c)
                s["criticality"] = round(val, 4) if math.isfinite(val) else 0.0
            raw.sort(key=lambda s: -s["criticality"])

        st.criticality_cache[hour] = {"status": "done", "hour": hour,
                                       "total": len(raw), "scores": raw}
        log.info("Criticality done: %d edges at hour %02d", len(raw), hour)

    if background_tasks:
        background_tasks.add_task(_run)
    else:
        threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "hour": hour, "total": len(candidates), "pct": 0}


# ── Flood ─────────────────────────────────────────────────────────────────────

@app.get("/flood_files")
def flood_files():
    import re
    if not GEOCODED_DIR.exists():
        return JSONResponse({"error": "GEOCODED directory not found", "files": []}, status_code=404)
    pat   = re.compile(r"D(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})\.geojson")
    files = []
    for fn in sorted(f.name for f in GEOCODED_DIR.glob("D*.geojson")):
        m = pat.match(fn)
        if m:
            y, mo, d, hr, mi = m.groups()
            files.append({
                "file":      fn,
                "label":     f"{y}-{mo}-{d} {hr}:{mi}",
                "timestamp": f"{y}{mo}{d}{hr}{mi}",
            })
    return {"files": files, "count": len(files)}


@app.get("/flood_mask")
def flood_mask(file: str, hour: int = 8, threshold: float = 0.0):
    st        = S.get_state()
    hour      = max(0, min(23, hour))
    fp        = GEOCODED_DIR / file.strip()
    if not fp.exists():
        return JSONResponse({"error": f"File not found: {file}"}, status_code=404)

    flooded, flood_geojson = st.flood_engine.detect_all(fp, threshold)
    return {
        "flooded_edges": flooded,
        "n_flooded":     len(flooded),
        "flood_geojson": flood_geojson,
        "file":          file,
        "threshold":     threshold,
        "hour":          hour,
    }


@app.get("/pump_priority")
def pump_priority(
    background_tasks: BackgroundTasks,
    file: str, hour: int = 8, threshold: float = 0.0, limit: int = 20,
):
    st = S.get_state()
    hour  = max(0, min(23, hour))
    limit = min(limit, 100)

    if not st.removal_loaded:
        return JSONResponse({"error": "Removal service loading"}, status_code=503)

    fp = GEOCODED_DIR / file.strip()
    if not fp.exists():
        return JSONResponse({"error": f"File not found: {file}"}, status_code=404)

    flooded, _ = st.flood_engine.detect(fp, threshold, min_depth=0.0)
    flooded_eids = [eid for eid, _ in flooded]

    if not flooded_eids:
        return {"task_id": None, "status": "done",
                "scores": [], "n_flooded": 0, "note": "No flooded edges"}

    task_id   = T.make_task()
    n_flooded = len(flooded_eids)

    def _run():
        try:
            from bpr_engine import remove_edge, get_base_dg
            el = st.edge_lookup

            def _cheap_rank(eid):
                rt        = el.get(eid, {}).get("road_type", "unclassified")
                chain_len = len(st.chains.expand_one(eid)) if st.chains else 1
                return _ROAD_CONN_WEIGHT.get(rt, 0.05) * chain_len

            eids = sorted(flooded_eids, key=_cheap_rank, reverse=True)
            eids = eids[:min(limit * 3, len(eids))]
            base_dg = get_base_dg(st.chain_graph, st.bpr_state, hour)

            def _score_one(eid):
                try:
                    return eid, remove_edge(
                        st.bpr_state, eid, hour,
                        chain_index     = st.chains,
                        chain_graph     = st.chain_graph,
                        k               = 1,
                        prebuilt_chain_dg = base_dg,
                    )
                except Exception:
                    return eid, None

            raw = []
            with ThreadPoolExecutor(max_workers=min(8, len(eids))) as pool:
                for eid, r in pool.map(_score_one, eids):
                    if r is None:
                        continue
                    rp   = r.rerouted_flow / max(r.displaced_flow, 1) * 100
                    rt   = el.get(eid, {}).get("road_type", "unclassified")
                    cw   = _ROAD_CONN_WEIGHT.get(rt, 0.10)
                    isol = 1.0 - min(rp, 100.0) / 100.0
                    raw.append({
                        "edge_id":           eid,
                        "chain_len":         len(r.chain),
                        "road_type":         rt,
                        "displaced_flow":    round(r.displaced_flow, 1),
                        "rerouted_pct":      round(rp, 1),
                        "total_delay_veh_h": round(r.total_delay / 3600, 3),
                        "n_congested":       int(r.newly_congested.sum()),
                        "connectivity":      round(cw * isol, 4),
                        "warning":           r.warning,
                    })

            if not raw:
                T.finish_task(task_id, {"scores": [], "n_flooded": n_flooded, "n_scored": 0})
                return

            def _norm(arr):
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                mn, mx = arr.min(), arr.max()
                return np.zeros_like(arr) if mx - mn < 1e-9 else (arr - mn) / (mx - mn)

            delays = np.array([s["total_delay_veh_h"] for s in raw], dtype=np.float64)
            flows  = np.array([s["displaced_flow"]     for s in raw], dtype=np.float64)
            cong   = np.array([s["n_congested"]        for s in raw], dtype=np.float64)
            isol   = np.array([1.0 - min(s["rerouted_pct"], 100) / 100 for s in raw], dtype=np.float64)
            conn   = np.array([s["connectivity"]       for s in raw], dtype=np.float64)
            comp   = np.clip(
                0.35*_norm(delays)+0.25*_norm(flows)+0.20*_norm(isol)+0.10*_norm(cong)+0.10*_norm(conn),
                0, 1,
            )
            for s, c in zip(raw, comp):
                s["pump_priority"] = round(float(c), 4) if math.isfinite(float(c)) else 0.0
            raw.sort(key=lambda s: -s["pump_priority"])
            T.finish_task(task_id, {
                "scores": raw[:limit], "n_flooded": n_flooded, "n_scored": len(raw),
                "file": file, "hour": hour,
            })
        except Exception as exc:
            T.fail_task(task_id, str(exc))

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "running", "n_flooded": n_flooded}


@app.get("/flood_apply")
def flood_apply(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    file: str, hour: int = 8, threshold: float = 0.0, limit: int = 80,
):
    st = S.get_state()
    hour  = max(0, min(23, hour))
    limit = min(limit, 200)

    if not st.removal_loaded:
        return JSONResponse({"error": "Removal service loading"}, status_code=503)

    fp = GEOCODED_DIR / file.strip()
    if not fp.exists():
        return JSONResponse({"error": f"File not found: {file}"}, status_code=404)

    sid     = _sid(request)
    task_id = T.make_task()
    _set_sid(response, request, sid)

    def _run():
        try:
            from bpr_engine import remove_edge, apply_capacity_tune
            from flood_engine import flood_capacity_factor, FLOOD_DEPTH_LIGHT

            flooded, _ = st.flood_engine.detect(fp, threshold, min_depth=FLOOD_DEPTH_LIGHT)
            if not flooded:
                T.finish_task(task_id, {"hard_blocked": [], "capacity_tuned": [],
                                        "n_hard_blocked": 0, "n_capacity_tuned": 0,
                                        "note": "No edges above depth threshold"})
                return

            flooded.sort(key=lambda x: x[1], reverse=True)
            flooded = flooded[:limit]

            hard_blocked:   list = []
            capacity_tuned: list = []

            for eid, depth in flooded:
                cf = flood_capacity_factor(depth)
                try:
                    flows_ov = _accumulated_flows(st, sid, hour)
                    if cf is None:
                        r = remove_edge(
                            st.bpr_state, eid, hour,
                            chain_index    = st.chains,
                            chain_graph    = st.chain_graph,
                            flows_override = flows_ov, k=1,
                        )
                        T.update_session_tt(sid, hour, r.to_dict())
                        hard_blocked.append({"edge_id": eid, "depth": round(depth, 4),
                                              "chain": r.chain})
                    else:
                        r = apply_capacity_tune(
                            st.bpr_state, eid, hour,
                            capacity_factor = cf,
                            chain_index     = st.chains,
                            chain_graph     = st.chain_graph,
                            flows_override  = flows_ov, k=1,
                        )
                        T.update_session_tt(sid, hour, r.to_dict())
                        capacity_tuned.append({"edge_id": eid, "depth": round(depth, 4),
                                               "capacity_factor": cf, "chain": r.chain})
                except Exception as exc:
                    log.warning("flood_apply %s: %s", eid, exc)

            T.finish_task(task_id, {
                "hard_blocked":     hard_blocked,
                "capacity_tuned":   capacity_tuned,
                "n_hard_blocked":   len(hard_blocked),
                "n_capacity_tuned": len(capacity_tuned),
                "hour": hour, "file": file, "threshold": threshold,
            })
        except Exception as exc:
            T.fail_task(task_id, str(exc))

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "running"}


@app.get("/task/{task_id}")
def get_task(task_id: str):
    task = T.get_task(task_id)
    if task is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    resp = {"task_id": task_id, "status": task["status"]}
    if task["status"] == "done" and task["result"]:
        resp.update(task["result"])
    elif task["status"] == "error":
        resp["error"] = task["error"]
    return resp


if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run(
        "main:app",
        host   = "0.0.0.0",
        port   = int(os.environ.get("PORT", 5001)),
        reload = False,
        workers= 1,
    )
