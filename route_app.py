"""
app.py  —  Gurugram Traffic Routing (movement-aware)
=====================================================
Integrates:
  - movement_router.py  : turn-cost routing via line graph
  - edge_removal.py     : flow redistribution under edge removal

Startup sequence (all background threads):
  1. load_data()               — GeoJSON (synchronous, fast)
  2. net.xml read              — sumolib (synchronous)
  3. G (nx DiGraph)            — built from GeoJSON + sumolib
  4. movement_router.load()    — movements.pkl
  5. removal_service.load()    — traffic arrays pkl
  6. removal_service.set_nx_graph(G)
"""

import json
import threading
import networkx as nx
from flask import Flask, render_template, request, jsonify
import sumolib
from edge_removal import EdgeRemovalService
from movement_router import MovementRouter

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
GEOJSON_PATH   = "outputs/graph_reconstruction/traffic_state_scored.geojson"
NET_PATH       = "outputs/networks/full.net.xml"
ARRAYS_PKL     = "outputs/graph_reconstruction/gurugram_traffic_arrays.pkl"
MOVEMENTS_PKL  = "outputs/networks/movements.pkl"

# ── Services ──────────────────────────────────────────────────────────────────
removal_service = EdgeRemovalService(ARRAYS_PKL, NET_PATH)
movement_router = MovementRouter(MOVEMENTS_PKL, fallback_penalty_s=5.0)

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
G          = nx.DiGraph()
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

    length    = props.get("length_m",           1.0)
    road_type = props.get("road_type",   "unclassified")
    confidence = props.get("confidence",         0.1)
    dom_dir   = props.get("dominant_direction", "unknown")

    G.add_edge(u, v, id=eid, tt=tt_arr, tt_low=tt_low, tt_high=tt_high,
               length=length, geom=coords, road_type=road_type,
               confidence=confidence, dom_dir=dom_dir)

    # In the edge_lookup assignment block, add "to_node":
    edge_lookup[eid] = {
        "u": u, "v": v, "to_node": v,   # ← add this
        "geom": coords,
        "tt": tt_arr, "tt_low": tt_low, "tt_high": tt_high,
        "length": length, "road_type": road_type,
        "confidence": confidence, "dom_dir": dom_dir,
    }

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# ── Background service startup ────────────────────────────────────────────────
# Load movements first (fast), then the heavy traffic arrays
threading.Thread(target=movement_router.load, daemon=True).start()

def _load_removal_service():
    removal_service.load()
    removal_service.set_nx_graph(G)

threading.Thread(target=_load_removal_service, daemon=True).start()


# ── Routing constants ─────────────────────────────────────────────────────────
ROAD_PENALTY = {
    "motorway": 0.6,    "trunk": 0.7,         "trunk_link": 0.7,
    "primary": 0.85,    "secondary": 1.0,     "secondary_link": 1.0,
    "tertiary": 1.2,    "tertiary_link": 1.2,
    "unclassified": 1.4, "residential": 1.6,
}
FALLBACK_TURN_PENALTY_S = 5   # align with movement_router fallback_penalty_s order of magnitude  


def _impedance_fallback(u, v, edge_data, hour, blocked_edges=None):
    """
    Original flat-penalty impedance — used when movement router is not loaded.
    """
    if blocked_edges and edge_data["id"] in blocked_edges:
        return float("inf")
    base_t = edge_data["tt"][hour]
    type_m = ROAD_PENALTY.get(edge_data.get("road_type", "unclassified"), 1.5)
    conf_m = 1.0 + (1.0 - edge_data.get("confidence", 0.1)) * 0.2
    return base_t * type_m * conf_m + FALLBACK_TURN_PENALTY_S


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/geojson")
def serve_geojson():
    return jsonify(data)

@app.route("/status")
def status():
    return jsonify({
        "graph_edges":       G.number_of_edges(),
        "graph_nodes":       G.number_of_nodes(),
        "movements_loaded":  movement_router.loaded,
        "movements_count":   len(movement_router._movements),
        "removal_loaded":    removal_service.loaded,
    })


@app.route("/route")
def get_route():
    start_id = request.args.get("start")
    end_id   = request.args.get("end")
    blocked_str   = request.args.get("blocked", "")
    blocked_edges = set(filter(None, blocked_str.split(",")))

    try:
        hour = max(0, min(23, int(request.args.get("hour", 0))))
    except ValueError:
        hour = 0

    use_movements = request.args.get("movements", "true").lower() == "true"

    if start_id not in edge_lookup or end_id not in edge_lookup:
        return jsonify({"error": f"Invalid edge IDs: {start_id} or {end_id}"}), 400

    start_data = edge_lookup[start_id]
    end_data   = edge_lookup[end_id]

    # ── Try movement-aware routing first ──────────────────────────────────────
    if use_movements and movement_router.loaded and movement_router.has_movements:
        result = movement_router.route_via_line_graph(
            G_nx          = G,
            edge_lookup   = edge_lookup,
            start_edge    = start_id,
            end_edge      = end_id,
            hour          = hour,
            blocked_edges = blocked_edges,
        )

        if result is not None:
            # Build route geometry from edge sequence
            route_coords = []
            for eid in result["edges"]:
                ed = edge_lookup.get(eid, {})
                geom = ed.get("geom", [])
                if not route_coords:
                    route_coords.extend(geom)
                else:
                    route_coords.extend(geom[1:])

            return jsonify({
                "geometry":        {"type": "LineString", "coordinates": route_coords},
                "time_range_s":    [result["total_time_low_s"], result["total_time_high_s"]],
                "time_expected_s": result["expected_time_s"],
                "time_total_s":    result["total_time_s"],
                "turn_time_s":     result["turn_time_s"],
                "n_turns":         result["n_turns"],
                "dist_m":          round(result["dist_m"], 2),
                "edges":           result["edges"],
                "routing_mode":    "movement_graph",
            })
        # Fall through to nx fallback if line-graph found no path

    # ── Fallback: standard NetworkX Dijkstra ──────────────────────────────────
    try:
        if start_id == end_id:
            path_edges    = [start_data]
            route_coords  = list(start_data["geom"])
            edge_sequence = [start_id]
        else:
            path_nodes = nx.shortest_path(
                G, start_data["v"], end_data["u"],
                weight=lambda u, v, d: _impedance_fallback(
                    u, v, d, hour, blocked_edges
                ),
            )
            path_edges    = [start_data]
            route_coords  = list(start_data["geom"])
            edge_sequence = [start_id]

            for i in range(len(path_nodes) - 1):
                u, v    = path_nodes[i], path_nodes[i + 1]
                edata   = G[u][v]
                path_edges.append(edata)
                route_coords.extend(edata["geom"][1:])
                edge_sequence.append(edata["id"])

            path_edges.append(end_data)
            route_coords.extend(end_data["geom"][1:])
            edge_sequence.append(end_id)

        # Aggregate metrics
        total_low = total_high = total_exp = total_dist = 0.0
        for e in path_edges:
            base_t = e["tt"][hour];  low_t = e["tt_low"][hour]; high_t = e["tt_high"][hour]
            ddir   = e.get("dom_dir", "unknown")
            total_low  += low_t;  total_high += high_t
            total_dist += e["length"]
            if ddir == "underestimate":
                adj = base_t + 0.75 * (high_t - base_t)
            elif ddir == "overestimate":
                adj = base_t - 0.75 * (base_t - low_t)
            else:
                adj = base_t
            total_exp += max(low_t, min(high_t, adj))

        n_trans     = max(0, len(path_edges) - 1)
        turn_total  = n_trans * FALLBACK_TURN_PENALTY_S
        total_low  += turn_total; total_high += turn_total; total_exp += turn_total

        return jsonify({
            "geometry":        {"type": "LineString", "coordinates": route_coords},
            "time_range_s":    [round(total_low, 2), round(total_high, 2)],
            "time_expected_s": round(total_exp, 2),
            "dist_m":          round(total_dist, 2),
            "edges":           edge_sequence,
            "routing_mode":    "nx_fallback",
        })

    except nx.NetworkXNoPath:
        return jsonify({"error": "No path exists between these edges"}), 404
    except nx.NodeNotFound:
        return jsonify({"error": "Disconnected network node"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/removal")
def get_removal():
    edge_id = request.args.get("edge")
    hour    = max(0, min(23, int(request.args.get("hour", 0))))

    if not edge_id or edge_id not in edge_lookup:
        return jsonify({"error": "Invalid edge ID"}), 400
    if not removal_service.loaded:
        return jsonify({"error": "Removal service loading — please wait"}), 503

    result  = removal_service.simulate(edge_id=edge_id, hour=hour)
    geojson = removal_service.simulate_geojson(edge_id=edge_id, hour=hour)

    if "error" not in geojson:
        result["delta_geojson"] = geojson
    else:
        result["delta_geojson"] = {"type": "FeatureCollection", "features": []}

    return jsonify(result)


@app.route("/turn_info")
def get_turn_info():
    """Debug endpoint — return turn info for a (from, to) edge pair."""
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

@app.route("/route",    methods=["OPTIONS"])
@app.route("/removal",  methods=["OPTIONS"])
@app.route("/turn_info",methods=["OPTIONS"])
def preflight():
    return "", 204


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, threaded=True, use_reloader=False)