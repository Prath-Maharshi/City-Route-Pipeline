"""
flood_engine.py — Shapely STRtree-based flood detection.

Replaces the grid + point-in-polygon approach from route_app.py.
STRtree gives O(log N_edges) candidate lookup instead of O(N_grid_cells).
Shapely geometry ops are C-level (GEOS), much faster than Python ray-casting.
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

from shapely.geometry import LineString, Polygon, shape
from shapely.strtree import STRtree

log = logging.getLogger("flood_engine")

# Depth thresholds (metres)
FLOOD_DEPTH_HARD  = 0.50
FLOOD_DEPTH_MED   = 0.30
FLOOD_DEPTH_LIGHT = 0.10


class FloodEngine:
    """
    Holds a Shapely STRtree over all network edge LineStrings.

    Built once at startup from edge_lookup.  Flood queries use the STRtree
    for O(log N) spatial pre-filtering, then exact Shapely intersects() for
    precise hit-testing — all at C (GEOS) speed.
    """

    def __init__(self, edge_lookup: dict):
        eids:  list[str]       = []
        geoms: list[LineString] = []
        for eid, info in edge_lookup.items():
            coords = info.get("geom", [])
            if len(coords) >= 2:
                geoms.append(LineString([(c[0], c[1]) for c in coords]))
                eids.append(eid)

        self._tree = STRtree(geoms)
        self._eids = eids      # parallel to tree.geometries

        log.info("FloodEngine: STRtree over %d edges", len(eids))

    def detect(
        self,
        fp: Path,
        threshold: float,
        min_depth: float = FLOOD_DEPTH_LIGHT,
    ) -> tuple[list[tuple[str, float]], dict]:
        """
        Return [(eid, max_depth), ...] for edges that intersect flood polygons
        above threshold with max_depth >= min_depth.

        Also returns the raw GeoJSON dict for map overlay.
        """
        polygons, depths, raw_fc = _load_flood_polygons(fp, threshold)
        if not polygons:
            return [], raw_fc

        flooded: list[tuple[str, float]] = []
        for i, poly in enumerate(polygons):
            depth = depths[i]
            if depth < min_depth:
                continue
            # Find edge candidates whose bbox overlaps this polygon
            for idx in self._tree.query(poly):
                edge_geom = self._tree.geometries[idx]
                if edge_geom.intersects(poly):
                    eid = self._eids[idx]
                    flooded.append((eid, depth))

        # Deduplicate: keep max depth per edge
        best: dict[str, float] = {}
        for eid, depth in flooded:
            if depth > best.get(eid, -1):
                best[eid] = depth

        return [(eid, depth) for eid, depth in best.items()], raw_fc

    def detect_all(
        self,
        fp: Path,
        threshold: float,
    ) -> tuple[list[dict], dict]:
        """
        Like detect() but returns all edges above threshold (any depth > 0),
        used by /flood_mask which doesn't filter by min_depth.
        """
        polygons, depths, raw_fc = _load_flood_polygons(fp, threshold)
        if not polygons:
            return [], raw_fc

        best: dict[str, float] = {}
        for i, poly in enumerate(polygons):
            depth = depths[i]
            for idx in self._tree.query(poly):
                if self._tree.geometries[idx].intersects(poly):
                    eid = self._eids[idx]
                    if depth > best.get(eid, -1):
                        best[eid] = depth

        result = [{"edge_id": eid, "depth": round(d, 5)} for eid, d in best.items()]
        return result, raw_fc


@functools.lru_cache(maxsize=8)
def _parse_flood_file(fp: Path) -> tuple[list[Polygon], list[float], dict]:
    """Parse GeoJSON flood file and cache ALL polygons (threshold-independent)."""
    with open(fp) as f:
        fc = json.load(f)

    polygons: list[Polygon] = []
    depths:   list[float]   = []

    for feat in fc.get("features", []):
        if feat["geometry"]["type"] != "Polygon":
            continue
        props = feat["properties"]
        depth = 0.0
        for k, v in props.items():
            if k != "geo_code" and isinstance(v, (int, float)):
                depth = float(v)
                break
        polygons.append(shape(feat["geometry"]))
        depths.append(depth)

    return polygons, depths, fc


def _load_flood_polygons(
    fp: Path, threshold: float
) -> tuple[list[Polygon], list[float], dict]:
    """Filter cached polygons by threshold at call time."""
    all_polys, all_depths, fc = _parse_flood_file(fp)
    polygons = [p for p, d in zip(all_polys, all_depths) if d >= threshold]
    depths   = [d for d in all_depths if d >= threshold]
    return polygons, depths, fc


def flood_capacity_factor(depth: float) -> float | None:
    """Map flood depth → capacity_factor, or None for hard block.

    Uses a continuous piecewise-linear model instead of discrete steps,
    avoiding sudden routing shifts when a polygon boundary barely crosses
    a threshold.  Capacity drops linearly from 1.0 at 0 m to 0.10 at
    FLOOD_DEPTH_HARD, with a hard block above that.
    """
    if depth >= FLOOD_DEPTH_HARD:
        return None   # full remove_edge
    if depth < FLOOD_DEPTH_LIGHT:
        return 1.0
    # Linear interpolation: 1.0 at FLOOD_DEPTH_LIGHT → 0.10 at FLOOD_DEPTH_HARD
    t = (depth - FLOOD_DEPTH_LIGHT) / (FLOOD_DEPTH_HARD - FLOOD_DEPTH_LIGHT)
    return round(max(0.10, 1.0 - t * 0.90), 4)
