"""
chain_utils.py
==============
Utilities for chain-aware edge blocking.

A "chain" is a sequence of SUMO edge fragments that form a 1-in-1-out
corridor (no branching junctions at internal nodes).  When one fragment
is blocked the entire chain must be blocked — traffic cannot enter or
exit mid-chain because there are no alternative junctions.

chains.pkl format (produced by graph_reconstruction.py Stage 3.5):
    List[List[str]]   — each inner list is one chain of edge IDs

    Example:
        [
            ["-123456789#0", "-123456789#1", "-123456789#2"],
            ["987654321#0"],          # singleton chain
            ...
        ]

Public API
----------
load_chains(pkl_path)          -> ChainIndex
ChainIndex.expand(edge_ids)    -> frozenset[str]   all chain members
ChainIndex.chain_of(edge_id)   -> list[str] | None
ChainIndex.representative(edge_id) -> str          first member (canonical)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("chain_utils")


class ChainIndex:
    """
    Bidirectional index: edge_id ↔ chain.

    Attributes
    ----------
    chains      : list of chains (each chain is a list of edge IDs)
    edge_to_chain : dict mapping every edge_id → its chain list
    """

    def __init__(self, chains: list[list[str]]):
        self.chains: list[list[str]] = chains
        self.edge_to_chain: dict[str, list[str]] = {}
        for chain in chains:
            for eid in chain:
                self.edge_to_chain[eid] = chain

        n_edges  = sum(len(c) for c in chains)
        n_multi  = sum(1 for c in chains if len(c) > 1)
        log.info(
            "ChainIndex: %d chains (%d multi-edge), %d total edges indexed",
            len(chains), n_multi, n_edges,
        )

    # ── Core lookups ──────────────────────────────────────────────────

    def chain_of(self, edge_id: str) -> Optional[list[str]]:
        """Return the full chain containing edge_id, or None if unknown."""
        return self.edge_to_chain.get(edge_id)

    def representative(self, edge_id: str) -> str:
        """
        Canonical representative of the chain (first member).
        Falls back to edge_id itself if not in any chain.
        """
        chain = self.edge_to_chain.get(edge_id)
        return chain[0] if chain else edge_id

    # ── Expansion ─────────────────────────────────────────────────────

    def expand_one(self, edge_id: str) -> list[str]:
        """
        Return all edge IDs in the same chain as edge_id.
        If edge_id is not in any indexed chain, returns [edge_id].
        """
        chain = self.edge_to_chain.get(edge_id)
        return chain if chain else [edge_id]

    def expand(self, edge_ids: Iterable[str]) -> frozenset[str]:
        """
        Given a set of edge IDs (e.g. blocked edges), return the full
        closure: every edge in any chain that contains at least one of
        the input edges.

        This is the main entry point for routing and removal.
        """
        result: set[str] = set()
        for eid in edge_ids:
            result.update(self.expand_one(eid))
        return frozenset(result)

    # ── Diagnostics ───────────────────────────────────────────────────

    def chain_summary(self, edge_id: str) -> dict:
        chain = self.expand_one(edge_id)
        return {
            "edge_id":    edge_id,
            "chain_len":  len(chain),
            "chain":      chain,
            "is_multi":   len(chain) > 1,
        }

    def __len__(self) -> int:
        return len(self.chains)

    def __contains__(self, edge_id: str) -> bool:
        return edge_id in self.edge_to_chain


class ChainGraph:
    """
    Chain-collapsed routing graph for fast Yen's k-shortest-paths.

    Internally a MultiDiGraph where each edge = one chain (keyed by chain rep =
    first edge ID).  Nodes are SUMO junction IDs for chain entry/exit points only;
    pass-through (internal) chain nodes are absent — shrinking the graph Dijkstra
    must traverse.

    Singletons (edges not covered by any multi-edge chain) are also added so the
    chain graph is a complete superset of the full road network topology.

    Usage
    -----
    cg = ChainGraph(chain_index, state)          # build once at startup
    DG = cg.as_digraph(blocked_eids, tt_current) # O(CG_E) view per query
    """

    def __init__(self, chain_index: "ChainIndex", state):
        import networkx as nx

        self._cg: "nx.MultiDiGraph" = nx.MultiDiGraph()
        # physical edge_id → chain representative (first eid in chain)
        self.eid_to_rep: dict[str, str] = {}

        for chain in chain_index.chains:
            self._add_chain(chain, state)

        # Any network edges not covered by the chain index → singleton chains
        indexed = set(chain_index.edge_to_chain.keys())
        for eid in state.edge_ids:
            if eid not in indexed:
                self._add_chain([eid], state)

        log.info(
            "ChainGraph: %d nodes, %d chain-edges (from %d network edges)",
            self._cg.number_of_nodes(),
            self._cg.number_of_edges(),
            len(state.edge_ids),
        )

    def _add_chain(self, chain: list, state) -> None:
        hi = state.edge_index.get(chain[0])
        ti = state.edge_index.get(chain[-1])
        if hi is None or ti is None:
            return
        u_n = int(state.nodes_from[hi])
        v_n = int(state.nodes_to[ti])
        if u_n < 0 or v_n < 0:
            return
        u_str = state.node_ids[u_n]
        v_str = state.node_ids[v_n]
        chain_idxs = [state.edge_index[e] for e in chain if e in state.edge_index]
        if not chain_idxs:
            return
        rep = chain[0]
        self._cg.add_edge(u_str, v_str, key=rep, chain_idxs=chain_idxs)
        for eid in chain:
            self.eid_to_rep[eid] = rep

    def as_digraph(self, blocked_eids: list, tt_current) -> "nx.DiGraph":
        """
        Build a DiGraph with blocked chains removed.

        For parallel chains between the same junction pair, keeps the
        lowest travel-time one.  O(CG_E) — fast because CG_E << full G edges.
        The returned DiGraph is safe to pass to nx.shortest_simple_paths.
        """
        import networkx as nx

        blocked_reps = {self.eid_to_rep[e] for e in blocked_eids if e in self.eid_to_rep}
        DG: "nx.DiGraph" = nx.DiGraph()
        DG.add_nodes_from(self._cg.nodes())
        for u, v, key, data in self._cg.edges(keys=True, data=True):
            if key in blocked_reps:
                continue
            idxs = data.get("chain_idxs", [])
            tt = sum(float(tt_current[i]) for i in idxs) if idxs else 30.0
            if not DG.has_edge(u, v) or tt < DG[u][v]["tt"]:
                DG.add_edge(u, v, tt=tt, chain_idxs=idxs)
        return DG

    def build_base_digraph(self, tt_current) -> "nx.DiGraph":
        """
        Build the full unblocked DiGraph, storing _chain_rep on every edge.

        Call this once when tt_current is stable (e.g. base hour state with no
        session accumulation).  Reuse the result across many per-edge queries by
        passing it to block_chains_in(), which wraps it in an O(1) restricted
        view rather than rebuilding O(CG_E) each time.
        """
        import networkx as nx

        DG: "nx.DiGraph" = nx.DiGraph()
        DG.add_nodes_from(self._cg.nodes())
        for u, v, key, data in self._cg.edges(keys=True, data=True):
            idxs = data.get("chain_idxs", [])
            tt = sum(float(tt_current[i]) for i in idxs) if idxs else 30.0
            if not DG.has_edge(u, v) or tt < DG[u][v]["tt"]:
                DG.add_edge(u, v, tt=tt, chain_idxs=idxs, _chain_rep=key)
        return DG

    def block_chains_in(self, base_dg, blocked_eids) -> "nx.DiGraph":
        """
        Return a restricted view of base_dg with the chains containing
        blocked_eids hidden.  O(1) graph creation — no copy of base_dg.

        base_dg must have been produced by build_base_digraph() so that each
        edge carries a _chain_rep attribute used for identification.
        Multiple threads may call this concurrently on the same base_dg safely
        because restricted_view is read-only.
        """
        import networkx as nx

        blocked_reps = {self.eid_to_rep[e] for e in blocked_eids if e in self.eid_to_rep}
        if not blocked_reps:
            return base_dg
        edges_to_hide = frozenset(
            (u, v)
            for u, v, data in base_dg.edges(data=True)
            if data.get("_chain_rep") in blocked_reps
        )
        if not edges_to_hide:
            return base_dg
        return nx.restricted_view(base_dg, set(), edges_to_hide)


def load_chains(pkl_path: str) -> ChainIndex:
    """
    Load chains.pkl and return a ChainIndex.

    The pickle may contain either:
      - List[List[str]]                    — plain chain lists
      - dict with key "chains"             — wrapped in a dict
      - dict mapping chain_id → List[str] — id-keyed dict

    All formats are handled gracefully.
    """
    path = Path(pkl_path)
    if not path.exists():
        log.warning("chains.pkl not found at %s — chain expansion disabled", pkl_path)
        return ChainIndex([])  # empty index → expand() is a no-op

    with open(path, "rb") as f:
        raw = pickle.load(f)

    chains = _normalise_chains(raw)
    log.info("Loaded %d chains from %s", len(chains), pkl_path)
    return ChainIndex(chains)


def _normalise_chains(raw) -> list[list[str]]:
    """
    Accept the actual chains.pkl format from graph_reconstruction.py and
    return List[List[str]].

    Actual format (confirmed from Stage 3.5 / Stage 8.5 code):
        {"chains": {chain_id: [edge_id, edge_id, ...], ...}}

    Also handles fallback formats gracefully.
    """
    # Format 1 (actual): {"chains": {cid: [eids], ...}}
    if isinstance(raw, dict) and "chains" in raw:
        inner = raw["chains"]
        if isinstance(inner, dict):
            out = []
            for v in inner.values():
                if isinstance(v, (list, tuple)) and v:
                    out.append([str(e) for e in v])
            return out
        # "chains" value is itself a list-of-lists
        return _normalise_chains(inner)

    # Format 2: plain dict mapping cid → [eids]  (no "chains" wrapper)
    if isinstance(raw, dict):
        out = []
        for v in raw.values():
            if isinstance(v, (list, tuple)) and v:
                out.append([str(e) for e in v])
        return out

    # Format 3: plain list of lists  [[eid, ...], ...]
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, (list, tuple)) and item:
                out.append([str(e) for e in item])
        return out

    log.error("Unrecognised chains.pkl format: %s", type(raw))
    return []
