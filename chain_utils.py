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
