"""Graph-context loader for the M5 link-pred service: namespace tunnel graph + pooled-qwen Room vectors.

The serving scorer needs a per-namespace CONTEXT:
  - Room<->Room edges from `memory_tunnel` (datamodel/memory_tunnel.go, migration 000169): columns
    from_room_uid / to_room_uid / weight / edge_class ("backbone"|"overlay"), keyed by (namespace_uid, user_uid).
  - a per-Room embedding pooled from that Room's Item qwen vectors (the M1 namespace vector space in Milvus).
    M5 stores NO per-Room vector table (ADR-0010 §heterogeneity), so Room vectors are DERIVED here by pooling.

Split:
  - pool_room_vectors / build_context / load_namespace_context — PURE, unit-tested below.
  - SyntheticFetcher — for tests/offline.
  - PostgresMilvusFetcher — the real source (SELECT-only memory_tunnel + Milvus pooling); the SQL is grounded
    in the datamodel, but the Milvus collection/membership wiring must be VALIDATED against a live stack
    before prod use (greenfield now → nothing to read).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def pool_room_vectors(room_to_item_vecs, method="normmean"):
    """{room_uid: [item_vec, ...]} -> {room_uid: pooled L2-normalized Room vector}. Empty rooms dropped."""
    out = {}
    for room, vecs in room_to_item_vecs.items():
        if vecs is None or len(vecs) == 0:
            continue
        V = np.asarray(vecs, dtype=np.float32)
        if method == "mean":
            p = V.mean(0)
        elif method == "normmean":                       # unit-normalize items, then mean (robust to scale)
            p = (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)).mean(0)
        else:
            raise ValueError(f"unknown pool method {method}")
        n = float(np.linalg.norm(p))
        out[room] = (p / n) if n > 1e-9 else p           # L2-normalize the Room vector (cosine-ready)
    return out


@dataclass
class Context:
    num_nodes: int
    edges: np.ndarray          # (E, 2) int node indices
    node_emb: np.ndarray       # (N, D) float32 pooled Room vectors (zeros for rooms with no items)
    idx_of: dict               # room_uid -> node index
    rooms: list                # node index -> room_uid


def build_context(tunnel_edges, room_vectors, dim=None):
    """tunnel_edges: iterable of (from_room, to_room[, weight, ...]); room_vectors: {room: vec}. -> Context."""
    edge_pairs = [(e[0], e[1]) for e in tunnel_edges]
    rooms = sorted({r for p in edge_pairs for r in p} | set(room_vectors), key=str)
    idx = {r: i for i, r in enumerate(rooms)}
    N = len(rooms)
    if dim is None:
        dim = len(next(iter(room_vectors.values()))) if room_vectors else 1
    emb = np.zeros((N, dim), dtype=np.float32)
    for r, v in room_vectors.items():
        emb[idx[r]] = v
    edges = (np.array([[idx[a], idx[b]] for a, b in edge_pairs], dtype=np.int64)
             if edge_pairs else np.zeros((0, 2), dtype=np.int64))
    return Context(N, edges, emb, idx, rooms)


def load_namespace_context(fetcher, namespace_uid, user_uid, pool="normmean"):
    edges = fetcher.tunnel_edges(namespace_uid, user_uid)
    room_vecs = pool_room_vectors(fetcher.room_item_vectors(namespace_uid, user_uid), method=pool)
    return build_context(edges, room_vecs)


class SyntheticFetcher:
    """In-memory fetcher for tests/offline (no stack)."""
    def __init__(self, edges, room_item_vecs):
        self._edges, self._vecs = edges, room_item_vecs

    def tunnel_edges(self, ns, user):
        return self._edges

    def room_item_vectors(self, ns, user):
        return self._vecs


class PostgresMilvusFetcher:
    """Real source. SELECT-only `memory_tunnel` + Milvus Item-vector pooling. VALIDATE against a live stack
    (collection/field names + Item->Room membership) before prod use — greenfield now, so untested here."""
    def __init__(self, pg_conn, milvus, edge_classes=None):
        # edge_classes=None → no class filter (cross-schema safe: `edge_class` is added by migration 000169;
        # older sandboxes — e.g. primary at v164 — lack the column. Pass ("backbone","overlay") only on a
        # migration-169+ stack). Validated: the core 3-column query runs against the live `agent.memory_tunnel`.
        self.pg, self.milvus, self.edge_classes = pg_conn, milvus, edge_classes

    def tunnel_edges(self, namespace_uid, user_uid):
        # SELECT-only, grounded in datamodel/memory_tunnel.go + verified against the live `agent` DB schema.
        sql = ("SELECT from_room_uid, to_room_uid, weight FROM memory_tunnel "
               "WHERE namespace_uid = %s AND user_uid = %s")
        params = [namespace_uid, user_uid]
        if self.edge_classes:
            sql += " AND edge_class = ANY(%s)"
            params.append(list(self.edge_classes))
        with self.pg.cursor() as cur:
            cur.execute(sql, params)
            return [(str(a), str(b), float(w)) for a, b, w in cur.fetchall()]

    def room_item_vectors(self, namespace_uid, user_uid):
        # TODO(validate-on-stack): group the namespace vector space by Room (collection) membership and pull
        # each member Item's qwen vector. Shape: {room_uid: [item_vec, ...]}. The Milvus collection/field
        # names + the Item->collection membership query are stack-specifics to confirm before prod.
        raise NotImplementedError("wire to the M1 Milvus namespace vector space + collection membership")


if __name__ == "__main__":
    # --- unit checks: pooling + assembly are pure and correct ---
    rng = np.random.default_rng(0)
    # pooling: mean of two unit vectors, then normalized
    rv = pool_room_vectors({"r1": [[3.0, 0.0], [0.0, 0.0]], "r2": [[1.0, 0.0], [0.0, 1.0]], "empty": []})
    assert "empty" not in rv, "empty room must be dropped"
    assert abs(np.linalg.norm(rv["r1"]) - 1.0) < 1e-5, "room vector must be L2-normalized"
    assert np.allclose(rv["r2"], [0.70710677, 0.70710677], atol=1e-5), f"normmean wrong: {rv['r2']}"
    # assembly: 3 rooms, 2 edges, dims propagate, indices consistent
    D = 64
    room_vecs = {f"room-{i}": rng.standard_normal(D).astype(np.float32) for i in range(3)}
    ctx = build_context([("room-0", "room-1", 1.0), ("room-1", "room-2", 2.0)], room_vecs)
    assert ctx.num_nodes == 3 and ctx.edges.shape == (2, 2) and ctx.node_emb.shape == (3, D)
    assert ctx.edges.max() < ctx.num_nodes and set(ctx.idx_of) == set(room_vecs)
    # end-to-end via the synthetic fetcher
    fetch = SyntheticFetcher([("room-0", "room-1"), ("room-1", "room-2")],
                             {f"room-{i}": [rng.standard_normal(D), rng.standard_normal(D)] for i in range(3)})
    ctx2 = load_namespace_context(fetch, "ns-x", "user-y")
    assert ctx2.num_nodes == 3 and abs(np.linalg.norm(ctx2.node_emb[0]) - 1.0) < 1e-5
    print(f"context.py self-checks PASS — pooling L2-normalized, assembly consistent "
          f"(N={ctx2.num_nodes}, E={ctx2.edges.shape[0]}, D={ctx2.node_emb.shape[1]})")
