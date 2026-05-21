"""D#174 М8 P4 — Hierarchical community detection output.

Builds 3-level hierarchy from flat Leiden communities and writes
`graphify-out/community-hierarchy.json` for downstream consumers
(LEELA worker.js `shiva_graph_query` with `level={leaf, community, root}`).

Pipeline integration:
    detect → extract → build_graph → cluster → hierarchy → analyze → ...
                                              ^^^^^^^^^
                                       this module (NEW · D#174)

Levels:
    leaf      — flat Leiden communities (output of cluster.cluster())
    community — meta-communities derived from a second Leiden pass on
                the community-graph (each node = one leaf community, edges
                weighted by inter-community edges in the original graph)
    root      — single aggregate over all communities

Output schema (graphify-out/community-hierarchy.json):
    {
      "version": "1",
      "metadata": {
        "node_count": int,
        "edge_count": int,
        "leaf_count": int,
        "meta_community_count": int,
        "algorithm": "leiden",
        "resolution_leaf": float,
        "resolution_community": float,
        "graph_sha256": str        # content-addressed cache key
      },
      "levels": {
        "leaf": {
          "<community_id>": {
            "nodes": [node_id, ...],
            "label": "<top-degree-node-label or anonymous>",
            "size": int,
            "cohesion": float       # intra-community edge density
          }
        },
        "community": {
          "<meta_community_id>": {
            "children": ["<leaf_community_id>", ...],
            "label": "<top-leaf-label or anonymous>",
            "node_count": int
          }
        },
        "root": {
          "all": {
            "summary": "<text representation of mandala structure>",
            "community_count": int,
            "node_count": int,
            "top_communities": [
              {"id": "<leaf_id>", "label": "<label>", "size": int}, ...
            ]
          }
        }
      }
    }

Determinism:
    - `random_seed=42` passed to Leiden via cluster._partition()
    - Community IDs sorted; stable IDs across runs (per cluster.py)
    - `graph_sha256` computed over canonicalised JSON of input graph

Concurrency (flock(2) race-mechanism · D#174 cycle-1 spec v0.12 §6):
    - Lock file at `<out_dir>/.lock` opened in `'a+'` mode (preserves prior
      lock-token content · does NOT truncate · per cycle-3 MINOR-2 corrigenda)
    - Write to community-hierarchy.json happens INSIDE the `with`-lock block
      (NOT after lock release · per cycle-3 MINOR-1 corrigenda)
    - On Windows (where fcntl.flock is unavailable) falls back to msvcrt
      file locking with a single-byte byte-range lock; semantically equivalent.

Cycle-1 spec reference:
    chitta/parikshya/2026-05-17-D174-m8-p4-graphrag-leiden-cycle-1-spec.md §2.1 + §6
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx

from graphify.cluster import _partition, cohesion_score

# Default resolution for the meta-community pass. Lower than leaf-level (1.0)
# to encourage fewer, larger meta-communities per spec §2.1 hierarchy-depth note.
_DEFAULT_RESOLUTION_COMMUNITY = 0.5

# Top-N communities surfaced in root summary text (token-budget-bounded).
_ROOT_SUMMARY_TOP_N = 20


# ─── Hierarchy builder ────────────────────────────────────────────────────────


def build_hierarchy(
    G: nx.Graph,
    communities: dict[int, list[str]],
    resolution_community: float = _DEFAULT_RESOLUTION_COMMUNITY,
) -> dict[str, Any]:
    """Build 3-level hierarchy from flat Leiden communities.

    Arguments:
        G: input graph (undirected; DiGraph is converted)
        communities: output of cluster.cluster() — {community_id: [node_ids]}
        resolution_community: Leiden resolution for the meta-community pass
            (default 0.5; lower → fewer larger meta-communities)

    Returns:
        Hierarchy dict matching the schema declared in the module docstring.
    """
    if G.is_directed():
        G = G.to_undirected()

    leaf_level = _build_leaf_level(G, communities)
    community_level = _build_community_level(G, communities, resolution_community)
    root_level = _build_root_level(leaf_level, community_level, G)

    return {
        "version": "1",
        "metadata": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "leaf_count": len(leaf_level),
            "meta_community_count": len(community_level),
            "algorithm": "leiden",
            "resolution_leaf": 1.0,
            "resolution_community": resolution_community,
            "graph_sha256": _graph_sha256(G),
        },
        "levels": {
            "leaf": leaf_level,
            "community": community_level,
            "root": root_level,
        },
    }


def _build_leaf_level(G: nx.Graph, communities: dict[int, list[str]]) -> dict[str, dict]:
    """Materialise leaf-level entries with label + cohesion."""
    leaf: dict[str, dict] = {}
    for cid, nodes in communities.items():
        nodes_sorted = sorted(nodes)
        leaf[str(cid)] = {
            "nodes": nodes_sorted,
            "label": _community_label(G, nodes_sorted),
            "size": len(nodes_sorted),
            "cohesion": round(cohesion_score(G, nodes_sorted), 6),
        }
    return dict(sorted(leaf.items(), key=lambda kv: int(kv[0])))


def _build_community_level(
    G: nx.Graph,
    communities: dict[int, list[str]],
    resolution: float,
) -> dict[str, dict]:
    """Run a second Leiden pass over the community-graph to get meta-communities.

    Each leaf community becomes a single node in the meta-graph; edges are
    weighted by the count of inter-community edges in the original graph G.
    """
    if len(communities) <= 1:
        only_cid = next(iter(communities), 0)
        return {
            "0": {
                "children": [str(only_cid)],
                "label": "all",
                "node_count": sum(len(nodes) for nodes in communities.values()),
            }
        }

    node_to_cid: dict[str, int] = {
        node: cid for cid, nodes in communities.items() for node in nodes
    }

    meta_graph = nx.Graph()
    meta_graph.add_nodes_from(communities.keys())
    cross_edges: dict[tuple[int, int], int] = {}
    for u, v in G.edges():
        cu = node_to_cid.get(u)
        cv = node_to_cid.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        key = (cu, cv) if cu < cv else (cv, cu)
        cross_edges[key] = cross_edges.get(key, 0) + 1

    for (cid_a, cid_b), weight in cross_edges.items():
        meta_graph.add_edge(cid_a, cid_b, weight=weight)

    if meta_graph.number_of_edges() == 0:
        # No inter-community edges — each leaf is its own meta-community.
        community_level: dict[str, dict] = {}
        for idx, cid in enumerate(sorted(communities.keys())):
            community_level[str(idx)] = {
                "children": [str(cid)],
                "label": _community_label(G, communities[cid]),
                "node_count": len(communities[cid]),
            }
        return community_level

    meta_partition = _partition(meta_graph, resolution=resolution)
    meta_to_children: dict[int, list[int]] = {}
    for cid, meta_cid in meta_partition.items():
        meta_to_children.setdefault(meta_cid, []).append(cid)

    # Re-index meta-community IDs by total node count descending for determinism.
    ordered_metas = sorted(
        meta_to_children.items(),
        key=lambda kv: (
            -sum(len(communities[cid]) for cid in kv[1]),
            kv[0],
        ),
    )

    community_level: dict[str, dict] = {}
    for new_idx, (_, children) in enumerate(ordered_metas):
        children_sorted = sorted(children)
        node_count = sum(len(communities[cid]) for cid in children_sorted)
        # Label the meta-community after its largest leaf community.
        largest_cid = max(children_sorted, key=lambda c: len(communities[c]))
        community_level[str(new_idx)] = {
            "children": [str(cid) for cid in children_sorted],
            "label": _community_label(G, communities[largest_cid]),
            "node_count": node_count,
        }
    return community_level


def _build_root_level(
    leaf_level: dict[str, dict],
    community_level: dict[str, dict],
    G: nx.Graph,
) -> dict[str, dict]:
    """Aggregate root summary across the whole mandala.

    The root summary is a token-bounded text representation enumerating
    the top-N communities by size. Downstream A/B harness counts tokens
    on this string to measure reduction vs full graph traversal.
    """
    top_communities = sorted(
        leaf_level.values(),
        key=lambda c: (-c["size"], c["label"]),
    )[:_ROOT_SUMMARY_TOP_N]

    summary_lines = [
        f"# Mandala root summary · {len(leaf_level)} leaf communities · "
        f"{len(community_level)} meta-communities · {G.number_of_nodes()} nodes",
        "",
        "## Top communities by size",
        "",
    ]
    for c in top_communities:
        summary_lines.append(
            f"- {c['label']} · {c['size']} nodes · cohesion {c['cohesion']:.3f}"
        )

    summary_lines.extend([
        "",
        "## Meta-community grouping",
        "",
    ])
    for meta_id, meta in community_level.items():
        summary_lines.append(
            f"- meta-{meta_id} ({meta['label']}) · "
            f"{len(meta['children'])} leaves · {meta['node_count']} nodes"
        )

    summary = "\n".join(summary_lines) + "\n"

    return {
        "all": {
            "summary": summary,
            "community_count": len(leaf_level),
            "node_count": G.number_of_nodes(),
            "top_communities": [
                {"id": cid, "label": leaf["label"], "size": leaf["size"]}
                for cid, leaf in sorted(
                    leaf_level.items(),
                    key=lambda kv: (-kv[1]["size"], kv[1]["label"]),
                )[:_ROOT_SUMMARY_TOP_N]
            ],
        }
    }


# ─── Label heuristic ──────────────────────────────────────────────────────────


def _community_label(G: nx.Graph, nodes: list[str]) -> str:
    """Pick the highest-degree node's label as the community label.

    Ties broken lexically for determinism. Returns 'anonymous' for empty
    communities (defensive — should not occur in normal flow).
    """
    if not nodes:
        return "anonymous"
    subgraph = G.subgraph(nodes)
    candidates = sorted(
        ((node, subgraph.degree(node)) for node in nodes),
        key=lambda nd: (-nd[1], str(nd[0])),
    )
    best_node = candidates[0][0]
    data = G.nodes[best_node] if best_node in G.nodes else {}
    return str(data.get("label", best_node))


# ─── Content addressing ───────────────────────────────────────────────────────


def _graph_sha256(G: nx.Graph) -> str:
    """Deterministic hash of graph structure for cache invalidation."""
    payload = {
        "nodes": sorted(str(n) for n in G.nodes()),
        "edges": sorted(
            (str(u), str(v)) if str(u) <= str(v) else (str(v), str(u))
            for u, v in G.edges()
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ─── flock(2) lock helpers ────────────────────────────────────────────────────


def _acquire_lock(fd: int) -> None:
    """Acquire exclusive lock on file descriptor. POSIX uses fcntl.flock,
    Windows falls back to msvcrt.locking on a single byte range."""
    if sys.platform.startswith("win"):
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)


def _release_lock(fd: int) -> None:
    """Release exclusive lock acquired via _acquire_lock."""
    if sys.platform.startswith("win"):
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            # Lock may already be released if file was closed first.
            pass
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


# ─── Artifact writer ──────────────────────────────────────────────────────────


def write_hierarchy_artifact(
    hierarchy: dict[str, Any],
    out_dir: Path,
    artifact_name: str = "community-hierarchy.json",
) -> Path:
    """Write hierarchy dict to `<out_dir>/<artifact_name>` under flock(2).

    Lock file at `<out_dir>/.lock` is opened in `'a+'` mode so existing
    lock-token content (if any) is preserved across runs (per cycle-3
    MINOR-2 corrigenda). The artifact write happens INSIDE the lock-held
    block — never after release (per cycle-3 MINOR-1 corrigenda).

    Returns the path of the written artifact.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / artifact_name
    lock_path = out_dir / ".lock"

    payload = json.dumps(hierarchy, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    # Open lock file in 'a+' mode — preserves existing lock-token content,
    # does NOT truncate on open (per cycle-3 MINOR-2 corrigenda).
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        _acquire_lock(lock_fp.fileno())
        try:
            # Artifact write MUST happen inside the with-block (per cycle-3
            # MINOR-1 corrigenda) — moving this after the `with` would release
            # the lock before the write completes, defeating mutual exclusion.
            artifact_path.write_text(payload, encoding="utf-8")
        finally:
            _release_lock(lock_fp.fileno())

    return artifact_path


# ─── CLI entry point ──────────────────────────────────────────────────────────


def _main(argv: list[str] | None = None) -> int:
    """Standalone CLI: read graph.json + communities → write hierarchy artifact.

    Invocation:
        python -m graphify.hierarchy [--graph PATH] [--out PATH]
                                     [--resolution-community FLOAT]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="D#174 М8 P4 hierarchical community detection artifact builder.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("graphify-out/graph.json"),
        help="path to graph.json (default: graphify-out/graph.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("graphify-out"),
        help="output directory for community-hierarchy.json (default: graphify-out/)",
    )
    parser.add_argument(
        "--resolution-community",
        type=float,
        default=_DEFAULT_RESOLUTION_COMMUNITY,
        help=f"Leiden resolution for meta-community pass (default {_DEFAULT_RESOLUTION_COMMUNITY})",
    )
    args = parser.parse_args(argv)

    if not args.graph.exists():
        print(f"[graphify.hierarchy] ERROR · graph.json missing: {args.graph}", file=sys.stderr)
        return 1

    payload = json.loads(args.graph.read_text(encoding="utf-8"))
    G = nx.node_link_graph(payload, edges="edges") if "edges" in payload else nx.node_link_graph(payload)

    from graphify.cluster import cluster as _cluster
    communities = _cluster(G)

    hierarchy = build_hierarchy(
        G,
        communities,
        resolution_community=args.resolution_community,
    )

    artifact_path = write_hierarchy_artifact(hierarchy, args.out)
    print(
        f"[graphify.hierarchy] wrote {artifact_path} · "
        f"{hierarchy['metadata']['leaf_count']} leaf · "
        f"{hierarchy['metadata']['meta_community_count']} meta · "
        f"{hierarchy['metadata']['node_count']} nodes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
