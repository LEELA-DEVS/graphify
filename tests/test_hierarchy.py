"""Unit tests for graphify.hierarchy (D#174 М8 P4 hierarchical output)."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import networkx as nx
import pytest

from graphify.cluster import cluster
from graphify.hierarchy import (
    build_hierarchy,
    write_hierarchy_artifact,
    _community_label,
    _graph_sha256,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def two_cluster_graph() -> nx.Graph:
    """Two cleanly separated triangles connected by a single bridge edge."""
    G = nx.Graph()
    # Cluster A: a1-a2-a3 triangle
    G.add_edges_from([("a1", "a2"), ("a2", "a3"), ("a3", "a1")])
    # Cluster B: b1-b2-b3 triangle
    G.add_edges_from([("b1", "b2"), ("b2", "b3"), ("b3", "b1")])
    # Bridge
    G.add_edge("a1", "b1")
    for n in G.nodes():
        G.nodes[n]["label"] = f"node_{n}"
    return G


@pytest.fixture
def single_node_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_node("only", label="only_node")
    return G


# ─── build_hierarchy ──────────────────────────────────────────────────────────


def test_build_hierarchy_returns_three_levels(two_cluster_graph: nx.Graph):
    communities = cluster(two_cluster_graph)
    hierarchy = build_hierarchy(two_cluster_graph, communities)
    assert "levels" in hierarchy
    assert set(hierarchy["levels"].keys()) == {"leaf", "community", "root"}


def test_metadata_includes_graph_sha256(two_cluster_graph: nx.Graph):
    communities = cluster(two_cluster_graph)
    hierarchy = build_hierarchy(two_cluster_graph, communities)
    meta = hierarchy["metadata"]
    assert meta["algorithm"] == "leiden"
    assert meta["node_count"] == 6
    assert meta["edge_count"] == 7
    assert len(meta["graph_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in meta["graph_sha256"])


def test_graph_sha256_is_deterministic(two_cluster_graph: nx.Graph):
    h1 = _graph_sha256(two_cluster_graph)
    h2 = _graph_sha256(two_cluster_graph)
    assert h1 == h2


def test_graph_sha256_changes_on_edge_addition(two_cluster_graph: nx.Graph):
    h_before = _graph_sha256(two_cluster_graph)
    G2 = two_cluster_graph.copy()
    G2.add_edge("a2", "b2")
    h_after = _graph_sha256(G2)
    assert h_before != h_after


def test_leaf_level_covers_all_nodes(two_cluster_graph: nx.Graph):
    communities = cluster(two_cluster_graph)
    hierarchy = build_hierarchy(two_cluster_graph, communities)
    all_leaf_nodes: set[str] = set()
    for leaf in hierarchy["levels"]["leaf"].values():
        all_leaf_nodes.update(leaf["nodes"])
    assert all_leaf_nodes == set(two_cluster_graph.nodes())


def test_root_summary_contains_top_communities(two_cluster_graph: nx.Graph):
    communities = cluster(two_cluster_graph)
    hierarchy = build_hierarchy(two_cluster_graph, communities)
    root = hierarchy["levels"]["root"]["all"]
    assert "summary" in root
    assert isinstance(root["summary"], str)
    assert root["community_count"] == len(hierarchy["levels"]["leaf"])
    assert root["node_count"] == 6
    assert len(root["top_communities"]) == len(hierarchy["levels"]["leaf"])


def test_single_community_collapses_to_one_meta(single_node_graph: nx.Graph):
    communities = cluster(single_node_graph)
    hierarchy = build_hierarchy(single_node_graph, communities)
    assert len(hierarchy["levels"]["community"]) == 1
    meta_zero = hierarchy["levels"]["community"]["0"]
    assert meta_zero["node_count"] == 1


# ─── _community_label ─────────────────────────────────────────────────────────


def test_community_label_picks_highest_degree(two_cluster_graph: nx.Graph):
    label = _community_label(two_cluster_graph, ["a1", "a2", "a3"])
    # All three nodes have the same degree (2) in this triangle, so lexical
    # tie-break picks 'a1' → label 'node_a1'.
    assert label == "node_a1"


def test_community_label_empty_returns_anonymous():
    G = nx.Graph()
    assert _community_label(G, []) == "anonymous"


# ─── write_hierarchy_artifact ─────────────────────────────────────────────────


def test_write_hierarchy_writes_json(tmp_path: Path, two_cluster_graph: nx.Graph):
    communities = cluster(two_cluster_graph)
    hierarchy = build_hierarchy(two_cluster_graph, communities)
    artifact_path = write_hierarchy_artifact(hierarchy, tmp_path)
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["leaf_count"] == len(hierarchy["levels"]["leaf"])


def test_write_hierarchy_creates_lock_file(tmp_path: Path, two_cluster_graph: nx.Graph):
    communities = cluster(two_cluster_graph)
    hierarchy = build_hierarchy(two_cluster_graph, communities)
    write_hierarchy_artifact(hierarchy, tmp_path)
    assert (tmp_path / ".lock").exists()


def test_write_hierarchy_lock_preserves_content_across_runs(
    tmp_path: Path,
    two_cluster_graph: nx.Graph,
):
    """`'a+'` open mode (per cycle-3 MINOR-2 corrigenda) must NOT truncate
    existing lock-token content. Plant a marker, then run write, then verify
    marker survived."""
    lock_path = tmp_path / ".lock"
    lock_path.write_text("pre-existing-marker-content\n", encoding="utf-8")

    communities = cluster(two_cluster_graph)
    hierarchy = build_hierarchy(two_cluster_graph, communities)
    write_hierarchy_artifact(hierarchy, tmp_path)

    assert lock_path.exists()
    after = lock_path.read_text(encoding="utf-8")
    assert "pre-existing-marker-content" in after, (
        "lock file opened in 'a+' mode must preserve prior content "
        "(cycle-3 MINOR-2 corrigenda)"
    )


def test_write_hierarchy_concurrent_writes_serialize(
    tmp_path: Path,
    two_cluster_graph: nx.Graph,
):
    """Two threads writing the same artifact must serialize via flock.
    Final file must be valid JSON (not corrupted)."""
    communities = cluster(two_cluster_graph)
    hierarchy_a = build_hierarchy(two_cluster_graph, communities)
    hierarchy_b = build_hierarchy(two_cluster_graph, communities)
    hierarchy_b["metadata"]["marker"] = "thread_b"

    errors: list[Exception] = []

    def worker(h: dict):
        try:
            for _ in range(5):
                write_hierarchy_artifact(h, tmp_path)
        except Exception as exc:  # pragma: no cover — test failure path
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(hierarchy_a,)),
        threading.Thread(target=worker, args=(hierarchy_b,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"flock concurrency errors: {errors}"
    # File must be parseable JSON (no partial-write corruption)
    parsed = json.loads((tmp_path / "community-hierarchy.json").read_text(encoding="utf-8"))
    assert parsed["metadata"]["leaf_count"] == len(hierarchy_a["levels"]["leaf"])
