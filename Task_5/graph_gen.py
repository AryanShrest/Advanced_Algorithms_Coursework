"""
graph_gen.py
------------
Utility to generate random directed graphs (adjacency list representation)
used for benchmarking sequential vs. parallel BFS traversal in Task 5.

The same adjacency-list representation used in Task 2 (Graph Algorithms)
is reused here so that Task 5 is a genuine "concurrent version of an
earlier algorithm", as required by the assignment brief.
"""

import random


def generate_random_graph(n_nodes: int, avg_degree: int = 6, seed: int = None) -> dict:
    """
    Generate a random directed graph as an adjacency list.

    Args:
        n_nodes:    number of vertices (labelled 0 .. n_nodes-1)
        avg_degree: average out-degree per vertex
        seed:       random seed for reproducibility

    Returns:
        dict mapping node -> list of neighbour nodes
    """
    if seed is not None:
        random.seed(seed)

    adj = {i: [] for i in range(n_nodes)}
    n_edges = n_nodes * avg_degree

    for _ in range(n_edges):
        u = random.randint(0, n_nodes - 1)
        v = random.randint(0, n_nodes - 1)
        if u != v:
            adj[u].append(v)

    return adj


if __name__ == "__main__":
    g = generate_random_graph(10, avg_degree=3, seed=1)
    for node, neighbours in g.items():
        print(node, "->", neighbours)
