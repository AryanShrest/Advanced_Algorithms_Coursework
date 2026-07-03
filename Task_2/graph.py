"""
graph.py

Weighted DIRECTED graph, implemented using an ADJACENCY LIST.

Why adjacency list over adjacency matrix?
------------------------------------------
A real transportation network (roads between cities) is SPARSE: each city
typically connects to only a handful of neighbouring cities, not to every
other city in the country. For a sparse graph with V vertices and E edges
where E is much smaller than V^2:

    - Adjacency list uses O(V + E) space.
    - Adjacency matrix uses O(V^2) space, regardless of how few edges
      actually exist -- wasteful for a sparse network.
    - Iterating over a vertex's neighbours (needed by Dijkstra/Prim's
      inner loop) is O(degree(v)) with an adjacency list, vs O(V) with
      a matrix (you must scan every column even if most are "no edge").

The trade-off: an adjacency matrix gives O(1) edge-weight lookups
(`is there an edge u->v, and what does it cost?`), which an adjacency
list cannot do without a linear scan. This is exactly why Task 2 also
includes a MATRIX-BASED Dijkstra implementation (dijkstra_matrix.py) for
comparison on DENSE graphs, where the matrix's O(1) lookup and simpler
inner loop can outperform a heap despite the worse Big-O.

For this coursework's use case (a Nepal city road network, inherently
sparse), the adjacency list is the primary and more appropriate
representation.
"""

from collections import defaultdict
import random


class Graph:
    def __init__(self, directed=True):
        self.directed = directed
        self.adj = defaultdict(list)   # vertex -> list of (neighbour, weight)
        self.vertices = set()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def add_vertex(self, v):
        self.vertices.add(v)
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u, v, weight):
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def neighbors(self, u):
        return self.adj[u]

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return sum(len(v) for v in self.adj.values())

    def edges(self):
        """Return a flat list of (u, v, weight) triples."""
        result = []
        seen_undirected = set()
        for u in self.adj:
            for v, w in self.adj[u]:
                if self.directed:
                    result.append((u, v, w))
                else:
                    key = frozenset((u, v))
                    if key not in seen_undirected:
                        seen_undirected.add(key)
                        result.append((u, v, w))
        return result

    def density(self):
        """Fraction of possible directed edges that actually exist."""
        v = self.num_vertices()
        if v < 2:
            return 0.0
        max_edges = v * (v - 1)
        return self.num_edges() / max_edges

    # ------------------------------------------------------------------ #
    # Conversions
    # ------------------------------------------------------------------ #
    def to_undirected(self):
        """Build an undirected version, useful for Prim's MST. Where an
        edge exists in both directions with different weights, the
        SMALLER weight is kept (optimistic: cheapest way to connect)."""
        g = Graph(directed=False)
        best = {}
        for u, v, w in self.edges():
            key = frozenset((u, v))
            if key not in best or w < best[key][2]:
                best[key] = (u, v, w)
        for u, v, w in best.values():
            g.add_edge(u, v, w)
        for v in self.vertices:
            g.add_vertex(v)
        return g

    def to_adjacency_matrix(self):
        """Returns (matrix, index_of) for algorithms/comparisons that need
        O(1) edge lookups (e.g. dijkstra_matrix.py)."""
        nodes = sorted(self.vertices, key=str)
        index_of = {v: i for i, v in enumerate(nodes)}
        n = len(nodes)
        INF = float("inf")
        matrix = [[INF] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 0
        for u in self.adj:
            for v, w in self.adj[u]:
                matrix[index_of[u]][index_of[v]] = min(matrix[index_of[u]][index_of[v]], w)
        return matrix, index_of, nodes

    # ------------------------------------------------------------------ #
    # Random graph generation (for empirical benchmarking)
    # ------------------------------------------------------------------ #
    @staticmethod
    def random_graph(n, density="sparse", weight_range=(1, 200),
                      directed=True, seed=None, allow_negative=False):
        """Generate a random weighted graph with n vertices.

        density="sparse": ~3 outgoing edges per vertex on average
                           (typical of a real road network).
        density="dense":  ~25% of all possible directed edges exist.
        """
        rng = random.Random(seed)
        g = Graph(directed=directed)
        nodes = list(range(n))
        for v in nodes:
            g.add_vertex(v)

        if density == "sparse":
            edges_per_node = 3
        elif density == "dense":
            edges_per_node = max(1, int(0.25 * (n - 1)))
        else:
            raise ValueError("density must be 'sparse' or 'dense'")

        for u in nodes:
            possible_targets = [x for x in nodes if x != u]
            k = min(edges_per_node, len(possible_targets))
            targets = rng.sample(possible_targets, k)
            for v in targets:
                low, high = weight_range
                if allow_negative:
                    w = rng.randint(-abs(low), high)
                else:
                    w = rng.randint(low, high)
                g.add_edge(u, v, w)

        return g

    def __repr__(self):
        return (f"Graph(directed={self.directed}, "
                f"V={self.num_vertices()}, E={self.num_edges()})")


if __name__ == "__main__":
    print("=" * 60)
    print("  GRAPH (ADJACENCY LIST) -- DEMO")
    print("=" * 60)

    g = Graph(directed=True)
    g.add_edge("Kathmandu", "Lalitpur", 5)
    g.add_edge("Kathmandu", "Bhaktapur", 13)
    g.add_edge("Kathmandu", "Pokhara", 200)
    g.add_edge("Pokhara", "Butwal", 150)
    g.add_edge("Butwal", "Itahari", 400)

    print(f"\n{g}")
    print(f"Density: {g.density():.4f}")
    print("\nEdges:")
    for u, v, w in g.edges():
        print(f"   {u} -> {v}  (weight {w})")

    print("\nAdjacency matrix conversion:")
    matrix, index_of, nodes = g.to_adjacency_matrix()
    print("   Nodes order:", nodes)
    for row in matrix:
        print("  ", ["inf" if x == float("inf") else x for x in row])
