"""
prim.py

Prim's algorithm for Minimum Spanning Tree (MST) construction, using a
binary min-heap (heapq) -- the "lazy deletion" variant, where stale heap
entries for already-visited vertices are simply skipped when popped
rather than removed proactively.

MST is defined for UNDIRECTED graphs. Since Task 2 models the network as
a weighted DIRECTED graph (roads may have direction-dependent costs, e.g.
one-way streets or tolls), we first convert it via Graph.to_undirected()
(see graph.py), which keeps the cheaper of the two directions wherever an
edge exists both ways. This is a documented modelling assumption:
"connect these two cities as cheaply as possible in either direction."

Theoretical complexity (V = vertices, E = edges), adjacency list + binary
heap:
    Time:  O(E log V)
    Space: O(V + E)

This is structurally almost identical to Dijkstra's complexity, since
both algorithms repeatedly extract a minimum from a heap and relax
neighbouring edges -- the difference is that Prim's relaxation compares
against the EDGE weight only (cost to CONNECT a new vertex), while
Dijkstra compares against the cumulative PATH distance (cost to REACH a
vertex from the source).
"""

import heapq


def prim_mst(graph, source=None, record_steps=False):
    """
    graph must be UNDIRECTED (call graph.to_undirected() first if needed).

    Returns:
        mst_edges    : list of (u, v, weight) edges in the MST
        total_weight : sum of MST edge weights
        steps        : list of (u, v, weight) in the order each edge was
                        added to the tree -- used for step-by-step
                        visualisation. Empty if record_steps=False.
    """
    if not graph.vertices:
        return [], 0, []

    if source is None:
        source = next(iter(graph.vertices))

    visited = {source}
    mst_edges = []
    steps = []
    total_weight = 0

    heap = []
    for v, w in graph.neighbors(source):
        heapq.heappush(heap, (w, source, v))

    while heap and len(visited) < graph.num_vertices():
        w, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, w))
        total_weight += w
        if record_steps:
            steps.append((u, v, w))

        for to, wt in graph.neighbors(v):
            if to not in visited:
                heapq.heappush(heap, (wt, v, to))

    return mst_edges, total_weight, steps


if __name__ == "__main__":
    from graph import Graph

    print("=" * 60)
    print("  PRIM'S ALGORITHM (MINIMUM SPANNING TREE) -- DEMO")
    print("  (cheapest way to connect all cities, Nepal network)")
    print("=" * 60)

    g = Graph(directed=True)
    edges = [
        ("Kathmandu", "Lalitpur", 5), ("Kathmandu", "Bhaktapur", 13),
        ("Kathmandu", "Pokhara", 200), ("Lalitpur", "Bhaktapur", 10),
        ("Pokhara", "Butwal", 150), ("Kathmandu", "Butwal", 278),
        ("Butwal", "Itahari", 400), ("Kathmandu", "Itahari", 503),
        ("Itahari", "Dharan", 25), ("Itahari", "Biratnagar", 22),
        ("Dharan", "Biratnagar", 45),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)

    undirected = g.to_undirected()
    print(f"\nConverted to undirected graph: {undirected}")

    mst_edges, total_weight, steps = prim_mst(undirected, source="Kathmandu",
                                               record_steps=True)

    print(f"\n[1] MST construction order (edge added, in order):")
    for i, (u, v, w) in enumerate(steps, 1):
        print(f"      {i}. {u} -- {v}  (weight {w})")

    print(f"\n[2] Total MST weight (cheapest way to connect every city): "
          f"{total_weight} km")

    print("\n" + "=" * 60)
    print("  END OF PRIM'S ALGORITHM DEMO")
    print("=" * 60)
