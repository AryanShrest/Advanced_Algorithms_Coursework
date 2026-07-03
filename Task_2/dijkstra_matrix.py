"""
dijkstra_matrix.py

A SECOND implementation of Dijkstra's algorithm, this time using a plain
adjacency MATRIX and a linear scan to find the next closest unvisited
vertex, instead of a binary heap.

Theoretical complexity (V = vertices):
    Time:  O(V^2)          -- V iterations, each doing an O(V) scan
    Space: O(V^2)           -- the matrix itself

Why include this alongside the heap-based version in dijkstra.py?
-------------------------------------------------------------------
Big-O comparison alone is misleading for dense graphs:

    - Heap-based (adjacency list): O((V + E) log V)
    - Matrix-based (array scan):   O(V^2)

For a SPARSE graph (E ~ O(V)), (V + E) log V is far smaller than V^2, so
the heap-based version wins clearly, both in theory and in practice.

For a DENSE graph, E approaches O(V^2), so the heap-based bound becomes
O(V^2 log V) -- theoretically WORSE than the matrix version's O(V^2)!
In practice the matrix version also has a much smaller constant factor
per iteration (a tight array scan with no heap bookkeeping, no push/pop
overhead, no object allocation), so it frequently outperforms the
heap-based version on dense graphs even before you look at the Big-O.

This file exists specifically to produce the empirical evidence for
Task 2's "suitability for dense vs sparse graphs" comparison.
"""


def dijkstra_matrix(matrix, index_of, nodes, source):
    n = len(nodes)
    src_idx = index_of[source]

    dist = [float("inf")] * n
    dist[src_idx] = 0
    visited = [False] * n
    prev = [None] * n

    for _ in range(n):
        # Linear scan for the closest unvisited vertex -- O(V) per
        # iteration, O(V^2) overall. This is the step a binary heap
        # would normally accelerate to O(log V).
        u = -1
        best = float("inf")
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1:
            break  # remaining vertices are unreachable
        visited[u] = True

        for v in range(n):
            w = matrix[u][v]
            if w == float("inf") or visited[v]:
                continue
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u

    dist_by_name = {nodes[i]: dist[i] for i in range(n)}
    prev_by_name = {nodes[i]: nodes[prev[i]] for i in range(n) if prev[i] is not None}
    return dist_by_name, prev_by_name


if __name__ == "__main__":
    from graph import Graph
    from dijkstra import dijkstra

    print("=" * 60)
    print("  MATRIX-BASED DIJKSTRA (O(V^2), NO HEAP) -- DEMO")
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

    matrix, index_of, nodes = g.to_adjacency_matrix()
    dist_matrix, prev_matrix = dijkstra_matrix(matrix, index_of, nodes, "Kathmandu")
    dist_heap, prev_heap, _ = dijkstra(g, "Kathmandu")

    print("\nCross-checking both implementations agree:")
    for city in sorted(dist_matrix):
        match = "OK" if dist_matrix[city] == dist_heap[city] else "MISMATCH"
        print(f"      {city:<12} heap={dist_heap[city]:<6} "
              f"matrix={dist_matrix[city]:<6} [{match}]")

    print("\n" + "=" * 60)
    print("  END OF MATRIX-BASED DIJKSTRA DEMO")
    print("=" * 60)
