"""
dijkstra.py

Dijkstra's shortest-path algorithm using a binary min-heap priority queue
(via Python's heapq -- functionally equivalent to the MinHeap built in
Task 1, which could be substituted here directly).

Theoretical complexity (V = vertices, E = edges), adjacency list + binary
heap:
    Time:  O((V + E) log V)
    Space: O(V + E)

Only works correctly with NON-NEGATIVE edge weights: once a vertex is
popped from the heap (finalised), Dijkstra assumes no cheaper path to it
can be found later. A negative edge can violate that assumption, which is
exactly why Bellman-Ford exists for graphs that may contain negative
weights (see bellman_ford.py).
"""

import heapq


def dijkstra(graph, source, record_steps=False):
    """
    Returns:
        dist  : dict {vertex: shortest distance from source}
        prev  : dict {vertex: predecessor on the shortest path}
        steps : list of (vertex, distance) in the order each vertex was
                FINALISED (popped from the heap) -- this is the order the
                shortest-path tree grows in, used for the step-by-step
                visualisation. Empty list if record_steps=False.
    """
    dist = {v: float("inf") for v in graph.vertices}
    prev = {}
    dist[source] = 0

    heap = [(0, source)]
    visited = set()
    steps = []

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if record_steps:
            steps.append((u, d))

        for v, w in graph.neighbors(u):
            if v in visited:
                continue
            new_dist = d + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, prev, steps


def shortest_path(prev, source, target):
    """Reconstruct the path from source to target using the `prev` map
    returned by dijkstra() or bellman_ford()."""
    if target == source:
        return [source]
    if target not in prev:
        return None  # unreachable
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path


if __name__ == "__main__":
    from graph import Graph

    print("=" * 60)
    print("  DIJKSTRA'S ALGORITHM -- DEMO")
    print("  (shortest paths from Kathmandu across a small Nepal network)")
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

    dist, prev, steps = dijkstra(g, "Kathmandu", record_steps=True)

    print("\n[1] Order in which cities were finalised (heap pop order):")
    for i, (city, d) in enumerate(steps, 1):
        print(f"      {i}. {city:<12} finalised at distance {d} km")

    print("\n[2] Shortest distance from Kathmandu to every city:")
    for city in sorted(dist, key=lambda c: dist[c]):
        print(f"      {city:<12}: {dist[city]} km")

    print("\n[3] Example reconstructed path, Kathmandu -> Biratnagar:")
    path = shortest_path(prev, "Kathmandu", "Biratnagar")
    print("     ", " -> ".join(path))

    print("\n" + "=" * 60)
    print("  END OF DIJKSTRA DEMO")
    print("=" * 60)
