"""
bellman_ford.py

Bellman-Ford shortest-path algorithm. Unlike Dijkstra, it correctly
handles NEGATIVE edge weights, and can DETECT negative-weight cycles
(a cycle whose total weight is negative, meaning "shortest path" is
undefined -- you could loop around it forever to make the path cheaper).

Theoretical complexity (V = vertices, E = edges):
    Time:  O(V * E)   -- V-1 relaxation rounds, each scanning all E edges,
                          plus one extra round to detect negative cycles
    Space: O(V)        -- distance/predecessor arrays
                          (O(V + E) if edges are also stored explicitly)

Why is this needed at all, given Dijkstra is faster?
-------------------------------------------------------
Dijkstra greedily finalises each vertex's distance the moment it's popped
from the heap, assuming no future discovery can make it cheaper. A
negative edge can break that assumption -- a longer-looking path taken
now could later be discounted by a negative edge, producing a cheaper
total than Dijkstra already "locked in". Bellman-Ford instead relaxes
every edge, repeatedly, for V-1 rounds -- slower, but correct even with
negative weights (as long as there's no negative cycle, in which case no
finite shortest path exists at all).
"""


def bellman_ford(graph, source):
    """
    Returns:
        dist            : dict {vertex: shortest distance from source}
        prev            : dict {vertex: predecessor on the shortest path}
        has_negative_cycle : bool
        cycle_edges     : list of (u, v) edges detected as still
                           relaxable after V-1 rounds (evidence of a
                           negative cycle reachable from source)
        history         : list of dicts, one snapshot of `dist` after
                           each relaxation round -- used to visualise
                           convergence over iterations
    """
    dist = {v: float("inf") for v in graph.vertices}
    prev = {}
    dist[source] = 0

    edge_list = graph.edges()
    v_count = graph.num_vertices()
    history = [dict(dist)]

    for _ in range(v_count - 1):
        updated = False
        for u, v, w in edge_list:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        history.append(dict(dist))
        if not updated:
            break  # early exit: distances have converged

    # One more pass: if anything can still be relaxed, a negative cycle
    # reachable from `source` exists.
    has_negative_cycle = False
    cycle_edges = []
    for u, v, w in edge_list:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            has_negative_cycle = True
            cycle_edges.append((u, v))

    return dist, prev, has_negative_cycle, cycle_edges, history


if __name__ == "__main__":
    from graph import Graph

    print("=" * 60)
    print("  BELLMAN-FORD ALGORITHM -- DEMO 1: negative weight, no cycle")
    print("  (hypothetical network with a toll REBATE modelled as a")
    print("   negative-weight edge)")
    print("=" * 60)

    g1 = Graph(directed=True)
    edges1 = [
        ("Kathmandu", "Lalitpur", 5),
        ("Kathmandu", "Bhaktapur", 13),
        ("Lalitpur", "Pokhara", 210),
        ("Bhaktapur", "Pokhara", -20),   # hypothetical rebate/discount
        ("Pokhara", "Butwal", 150),
    ]
    for u, v, w in edges1:
        g1.add_edge(u, v, w)

    dist, prev, has_cycle, cycle_edges, history = bellman_ford(g1, "Kathmandu")

    print(f"\n[1] Negative cycle detected: {has_cycle}")
    print("\n[2] Shortest distances from Kathmandu:")
    for city in sorted(dist, key=lambda c: dist[c]):
        print(f"      {city:<12}: {dist[city]}")
    print(f"\n      Note: Kathmandu->Bhaktapur->Pokhara (13 + -20 = -7) is "
          f"cheaper than Kathmandu->Lalitpur->Pokhara (5 + 210 = 215).")
    print(f"      Dijkstra would have gotten this WRONG, because it "
          f"finalises Lalitpur/Bhaktapur greedily before seeing the "
          f"rebate on the Bhaktapur->Pokhara edge.")

    print("\n[3] Convergence history (distance to Pokhara after each round):")
    for i, snapshot in enumerate(history):
        val = snapshot.get("Pokhara", float("inf"))
        print(f"      Round {i}: {val}")

    print("\n" + "=" * 60)
    print("  DEMO 2: negative CYCLE (no valid shortest path exists)")
    print("=" * 60)

    g2 = Graph(directed=True)
    edges2 = [
        ("A", "B", 1),
        ("B", "C", -3),
        ("C", "A", 1),   # A -> B -> C -> A has total weight 1 - 3 + 1 = -1
        ("A", "D", 10),
    ]
    for u, v, w in edges2:
        g2.add_edge(u, v, w)

    dist2, prev2, has_cycle2, cycle_edges2, _ = bellman_ford(g2, "A")
    print(f"\n[1] Negative cycle detected: {has_cycle2}")
    print(f"[2] Edges still relaxable (part of/reachable from the cycle): "
          f"{cycle_edges2}")

    print("\n" + "=" * 60)
    print("  END OF BELLMAN-FORD DEMO")
    print("=" * 60)
