"""
sequential_bfs.py
------------------
Baseline sequential Breadth-First Search (BFS).
This is the "sequential version" against which the parallel
implementation in parallel_bfs.py is compared (Task 5 requirement:
"Measure and report speedup achieved compared to the sequential version").
"""

from collections import deque


def bfs_sequential(adj: dict, source: int):
    """
    Standard single-threaded BFS traversal.

    Args:
        adj:    adjacency list {node: [neighbours]}
        source: starting node

    Returns:
        list of nodes in the order they were first visited
    """
    n = len(adj)
    visited = [False] * n
    visited[source] = True
    order = [source]

    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                order.append(v)
                queue.append(v)

    return order


if __name__ == "__main__":
    from graph_gen import generate_random_graph

    g = generate_random_graph(10, avg_degree=3, seed=1)
    print(bfs_sequential(g, 0))
