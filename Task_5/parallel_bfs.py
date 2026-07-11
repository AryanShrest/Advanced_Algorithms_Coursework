"""
parallel_bfs.py
-----------------
Concurrent (multi-threaded) Breadth-First Search.

Design: LEVEL-SYNCHRONOUS parallel BFS.
    - BFS naturally proceeds in "levels" (frontiers). All nodes in the
      current frontier can have their neighbours explored independently
      of one another, which makes BFS a good fit for a shared-memory
      threaded implementation.
    - At each level, the current frontier is partitioned evenly across
      `num_threads` worker threads.
    - Each worker thread scans the neighbours of its assigned nodes and
      tries to mark them visited. Because the `visited` array is shared
      mutable state, the check-and-set on `visited[v]` is a CRITICAL
      SECTION and is protected with a threading.Lock (mutex) to prevent
      two threads from both seeing `visited[v] == False` and adding the
      same node twice (a classic race condition).
    - Each thread appends newly discovered nodes to its OWN local list
      (no lock needed there), and the lists are merged after the threads
      for this level have finished (threading.Thread.join() acts as an
      implicit barrier between levels, ensuring level L+1 never starts
      before level L has fully completed).

Synchronisation primitives used:
    - threading.Lock()   -> protects the shared `visited` array (mutex)
    - Thread.join()      -> barrier-style synchronisation between BFS levels
"""

import threading


def bfs_parallel(adj: dict, source: int, num_threads: int = 4):
    """
    Level-synchronous parallel BFS.

    Args:
        adj:         adjacency list {node: [neighbours]}
        source:      starting node
        num_threads: number of worker threads to use per level

    Returns:
        list of nodes in the order they were first visited
    """
    n = len(adj)
    visited = [False] * n
    visited[source] = True
    visited_lock = threading.Lock()          # protects `visited` (critical section)

    order = [source]
    current_frontier = [source]

    while current_frontier:
        # --- Partition the current frontier across threads ---
        chunks = [[] for _ in range(num_threads)]
        for idx, node in enumerate(current_frontier):
            chunks[idx % num_threads].append(node)

        local_results = [[] for _ in range(num_threads)]

        def worker(chunk, local_result):
            for u in chunk:
                for v in adj[u]:
                    # ---- CRITICAL SECTION: check-and-set must be atomic ----
                    with visited_lock:
                        if visited[v]:
                            continue
                        visited[v] = True
                    # -----------------------------------------------------
                    local_result.append(v)   # thread-local, no lock needed

        threads = [
            threading.Thread(target=worker, args=(chunks[i], local_results[i]))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()                          # implicit barrier: wait for level to finish

        next_frontier = []
        for lr in local_results:
            next_frontier.extend(lr)

        order.extend(next_frontier)
        current_frontier = next_frontier

    return order


if __name__ == "__main__":
    from graph_gen import generate_random_graph

    g = generate_random_graph(10, avg_degree=3, seed=1)
    print(bfs_parallel(g, 0, num_threads=4))
