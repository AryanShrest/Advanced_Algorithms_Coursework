"""
parallel_bfs_multiprocessing.py
---------------------------------
OPTIONAL / BONUS implementation.

Python's `threading` module (used in parallel_bfs.py) satisfies the
assignment requirement of "pthreads or an equivalent threading library"
and correctly demonstrates mutex-based synchronisation. However, CPython
has a Global Interpreter Lock (GIL): only one thread executes Python
bytecode at a time, so CPU-bound work (like scanning adjacency lists)
will NOT show real speedup with threading -- this is itself an important
scalability finding to report and discuss (see README_task5.md).

This file provides a `multiprocessing`-based version that uses SEPARATE
OS PROCESSES (each with its own interpreter and no shared GIL) to obtain
genuine parallel speedup. It uses a multiprocessing.Manager list and Lock
to emulate the shared `visited` structure across processes -- the
process equivalent of a mutex-protected critical section.

Use this file if you want to show a scalability comparison between:
    (a) threading   -> limited by the GIL (expect little/no speedup)
    (b) multiprocessing -> true parallel speedup (with process overhead)
in your Task 5 analysis and discussion of "overheads that limit scalability".
"""

import multiprocessing as mp


def _worker(chunk, adj, visited, lock, result_queue):
    local_new = []
    for u in chunk:
        for v in adj[u]:
            with lock:                     # critical section across processes
                if visited[v]:
                    continue
                visited[v] = True
            local_new.append(v)
    result_queue.put(local_new)


def bfs_parallel_mp(adj: dict, source: int, num_workers: int = 4):
    """
    Level-synchronous parallel BFS using multiprocessing (true parallelism,
    no GIL contention). Mirrors the design of bfs_parallel() in
    parallel_bfs.py but with processes instead of threads.
    """
    n = len(adj)
    manager = mp.Manager()
    visited = manager.list([False] * n)
    visited[source] = True
    lock = manager.Lock()

    order = [source]
    current_frontier = [source]

    while current_frontier:
        chunks = [[] for _ in range(num_workers)]
        for idx, node in enumerate(current_frontier):
            chunks[idx % num_workers].append(node)

        result_queue = manager.Queue()
        processes = [
            mp.Process(target=_worker, args=(chunks[i], adj, visited, lock, result_queue))
            for i in range(num_workers)
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        next_frontier = []
        while not result_queue.empty():
            next_frontier.extend(result_queue.get())

        order.extend(next_frontier)
        current_frontier = next_frontier

    return order


if __name__ == "__main__":
    from graph_gen import generate_random_graph

    g = generate_random_graph(10, avg_degree=3, seed=1)
    print(bfs_parallel_mp(g, 0, num_workers=4))
