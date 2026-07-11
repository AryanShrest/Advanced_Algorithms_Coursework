"""
benchmark_task5.py
--------------------
Runs the empirical experiments required by Task 5:
    - Sequential vs. parallel BFS wall-clock timing
    - Thread counts: 1, 2, 4, 8
    - Reports speedup = T_sequential / T_parallel
    - Returns results as a list of dicts (kept in memory only -- no
      CSV/file output is written by this script).

Run:
    python benchmark_task5.py
"""

import time
import statistics

from graph_gen import generate_random_graph
from sequential_bfs import bfs_sequential
from parallel_bfs import bfs_parallel

# Optional: include the multiprocessing version in the comparison too.
try:
    from parallel_bfs_multiprocessing import bfs_parallel_mp
    HAVE_MP = True
except Exception:
    HAVE_MP = False


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result


def run_benchmark(sizes=(1000, 5000, 10000), thread_counts=(1, 2, 4, 8), trials=3):
    results = []

    for n in sizes:
        print(f"\n=== Graph size: {n} nodes ===")
        adj = generate_random_graph(n, avg_degree=6, seed=42)
        source = 0

        # --- Sequential baseline ---
        seq_times = [time_call(bfs_sequential, adj, source)[0] for _ in range(trials)]
        seq_avg = statistics.mean(seq_times)
        print(f"Sequential: {seq_avg:.4f}s (avg of {trials} runs)")
        results.append({
            "nodes": n, "threads": 1, "type": "sequential",
            "time_seconds": round(seq_avg, 6), "speedup": 1.0
        })

        # --- Threaded parallel version ---
        for k in thread_counts:
            par_times = [time_call(bfs_parallel, adj, source, k)[0] for _ in range(trials)]
            par_avg = statistics.mean(par_times)
            speedup = seq_avg / par_avg if par_avg > 0 else float("inf")
            print(f"Threading (threads={k}): {par_avg:.4f}s -> speedup {speedup:.2f}x")
            results.append({
                "nodes": n, "threads": k, "type": "threading",
                "time_seconds": round(par_avg, 6), "speedup": round(speedup, 3)
            })

        # --- Multiprocessing parallel version (optional, for comparison) ---
        if HAVE_MP:
            for k in thread_counts:
                mp_times = [time_call(bfs_parallel_mp, adj, source, k)[0] for _ in range(trials)]
                mp_avg = statistics.mean(mp_times)
                speedup = seq_avg / mp_avg if mp_avg > 0 else float("inf")
                print(f"Multiprocessing (workers={k}): {mp_avg:.4f}s -> speedup {speedup:.2f}x")
                results.append({
                    "nodes": n, "threads": k, "type": "multiprocessing",
                    "time_seconds": round(mp_avg, 6), "speedup": round(speedup, 3)
                })

    # Results are returned in memory only (no file is written here).
    # Print a simple summary table to the console for quick inspection.
    print("\n--- Summary ---")
    print(f"{'nodes':>7} {'threads':>8} {'type':>16} {'time(s)':>10} {'speedup':>9}")
    for r in results:
        print(f"{r['nodes']:>7} {r['threads']:>8} {r['type']:>16} "
              f"{r['time_seconds']:>10.4f} {r['speedup']:>9.3f}")

    return results


if __name__ == "__main__":
    run_benchmark()
