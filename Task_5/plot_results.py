"""
plot_results.py
------------------
Generates the "Plot speedup vs thread count" figure required by Task 5.

Takes the in-memory results produced by benchmark_task5.run_benchmark()
directly -- no intermediate CSV/data file is read or written. The only
file this script produces is the plot image itself (speedup_plot.png),
which is the actual required deliverable (a labelled graph).

Run:
    python plot_results.py
Produces:
    speedup_plot.png
"""

from collections import defaultdict
import matplotlib.pyplot as plt


def plot_speedup(results, out_file="speedup_plot.png"):
    """
    Args:
        results: list of dicts as returned by benchmark_task5.run_benchmark(),
                  e.g. {"nodes": 1000, "threads": 4, "type": "threading",
                        "time_seconds": 0.01, "speedup": 2.1}
        out_file: path to save the resulting plot image
    """
    data = defaultdict(lambda: defaultdict(list))  # data[type][nodes] = [(threads, speedup)]

    for row in results:
        if row["type"] == "sequential":
            continue
        data[row["type"]][row["nodes"]].append((row["threads"], row["speedup"]))

    plt.figure(figsize=(9, 6))
    styles = {"threading": "-o", "multiprocessing": "-s"}

    max_threads = 1
    for impl_type, per_size in data.items():
        for n, points in sorted(per_size.items()):
            points.sort()
            threads = [p[0] for p in points]
            speedups = [p[1] for p in points]
            max_threads = max(max_threads, max(threads))
            style = styles.get(impl_type, "-o")
            plt.plot(threads, speedups, style, label=f"{impl_type} ({n} nodes)")

    plt.plot([1, max_threads], [1, max_threads], linestyle="--", color="gray", label="Ideal linear speedup")

    plt.xlabel("Number of Threads / Processes")
    plt.ylabel("Speedup (T_sequential / T_parallel)")
    plt.title("Parallel BFS Speedup vs Thread Count")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    from benchmark_task5 import run_benchmark

    results = run_benchmark()
    plot_speedup(results)
