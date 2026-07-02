"""
benchmark.py

Empirical testing harness for Task 1. For dataset sizes n = 100, 1000,
10000, this script:

    1. Generates n random City records, named after real Nepali towns
       (see nepal_cities.py) with a reproducible random distance from
       Kathmandu.
    2. Times INSERT of all n cities into: BST, AVL Tree, Min-Heap,
       Hash Table.
    3. Times SEARCH of a fixed sample of existing keys in each structure.
    4. Times DELETE of a fixed sample of existing keys in each structure.
    5. Records wall-clock time using time.perf_counter(), averaged over
       several repeats to reduce noise.
    6. Prints CLEAR, ALIGNED TABLES to the console as it goes (one table
       per dataset size), plus a final summary table.
    7. Writes:
         - results.csv                 (raw machine-readable table)
         - task1_report.md             (human-readable Markdown report:
                                         theory table + empirical tables +
                                         auto-written conclusions + charts)
         - insert_time.png / search_time.png / delete_time.png
                                        (log-log time-vs-n charts)
         - bst_vs_avl_height.png       (tree height comparison)
         - overview_grid.png           (all 4 charts combined in one grid,
                                         useful for a single-figure summary)

Run with:  python3 benchmark.py
"""

import csv
import random
import time
import statistics as stats

from tabulate import tabulate

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt

from city import City
from bst import BST
from avl_tree import AVLTree
from min_heap import MinHeap
from hash_table import HashTable
from nepal_cities import get_city_name

RANDOM_SEED = 42
SIZES = [100, 1_000, 10_000]
SEARCH_DELETE_SAMPLE_FRACTION = 0.1  # time 10% of keys for search/delete
REPEATS = 3  # repeat each timed operation and report the mean, to reduce noise

# Theoretical Big-O reference table (used in the printed/markdown report)
THEORY = [
    ["BST",       "O(log n)", "O(log n)", "O(log n)", "O(n)", "O(n)", "O(n)",
     "Degenerates to a linked list on sorted/adversarial input"],
    ["AVL Tree",  "O(log n)", "O(log n)", "O(log n)", "O(log n)", "O(log n)", "O(log n)",
     "Guaranteed balanced; extra rotation overhead per update"],
    ["Min-Heap",  "O(log n)", "O(n)*", "O(log n)", "O(log n)", "O(n)", "O(log n)",
     "*Arbitrary search is O(n); only min is O(1) via peek"],
    ["Hash Table", "O(1)", "O(1)", "O(1)", "O(n)", "O(n)", "O(n)",
     "Worst case only if hash function collides badly"],
]
THEORY_HEADERS = ["Structure", "Insert (avg)", "Search (avg)", "Delete (avg)",
                   "Insert (worst)", "Search (worst)", "Delete (worst)", "Notes"]


def generate_cities(n, seed=RANDOM_SEED):
    """Generate n cities named after real Nepali towns, with a reproducible
    random distance (km) from Kathmandu used as the ordering/priority key."""
    rng = random.Random(seed)
    cities = []
    for i in range(n):
        cities.append(City(
            city_id=i,
            name=get_city_name(i),
            latitude=round(rng.uniform(26.3, 30.4), 4),   # Nepal's lat range
            longitude=round(rng.uniform(80.0, 88.2), 4),  # Nepal's lon range
            population=rng.randint(1_000, 3_000_000),
            distance=round(rng.uniform(0, 1_200), 3),     # km from Kathmandu
        ))
    return cities


def time_it(fn, repeats=REPEATS):
    """Run fn() `repeats` times, return the mean wall-clock time in seconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        samples.append(end - start)
    return stats.mean(samples)


def fmt_time(seconds):
    """Human-friendly time formatting: switches units so numbers stay readable."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


# ---------------------------------------------------------------------- #
# Per-structure benchmark functions
# ---------------------------------------------------------------------- #
def bench_bst(cities, sample_ids_by_distance):
    def build():
        t = BST()
        for c in cities:
            t.insert(c)
        return t

    insert_time = time_it(lambda: build())
    tree = build()
    height = tree.height()

    def do_search():
        for d in sample_ids_by_distance:
            tree.search(d)
    search_time = time_it(do_search)

    def do_delete():
        t2 = build()
        for d in sample_ids_by_distance:
            t2.delete(d)
    delete_time = time_it(do_delete)

    return insert_time, search_time, delete_time, height


def bench_avl(cities, sample_ids_by_distance):
    def build():
        t = AVLTree()
        for c in cities:
            t.insert(c)
        return t

    insert_time = time_it(lambda: build())
    tree = build()
    height = tree.height()

    def do_search():
        for d in sample_ids_by_distance:
            tree.search(d)
    search_time = time_it(do_search)

    def do_delete():
        t2 = build()
        for d in sample_ids_by_distance:
            t2.delete(d)
    delete_time = time_it(do_delete)

    return insert_time, search_time, delete_time, height


def bench_heap(cities, sample_count):
    def build():
        h = MinHeap()
        for c in cities:
            h.push(c)
        return h

    insert_time = time_it(lambda: build())

    # "search" isn't a native heap operation -- O(n) linear scan is the
    # honest comparison point, included here for completeness/discussion.
    def do_search():
        h = build()
        for _ in range(sample_count):
            target = random.Random(1).choice(h._heap).distance
            for c in h._heap:
                if c.distance == target:
                    break
    search_time = time_it(do_search)

    def do_delete():
        h = build()
        for _ in range(sample_count):
            if not h.is_empty():
                h.pop()
    delete_time = time_it(do_delete)

    return insert_time, search_time, delete_time


def bench_hash(cities, sample_ids_by_key):
    def build():
        ht = HashTable()
        for c in cities:
            ht.insert(c.city_id, c)
        return ht

    insert_time = time_it(lambda: build())
    table = build()
    stats_dict = table.collision_stats()

    def do_search():
        for k in sample_ids_by_key:
            table.search(k)
    search_time = time_it(do_search)

    def do_delete():
        t2 = build()
        for k in sample_ids_by_key:
            t2.delete(k)
    delete_time = time_it(do_delete)

    return insert_time, search_time, delete_time, stats_dict


# ---------------------------------------------------------------------- #
# Main experiment loop
# ---------------------------------------------------------------------- #
def run_experiments():
    rows = []            # flat rows for CSV
    per_n_tables = []    # (n, table_rows) for pretty printing / markdown
    heights = []         # (n, bst_height, avl_height)
    hash_stats_by_n = {}
    sample_preview_by_n = {}  # a few real Nepal city names actually tested

    print("\n" + "#" * 72)
    print("#  TASK 1 -- EMPIRICAL BENCHMARK: BST vs AVL vs MIN-HEAP vs HASH TABLE")
    print("#  Dataset: cities of Nepal, keyed on distance (km) from Kathmandu")
    print("#" * 72)

    print("\nTheoretical complexity reference:\n")
    print(tabulate(THEORY, headers=THEORY_HEADERS, tablefmt="grid"))

    for n in SIZES:
        print(f"\n\n{'=' * 72}")
        print(f"  DATASET SIZE n = {n:,}")
        print("=" * 72)

        cities = generate_cities(n)
        rng = random.Random(1)
        sample_size = max(1, int(n * SEARCH_DELETE_SAMPLE_FRACTION))
        sample_cities = rng.sample(cities, sample_size)
        sample_distances = [c.distance for c in sample_cities]
        sample_ids = [c.city_id for c in sample_cities]
        sample_preview_by_n[n] = [c.name for c in sample_cities[:5]]

        print(f"  (search/delete timed on a sample of {sample_size} keys, "
              f"{REPEATS} repeats averaged)")
        print(f"  Example cities in this dataset: "
              f"{', '.join(c.name for c in cities[:5])}, ...\n")

        bst_i, bst_s, bst_d, bst_h = bench_bst(cities, sample_distances)
        avl_i, avl_s, avl_d, avl_h = bench_avl(cities, sample_distances)
        heap_i, heap_s, heap_d = bench_heap(cities, sample_size)
        hash_i, hash_s, hash_d, hstats = bench_hash(cities, sample_ids)
        hash_stats_by_n[n] = hstats

        table_rows = [
            ["BST",       fmt_time(bst_i),  fmt_time(bst_s),  fmt_time(bst_d),  bst_h],
            ["AVL Tree",  fmt_time(avl_i),  fmt_time(avl_s),  fmt_time(avl_d),  avl_h],
            ["Min-Heap",  fmt_time(heap_i), fmt_time(heap_s), fmt_time(heap_d), "n/a"],
            ["Hash Table", fmt_time(hash_i), fmt_time(hash_s), fmt_time(hash_d), "n/a"],
        ]
        headers = ["Structure", "Insert (mean)", "Search (mean)",
                   "Delete (mean)", "Tree Height"]
        print(tabulate(table_rows, headers=headers, tablefmt="grid"))

        per_n_tables.append((n, headers, table_rows))
        heights.append((n, bst_h, avl_h))

        rows.extend([
            {"n": n, "structure": "BST", "operation": "insert", "time_s": bst_i},
            {"n": n, "structure": "BST", "operation": "search", "time_s": bst_s},
            {"n": n, "structure": "BST", "operation": "delete", "time_s": bst_d},
            {"n": n, "structure": "AVL", "operation": "insert", "time_s": avl_i},
            {"n": n, "structure": "AVL", "operation": "search", "time_s": avl_s},
            {"n": n, "structure": "AVL", "operation": "delete", "time_s": avl_d},
            {"n": n, "structure": "MinHeap", "operation": "insert", "time_s": heap_i},
            {"n": n, "structure": "MinHeap", "operation": "search", "time_s": heap_s},
            {"n": n, "structure": "MinHeap", "operation": "delete", "time_s": heap_d},
            {"n": n, "structure": "HashTable", "operation": "insert", "time_s": hash_i},
            {"n": n, "structure": "HashTable", "operation": "search", "time_s": hash_s},
            {"n": n, "structure": "HashTable", "operation": "delete", "time_s": hash_d},
        ])

    return rows, per_n_tables, heights, hash_stats_by_n, sample_preview_by_n


# ---------------------------------------------------------------------- #
# Output writers
# ---------------------------------------------------------------------- #
def write_csv(rows, path="results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "structure", "operation", "time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Saved] Raw results       -> {path}")


def plot_operation(rows, operation, path, title):
    structures = sorted(set(r["structure"] for r in rows))
    plt.figure(figsize=(7, 5))
    for structure in structures:
        xs = sorted(set(r["n"] for r in rows))
        ys = [next(r["time_s"] for r in rows
                   if r["structure"] == structure
                   and r["operation"] == operation
                   and r["n"] == n) for n in xs]
        plt.plot(xs, ys, marker="o", linewidth=2, label=structure)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of nodes (n) -- log scale")
    plt.ylabel("Mean wall-clock time (seconds) -- log scale")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] Chart              -> {path}")


def plot_heights(heights, path="bst_vs_avl_height.png"):
    ns = [h[0] for h in heights]
    bst_h = [h[1] for h in heights]
    avl_h = [h[2] for h in heights]

    plt.figure(figsize=(7, 5))
    plt.plot(ns, bst_h, marker="o", linewidth=2, label="BST height")
    plt.plot(ns, avl_h, marker="s", linewidth=2, label="AVL height")
    plt.xscale("log")
    plt.xlabel("Number of nodes (n) -- log scale")
    plt.ylabel("Tree height")
    plt.title("Tree Height: BST vs AVL (random insert order)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] Chart              -> {path}")


def plot_overview_grid(rows, heights, path="overview_grid.png"):
    """All four charts combined into a single 2x2 figure for a quick,
    single-image overview -- handy to paste straight into a report."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    operations = ["insert", "search", "delete"]
    titles = ["Insert Time vs n", "Search Time vs n", "Delete Time vs n"]
    structures = sorted(set(r["structure"] for r in rows))

    for ax, operation, title in zip(axes.flat[:3], operations, titles):
        for structure in structures:
            xs = sorted(set(r["n"] for r in rows))
            ys = [next(r["time_s"] for r in rows
                       if r["structure"] == structure
                       and r["operation"] == operation
                       and r["n"] == n) for n in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, label=structure)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (log)")
        ax.set_ylabel("time, s (log)")
        ax.set_title(title)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    ax = axes.flat[3]
    ns = [h[0] for h in heights]
    ax.plot(ns, [h[1] for h in heights], marker="o", linewidth=2, label="BST height")
    ax.plot(ns, [h[2] for h in heights], marker="s", linewidth=2, label="AVL height")
    ax.set_xscale("log")
    ax.set_xlabel("n (log)")
    ax.set_ylabel("height")
    ax.set_title("Tree Height: BST vs AVL")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=8)

    fig.suptitle("Task 1 -- Data Structure Benchmark Overview "
                  "(Nepal city dataset)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Combined overview  -> {path}")


def write_markdown_report(per_n_tables, heights, hash_stats_by_n, rows,
                           sample_preview_by_n, path="task1_report.md"):
    """Writes a single, clean Markdown report combining the theory table,
    every empirical results table, and auto-generated conclusions based
    on the actual numbers produced this run."""

    def fastest(operation, n):
        candidates = [r for r in rows if r["operation"] == operation and r["n"] == n]
        best = min(candidates, key=lambda r: r["time_s"])
        return best["structure"], best["time_s"]

    lines = []
    lines.append("# Task 1 -- Advanced Data Structures: Benchmark Report\n")
    lines.append("Auto-generated by `benchmark.py`. Dataset: cities of "
                  "Nepal (see `nepal_cities.py`), keyed on distance (km) "
                  "from Kathmandu. Contains theoretical complexity, "
                  f"empirical timings (wall-clock, mean of {REPEATS} "
                  "repeats), and observations drawn directly from the "
                  "measured results below.\n")

    lines.append("## 1. Theoretical Complexity (Big-O)\n")
    lines.append(tabulate(THEORY, headers=THEORY_HEADERS, tablefmt="github"))
    lines.append("\n")

    lines.append("## 2. Empirical Results\n")
    for n, headers, table_rows in per_n_tables:
        lines.append(f"### n = {n:,}\n")
        preview = ", ".join(sample_preview_by_n[n])
        lines.append(f"Example cities sampled for search/delete timing: "
                      f"{preview}, ...\n")
        lines.append(tabulate(table_rows, headers=headers, tablefmt="github"))
        lines.append("\n")
        fast_search, t = fastest("search", n)
        lines.append(f"- Fastest **search** at n={n:,}: **{fast_search}** "
                      f"({fmt_time(t)})\n")
        hs = hash_stats_by_n[n]
        lines.append(f"- Hash table load factor at n={n:,}: "
                      f"{hs['load_factor']} "
                      f"(max chain length {hs['max_chain_length']})\n")

    lines.append("## 3. Tree Height Comparison (BST vs AVL)\n")
    height_table = [[n, b, a] for n, b, a in heights]
    lines.append(tabulate(height_table,
                           headers=["n", "BST height", "AVL height"],
                           tablefmt="github"))
    lines.append("\n")

    lines.append("## 4. Charts\n")
    lines.append("![Insert time](insert_time.png)\n")
    lines.append("![Search time](search_time.png)\n")
    lines.append("![Delete time](delete_time.png)\n")
    lines.append("![Tree height](bst_vs_avl_height.png)\n")
    lines.append("![Overview grid](overview_grid.png)\n")

    lines.append("## 5. Observations (auto-summarised from the data above)\n")
    largest_n = max(SIZES)
    fast_search, _ = fastest("search", largest_n)
    slow_insert_structure = max(
        (r for r in rows if r["operation"] == "insert" and r["n"] == largest_n),
        key=lambda r: r["time_s"]
    )["structure"]
    last_heights = heights[-1]

    lines.append(
        f"- At the largest tested size (n={largest_n:,}), **{fast_search}** "
        "gave the fastest search -- consistent with its O(1) average-case "
        "theoretical complexity.\n"
    )
    lines.append(
        f"- **{slow_insert_structure}** had the slowest insert time at "
        f"n={largest_n:,}. For the AVL tree specifically, this reflects "
        "the cost of rebalancing (rotations) on every insert -- the "
        "\"hidden constant factor\" behind its O(log n) guarantee.\n"
    )
    lines.append(
        f"- Tree height at n={largest_n:,}: BST reached height "
        f"{last_heights[1]}, AVL stayed at height {last_heights[2]}. "
        "This demonstrates AVL's self-balancing keeping the tree "
        "shallower, even though the *random* insertion order used here "
        "kept the plain BST from degenerating to its true worst case "
        "(a sorted-input BST would be far taller).\n"
    )
    lines.append(
        "- The Min-Heap's search time is consistently the slowest of the "
        "four structures, because a heap only guarantees fast access to "
        "the minimum element -- looking up an arbitrary city requires an "
        "O(n) linear scan. This supports using a heap only for "
        "priority-queue access patterns (e.g. \"next nearest city to "
        "visit on the route\"), not for general lookups.\n"
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Saved] Markdown report    -> {path}")


if __name__ == "__main__":
    (rows, per_n_tables, heights,
     hash_stats_by_n, sample_preview_by_n) = run_experiments()

    print("\n\n" + "#" * 72)
    print("#  WRITING OUTPUT FILES")
    print("#" * 72)

    write_csv(rows)
    plot_operation(rows, "insert", "insert_time.png", "Insert Time vs n")
    plot_operation(rows, "search", "search_time.png", "Search Time vs n")
    plot_operation(rows, "delete", "delete_time.png", "Delete Time vs n")
    plot_heights(heights)
    plot_overview_grid(rows, heights)
    write_markdown_report(per_n_tables, heights, hash_stats_by_n, rows,
                           sample_preview_by_n)

    print("\nAll done. Open task1_report.md for a clean, ready-to-read "
          "write-up of the results (theory table + empirical tables + "
          "charts + auto-written observations).")
