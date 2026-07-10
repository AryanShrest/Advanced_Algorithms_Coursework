"""
Heuristic 2: Simulated Annealing (SA)
========================================

STRATEGY
--------
Simulated Annealing improves on a starting solution by exploring
neighbouring solutions, occasionally accepting *worse* moves (with a
probability that shrinks over time) to escape local optima - unlike
plain hill-climbing / local search, which only ever accepts improving
moves and gets stuck at the first local optimum it finds.

1. START from an initial feasible solution (we warm-start from the
   Greedy FFD heuristic - a common and effective SA design choice,
   since it gives the search a good starting point rather than a
   random one).
2. NEIGHBOUR MOVE: pick a random item and try to move it to a
   different, randomly chosen bin (or a new empty bin). Only feasible
   moves (that do not violate any dimension's capacity) are applied.
3. OBJECTIVE (COST) FUNCTION: we do not use "number of bins" directly,
   because that signal is too coarse (most single moves don't change
   the bin count, so the search would be blind). Instead we use the
   standard Falkenauer fitness for bin packing:

        cost(solution) = num_bins - (1 / num_bins) * sum_over_bins(fill_ratio(b)^2)

   This rewards solutions that concentrate load into fewer, fuller
   bins, and gives a smooth gradient towards emptying out
   under-utilised bins (which can then be removed), even before the
   bin count itself drops.
4. ACCEPTANCE: a move that improves cost is always accepted. A move
   that worsens cost is accepted with probability exp(-delta / T),
   where T (temperature) starts high and is multiplied by a cooling
   rate < 1 after every iteration, so the search becomes progressively
   more selective (closer to pure hill-climbing) over time.
5. After all iterations, empty bins are discarded and the best
   solution seen during the run is returned.

COMPLEXITY
----------
Let n = number of items, d = dimensions, I = number of SA iterations
(a fixed budget we choose, independent of n).

- Each iteration: O(1) random choices + O(d) feasibility check for the
  candidate bin + O(d) cost recomputation for the (at most 2) affected
  bins  ->  O(d) per iteration.
- Total: O(I * d).

Because I is fixed (a tunable heuristic parameter, not derived from
input size), SA's running time is largely INDEPENDENT of n for the
search phase itself, though building/scoring the initial solution
still costs O(n log n + n * m * d) (from the Greedy warm start). This
is the key practical trade-off versus the Greedy heuristic: SA spends
extra, tunable, fixed-budget time to try to reduce the bin count
found by Greedy, at the cost of no longer being a single deterministic
pass.
"""

import time
import math
import random
from typing import List, Tuple

from bin_packing_problem import Item, Bin
from greedy_heuristic import greedy_first_fit_decreasing


def _solution_cost(bins: List[Bin]) -> float:
    """Falkenauer-style fitness: fewer, fuller bins score lower (better)."""
    active_bins = [b for b in bins if b.items]
    m = len(active_bins)
    if m == 0:
        return 0.0
    fill_sq_sum = sum(b.fill_ratio() ** 2 for b in active_bins)
    return m - (fill_sq_sum / m)


def _copy_bins(bins: List[Bin], num_dims: int) -> List[Bin]:
    new_bins = []
    for b in bins:
        nb = Bin(num_dims)
        for it in b.items:
            nb.add(it)
        new_bins.append(nb)
    return new_bins


def simulated_annealing(
    items: List[Item],
    num_dims: int,
    iterations: int = 3000,
    initial_temp: float = 2.0,
    cooling_rate: float = 0.995,
    seed: int = 7,
) -> Tuple[List[Bin], float]:
    """
    Run Simulated Annealing bin packing, warm-started from Greedy FFD.

    Returns (bins, elapsed_seconds).
    """
    rng = random.Random(seed)
    start = time.perf_counter()

    # Warm start from the greedy solution (its own runtime is counted
    # separately by the greedy heuristic module; here we only pay the
    # cost of constructing it once as our SA starting point).
    bins, _ = greedy_first_fit_decreasing(items, num_dims)

    current_cost = _solution_cost(bins)
    best_bins = _copy_bins(bins, num_dims)
    best_cost = current_cost

    temp = initial_temp

    for _ in range(iterations):
        if not bins:
            break

        # Pick a random non-empty source bin and a random item in it
        non_empty = [b for b in bins if b.items]
        if not non_empty:
            break
        src_bin = rng.choice(non_empty)
        item = rng.choice(src_bin.items)

        # Pick a destination: an existing bin (not the source) or a new bin
        candidates = [b for b in bins if b is not src_bin]
        open_new = rng.random() < 0.1 or not candidates
        if open_new:
            dst_bin = Bin(num_dims)
            bins.append(dst_bin)
        else:
            dst_bin = rng.choice(candidates)

        if not dst_bin.can_fit(item):
            # Move infeasible - reject and clean up any newly opened empty bin
            if open_new and not dst_bin.items:
                bins.remove(dst_bin)
            temp *= cooling_rate
            continue

        # Tentatively apply the move
        src_bin.remove(item)
        dst_bin.add(item)

        # Drop bins left empty by the move
        bins = [b for b in bins if b.items]

        new_cost = _solution_cost(bins)
        delta = new_cost - current_cost

        accept = delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-6))

        if accept:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_bins = _copy_bins(bins, num_dims)
        else:
            # Revert the move
            dst_bin.remove(item)
            if not dst_bin.items and dst_bin in bins:
                bins.remove(dst_bin)
            src_bin.add(item)

        temp *= cooling_rate

    elapsed = time.perf_counter() - start
    return best_bins, elapsed


if __name__ == "__main__":
    from bin_packing_problem import generate_instance, solution_is_valid, num_bins_used

    items = generate_instance(num_items=50, num_dims=3, seed=1)
    bins, t = simulated_annealing(items, num_dims=3, iterations=3000)
    print(f"Simulated Annealing: {num_bins_used(bins)} bins used in {t*1000:.3f} ms")
    print("Valid solution:", solution_is_valid(bins, items))
