"""
Heuristic 1: Greedy Construction Heuristic
=============================================
First-Fit Decreasing (FFD), generalised to d dimensions.

STRATEGY
--------
1. Sort items in DECREASING order of total resource demand
   (sum over all dimensions). Placing "big" items first tends to
   avoid leaving awkward leftover capacity that no later item fits
   into - this is the same intuition behind FFD's strong worst-case
   guarantee for the 1D case.
2. For each item (in that order), scan existing open bins in the
   order they were created and place the item in the FIRST bin that
   has enough remaining capacity in *every* dimension.
3. If no open bin fits the item, open a new bin.

This is a pure construction heuristic: it builds one solution in a
single greedy pass and never revisits earlier decisions.

COMPLEXITY
----------
Let n = number of items, d = number of dimensions, m = number of bins
opened by the algorithm (m <= n).

- Sorting items:                    O(n log n)
- For each item, scanning bins:     O(m) bins, each feasibility check
  costs O(d)                     -> O(m * d) per item
- Overall:                          O(n log n + n * m * d)

In the worst case m = O(n), giving O(n^2 * d), but in practice m is
much smaller than n (typically O(n / avg_bin_utilisation)), so the
observed runtime is close to O(n * d) for the placement phase, with
sorting dominating for large n. The hidden constant is small - each
feasibility check is just d floating point comparisons.
"""

import time
from typing import List, Tuple

from bin_packing_problem import Item, Bin


def greedy_first_fit_decreasing(items: List[Item], num_dims: int) -> Tuple[List[Bin], float]:
    """
    Run First-Fit-Decreasing bin packing.

    Returns (bins, elapsed_seconds).
    """
    start = time.perf_counter()

    # Sort by total demand, descending
    sorted_items = sorted(items, key=lambda it: sum(it.demand), reverse=True)

    bins: List[Bin] = []
    for item in sorted_items:
        placed = False
        for b in bins:
            if b.can_fit(item):
                b.add(item)
                placed = True
                break
        if not placed:
            new_bin = Bin(num_dims)
            new_bin.add(item)
            bins.append(new_bin)

    elapsed = time.perf_counter() - start
    return bins, elapsed


if __name__ == "__main__":
    from bin_packing_problem import generate_instance, solution_is_valid, num_bins_used

    items = generate_instance(num_items=50, num_dims=3, seed=1)
    bins, t = greedy_first_fit_decreasing(items, num_dims=3)
    print(f"Greedy FFD: {num_bins_used(bins)} bins used in {t*1000:.3f} ms")
    print("Valid solution:", solution_is_valid(bins, items))
