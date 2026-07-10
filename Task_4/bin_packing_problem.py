"""
Task 4: NP-Hard Problem - Multi-dimensional Bin Packing (MDBP)
================================================================

PROBLEM DEFINITION
-------------------
We are given a set of n items. Each item i has a resource-demand vector
    w_i = (w_i^1, w_i^2, ..., w_i^d)
across d resource dimensions (e.g. CPU, RAM, bandwidth), where each
w_i^k is normalised to lie in (0, 1].

We are given bins of unit capacity in every dimension (capacity = 1.0
for each of the d resources). An assignment of items to bins is
FEASIBLE for a bin B if, for every dimension k:
    sum_{i in B} w_i^k <= 1.0

GOAL: partition all n items into the minimum number of bins such that
every bin's assignment is feasible.

WHY THIS PROBLEM IS NP-HARD
-----------------------------
Multi-dimensional Bin Packing (MDBP) is NP-Hard. This can be shown
formally by a restriction argument (a special case of a well-known
NP-Hard reduction chain):

1. Classic 1-Dimensional Bin Packing (1D-BP) is known to be NP-Hard.
   This is proven by a reduction from the PARTITION problem: given a
   multiset S of positive integers with total sum T, PARTITION asks
   whether S can be split into two subsets each summing to T/2. Any
   instance of PARTITION can be transformed into a 1D-BP instance with
   bin capacity T/2 by treating each element of S as an item weight:
   PARTITION has a "yes" answer if and only if the resulting 1D-BP
   instance can be packed into exactly 2 bins. Since PARTITION is
   NP-Complete, 1D-BP is NP-Hard.

2. MDBP with d = 1 dimension is *exactly* 1D-BP. Because 1D-BP is a
   special case (restriction) of MDBP obtained simply by setting
   d = 1, and 1D-BP is already NP-Hard, MDBP (for any d >= 1) must
   also be NP-Hard: if we could solve the general d-dimensional
   problem efficiently, we could solve the 1-dimensional (and hence
   PARTITION) special case efficiently too, which is assumed
   impossible unless P = NP.

3. In fact MDBP is strictly harder in practice than 1D-BP for d >= 2
   because feasibility now requires simultaneous satisfaction of d
   independent capacity constraints, which shrinks the feasible search
   space and makes even the *decision* version (e.g. "can this be
   packed into k bins?") more constrained and harder to solve near-
   optimally with simple heuristics such as First-Fit.

Because exact algorithms (e.g. exhaustive search / ILP) scale
exponentially with the number of items, we instead use HEURISTICS
(Task 4 requirement) to find good, not necessarily optimal, solutions
in polynomial time.
"""

import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class Item:
    """A single item with a multi-dimensional resource demand vector."""
    item_id: int
    demand: List[float]  # length == num_dims, each value in (0, 1]

    def __repr__(self):
        vec = ", ".join(f"{v:.2f}" for v in self.demand)
        return f"Item({self.item_id}: [{vec}])"


class Bin:
    """A bin with unit capacity (1.0) in every dimension."""

    def __init__(self, num_dims: int):
        self.num_dims = num_dims
        self.load = [0.0] * num_dims
        self.items: List[Item] = []

    def can_fit(self, item: Item) -> bool:
        return all(self.load[k] + item.demand[k] <= 1.0 + 1e-9
                    for k in range(self.num_dims))

    def add(self, item: Item) -> None:
        for k in range(self.num_dims):
            self.load[k] += item.demand[k]
        self.items.append(item)

    def remove(self, item: Item) -> None:
        for k in range(self.num_dims):
            self.load[k] -= item.demand[k]
        self.items.remove(item)

    def fill_ratio(self) -> float:
        """Average utilisation across dimensions - used as a fitness signal."""
        return sum(self.load) / self.num_dims if self.num_dims else 0.0

    def __repr__(self):
        return f"Bin(load={[round(x, 2) for x in self.load]}, n_items={len(self.items)})"


def generate_instance(num_items: int, num_dims: int = 3, seed: int = 42) -> List[Item]:
    """
    Generate a random Multi-dimensional Bin Packing instance.

    Item demands are drawn from Uniform(0.05, 0.5) per dimension, which
    is a standard way to generate non-trivial bin-packing benchmarks
    (guarantees at least 2 items can potentially co-exist in a bin,
    while still forcing genuine packing decisions).
    """
    rng = random.Random(seed)
    items = []
    for i in range(num_items):
        demand = [round(rng.uniform(0.05, 0.5), 3) for _ in range(num_dims)]
        items.append(Item(item_id=i, demand=demand))
    return items


def solution_is_valid(bins: List[Bin], items: List[Item]) -> bool:
    """Sanity check: every item assigned exactly once, no bin over capacity."""
    assigned_ids = set()
    for b in bins:
        for k in range(b.num_dims):
            if b.load[k] > 1.0 + 1e-6:
                return False
        for it in b.items:
            if it.item_id in assigned_ids:
                return False  # assigned twice
            assigned_ids.add(it.item_id)
    return assigned_ids == {it.item_id for it in items}


def num_bins_used(bins: List[Bin]) -> int:
    return len([b for b in bins if b.items])
