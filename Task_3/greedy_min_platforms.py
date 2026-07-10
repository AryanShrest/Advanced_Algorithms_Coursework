"""
Task 3b - Greedy Algorithm
Problem: Minimum Number of Platforms

Given the arrival and departure times of n trains at a station, find the
minimum number of platforms required so that no train has to wait.

--------------------------------------------------------------------------
GREEDY CHOICE
--------------------------------------------------------------------------
Sort all arrival times and all departure times independently. Then sweep
through events in chronological order using two pointers:
  * if the next event (by time) is an arrival, a train needs a platform
    -> increment the "platforms currently in use" counter.
  * if the next event is a departure, a platform becomes free
    -> decrement the counter.
Track the maximum value the counter ever reaches; that is the answer.

Intuition: the minimum number of platforms needed at any instant equals
the number of trains simultaneously "in the station" at that instant.
The greedy processes events strictly in time order, so the running
counter at any point in the sweep exactly equals the number of
overlapping intervals at that instant. Taking the maximum over the whole
sweep therefore gives the true minimum platforms required - not an
approximation.

PROOF OF OPTIMALITY
  Lower bound: if at some instant t there are k trains simultaneously at
  the station, then at least k platforms are required at that instant
  (pigeonhole: two trains cannot share one platform at the same time).
  Hence the minimum number of platforms is >= max over t of the number
  of trains present at t.

  Upper bound (achievability): the greedy computes exactly
  max_t (trains present at t), via the sweep. Because it directly counts
  live intervals rather than following a heuristic proxy, the value it
  outputs is achievable by construction (assign the k-th arriving train,
  among those simultaneously present, to the k-th platform) and is
  never larger than the true requirement.

  Since the greedy's output is both a valid lower bound and an
  achievable upper bound, it is optimal. (Note: unlike some greedy
  algorithms - e.g. weighted interval scheduling - this greedy choice
  requires no exchange argument because it computes the quantity being
  optimised directly, rather than approximating it.)

  Edge case handled: if one train departs at exactly the same time
  another arrives, the departure is processed first (freeing the
  platform) provided the platform can be reused instantly; this
  assumption is stated explicitly in `min_platforms()` and can be
  flipped by changing the tie-break rule.

--------------------------------------------------------------------------
COMPLEXITY ANALYSIS
--------------------------------------------------------------------------
Let n = number of trains.
  Sorting arrivals and departures: O(n log n)
  Two-pointer sweep over 2n events: O(n)
  ------------------------------------------
  TOTAL TIME:  O(n log n)
  TOTAL SPACE: O(n) (for the sorted copies of arrival/departure arrays)

Hidden constant: two independent sorts (2 x O(n log n) with Python's
Timsort) dominate the constant factor; the sweep itself is a single
cheap O(n) pass with only comparisons and increments/decrements.

--------------------------------------------------------------------------
COMPARISON WITH AN EXACT (BRUTE-FORCE) APPROACH
--------------------------------------------------------------------------
A brute-force approach checks, for every train's arrival time t, how many
other trains have arrival <= t <= departure (a direct pairwise overlap
count), and takes the maximum -> O(n^2). Because the greedy computes the
exact same quantity (max simultaneous overlap) but does so via a single
sorted sweep instead of pairwise comparison, greedy is *always* optimal
here, not just a fast approximation - this problem is one of the cases
where "greedy" and "exact" coincide. The two implementations are
compared below for a correctness check and to quantify the O(n log n)
vs O(n^2) gap empirically.
"""

from __future__ import annotations
import random
import time
from typing import List, Tuple


def min_platforms(arrivals: List[int], departures: List[int]) -> int:
    """O(n log n) greedy sweep. Assumes a departing train frees its
    platform in time for a simultaneously-arriving train (i.e. on a tie,
    process the departure event first)."""
    arr = sorted(arrivals)
    dep = sorted(departures)
    n = len(arr)

    platforms_needed = 0
    max_platforms = 0
    i = j = 0
    while i < n and j < n:
        if arr[i] <= dep[j]:
            platforms_needed += 1
            max_platforms = max(max_platforms, platforms_needed)
            i += 1
        else:
            platforms_needed -= 1
            j += 1
    return max_platforms


def min_platforms_brute_force(arrivals: List[int], departures: List[int]) -> int:
    """O(n^2) exact baseline: for each train, count how many trains
    (including itself) are present at its arrival instant."""
    n = len(arrivals)
    best = 0
    for i in range(n):
        count = 0
        t = arrivals[i]
        for k in range(n):
            if arrivals[k] <= t <= departures[k]:
                count += 1
        best = max(best, count)
    return best


def generate_random_schedule(n: int, seed: int = 7) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    arrivals, departures = [], []
    for _ in range(n):
        a = rng.randint(0, 2359)
        d = a + rng.randint(1, 120)
        arrivals.append(a)
        departures.append(d)
    return arrivals, departures


def empirical_timing():
    print("\n--- Empirical timing: greedy O(n log n) vs brute force O(n^2) ---")
    print(f"{'n':>6} | {'greedy (s)':>12} | {'brute force (s)':>16}")
    for n in [50, 100, 500, 1_000, 5_000]:
        arrivals, departures = generate_random_schedule(n)

        t0 = time.perf_counter()
        g = min_platforms(arrivals, departures)
        t1 = time.perf_counter()

        # brute force gets slow fast; skip for very large n
        if n <= 1_000:
            t2 = time.perf_counter()
            bf = min_platforms_brute_force(arrivals, departures)
            t3 = time.perf_counter()
            assert g == bf, f"Mismatch at n={n}: greedy={g}, brute={bf}"
            print(f"{n:>6} | {t1 - t0:>12.6f} | {t3 - t2:>16.6f}")
        else:
            print(f"{n:>6} | {t1 - t0:>12.6f} | {'skipped':>16}")


if __name__ == "__main__":
    arrivals = [900, 940, 950, 1100, 1500, 1800]
    departures = [910, 1200, 1120, 1130, 1900, 2000]

    result = min_platforms(arrivals, departures)
    print("Arrivals:  ", arrivals)
    print("Departures:", departures)
    print(f"\nMinimum platforms required (greedy): {result}")

    bf = min_platforms_brute_force(arrivals, departures)
    print(f"Minimum platforms required (brute force check): {bf}")
    assert result == bf
    print("Greedy result verified against brute force. ✔")

    empirical_timing()
