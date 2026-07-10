"""
Task 3a - Dynamic Programming
Problem: Weighted Job Scheduling with Time Windows

Each job j has a start time s(j), an end/finish time f(j) (the time window
it occupies) and a profit p(j). Two jobs are compatible if their time
windows do not overlap (job i and job j are compatible if f(i) <= s(j)
or f(j) <= s(i)). The goal is to select a subset of mutually compatible
jobs that maximises the total profit.

--------------------------------------------------------------------------
SUBPROBLEM DEFINITION
--------------------------------------------------------------------------
1. Sort all n jobs by finish time: f(1) <= f(2) <= ... <= f(n).
2. For each job i, define p(i) = the largest index k < i such that job k
   is compatible with job i (i.e. f(k) <= s(i)). This is found with
   binary search over the sorted finish times -> O(log n) per job.
3. Let OPT(i) = the maximum achievable profit considering only jobs
   1..i (in finish-time order).

RECURRENCE RELATION
   OPT(0) = 0
   OPT(i) = max( OPT(i-1),                      # exclude job i
                 profit(i) + OPT(p(i)) )         # include job i
   Answer = OPT(n)

This is the classical "Weighted Interval Scheduling" recurrence, extended
here with an explicit notion of a time window per job (start, end).

--------------------------------------------------------------------------
MEMOISATION / BOTTOM-UP STRATEGY
--------------------------------------------------------------------------
Two implementations are provided:
  * top_down_memo()  - recursive, memoised with a dict/array cache.
  * bottom_up_table() - iterative, fills a 1-D DP table OPT[0..n] in
    increasing order of finish time, then reconstructs the chosen jobs
    by walking the table backwards.

Bottom-up is preferred in production code: it avoids recursion-depth
limits for large n and has slightly better constant factors (no function
call / stack-frame overhead per subproblem).

--------------------------------------------------------------------------
COMPLEXITY ANALYSIS
--------------------------------------------------------------------------
Let n = number of jobs.

Sorting jobs by finish time:               O(n log n)
Computing p(i) for every job (binary
search over sorted starts):                O(n log n)
Filling the DP table (n states, O(1)
work per state given p(i) is
precomputed):                              O(n)
Reconstructing the solution:               O(n)
-----------------------------------------------------
TOTAL TIME:                                O(n log n)
TOTAL SPACE:                               O(n)  (DP table + p(i) array)

Hidden constant factor: each "O(1)" transition is actually 2-3 array
reads, 1 comparison and 1 addition; the O(log n) binary searches involve
list slicing/indexing overhead in Python (no low-level array access as in
C). In practice the log-n binary search term and Python's interpreter
overhead (attribute lookups, dynamic typing, list indexing) dominate the
wall-clock time far more than the asymptotic bound suggests - this is why
n log n Python code can be measurably slower than a naive O(n^2) solution
written in C for small/medium n. The empirical section below quantifies
this.
"""

from __future__ import annotations
import bisect
import random
import time
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Job:
    start: int
    end: int
    profit: int

    def __repr__(self):
        return f"Job(start={self.start}, end={self.end}, profit={self.profit})"


def _compute_predecessors(jobs: List[Job]) -> List[int]:
    """For each job i (0-indexed, sorted by end time) find p(i): the
    largest index k < i whose end time <= jobs[i].start. Returns -1 if
    no such job exists. Uses binary search -> O(log n) per job."""
    ends = [j.end for j in jobs]
    preds = []
    for i, job in enumerate(jobs):
        # rightmost index with end <= job.start, searched in ends[0:i]
        idx = bisect.bisect_right(ends, job.start, 0, i) - 1
        preds.append(idx)
    return preds


def top_down_memo(jobs: List[Job]) -> Tuple[int, List[Job]]:
    """Recursive memoised DP. O(n log n) time, O(n) extra space
    (memo table + recursion stack)."""
    jobs = sorted(jobs, key=lambda j: j.end)
    n = len(jobs)
    preds = _compute_predecessors(jobs)
    memo = [None] * (n + 1)

    def opt(i: int) -> int:
        # i is 1-indexed count of jobs considered; OPT(0) = 0
        if i == 0:
            return 0
        if memo[i] is not None:
            return memo[i]
        job = jobs[i - 1]
        p = preds[i - 1] + 1  # convert to 1-indexed
        include = job.profit + opt(p)
        exclude = opt(i - 1)
        memo[i] = max(include, exclude)
        return memo[i]

    best = opt(n)

    # reconstruct
    chosen = []
    i = n
    while i > 0:
        job = jobs[i - 1]
        p = preds[i - 1] + 1
        include = job.profit + opt(p)
        exclude = opt(i - 1)
        if include >= exclude:
            chosen.append(job)
            i = p
        else:
            i -= 1
    chosen.reverse()
    return best, chosen


def bottom_up_table(jobs: List[Job]) -> Tuple[int, List[Job]]:
    """Iterative bottom-up DP. O(n log n) time, O(n) space."""
    jobs = sorted(jobs, key=lambda j: j.end)
    n = len(jobs)
    preds = _compute_predecessors(jobs)

    OPT = [0] * (n + 1)
    for i in range(1, n + 1):
        job = jobs[i - 1]
        p = preds[i - 1] + 1
        include = job.profit + OPT[p]
        exclude = OPT[i - 1]
        OPT[i] = max(include, exclude)

    # reconstruct chosen jobs by walking backwards
    chosen = []
    i = n
    while i > 0:
        job = jobs[i - 1]
        p = preds[i - 1] + 1
        include = job.profit + OPT[p]
        exclude = OPT[i - 1]
        if include >= exclude:
            chosen.append(job)
            i = p
        else:
            i -= 1
    chosen.reverse()
    return OPT[n], chosen


def brute_force(jobs: List[Job]) -> int:
    """Exponential O(2^n) baseline used only to validate correctness on
    small inputs, and to illustrate the gap DP closes."""
    n = len(jobs)
    best = 0

    def compatible(a: Job, b: Job) -> bool:
        return a.end <= b.start or b.end <= a.start

    def rec(i: int, chosen: List[Job], profit: int):
        nonlocal best
        if i == n:
            best = max(best, profit)
            return
        # exclude
        rec(i + 1, chosen, profit)
        # include, if compatible with all chosen so far
        j = jobs[i]
        if all(compatible(j, c) for c in chosen):
            rec(i + 1, chosen + [j], profit + j.profit)

    rec(0, [], 0)
    return best


def generate_random_jobs(n: int, seed: int = 42) -> List[Job]:
    rng = random.Random(seed)
    jobs = []
    for _ in range(n):
        s = rng.randint(0, 1000)
        e = s + rng.randint(1, 50)
        p = rng.randint(1, 100)
        jobs.append(Job(s, e, p))
    return jobs


def empirical_timing():
    import sys
    sys.setrecursionlimit(200_000)
    print("\n--- Empirical timing: top-down vs bottom-up DP ---")
    print(f"{'n':>8} | {'top-down (s)':>14} | {'bottom-up (s)':>14}")
    for n in [100, 1_000, 10_000]:
        jobs = generate_random_jobs(n)

        t0 = time.perf_counter()
        best_td, _ = top_down_memo(jobs)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        best_bu, _ = bottom_up_table(jobs)
        t3 = time.perf_counter()

        assert best_td == best_bu, "Mismatch between DP implementations!"
        print(f"{n:>8} | {t1 - t0:>14.6f} | {t3 - t2:>14.6f}")

    # bottom-up only for larger n (recursion depth limits top-down in Python)
    print("\n(bottom-up only, top-down skipped: Python recursion-depth limit)")
    for n in [100_000]:
        jobs = generate_random_jobs(n)
        t2 = time.perf_counter()
        bottom_up_table(jobs)
        t3 = time.perf_counter()
        print(f"{n:>8} | {'--':>14} | {t3 - t2:>14.6f}")


if __name__ == "__main__":
    # small worked example
    sample_jobs = [
        Job(1, 3, 5),
        Job(2, 5, 6),
        Job(4, 6, 5),
        Job(6, 7, 4),
        Job(5, 8, 11),
        Job(7, 9, 2),
    ]
    print("Sample jobs:", sample_jobs)

    best_profit, chosen_jobs = bottom_up_table(sample_jobs)
    print(f"\nMax profit (bottom-up): {best_profit}")
    print("Chosen jobs:", chosen_jobs)

    best_profit_td, chosen_jobs_td = top_down_memo(sample_jobs)
    print(f"\nMax profit (top-down):  {best_profit_td}")
    print("Chosen jobs:", chosen_jobs_td)

    bf = brute_force(sample_jobs)
    print(f"\nMax profit (brute force check): {bf}")
    assert bf == best_profit == best_profit_td, "DP result disagrees with brute force!"
    print("DP results verified against brute force. ✔")

    empirical_timing()
