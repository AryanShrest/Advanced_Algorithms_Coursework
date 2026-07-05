"""
Task 3 - Dynamic Programming
Weighted Job Scheduling with Time Windows

Problem: Given n jobs, each with a start time, end time and profit,
select a subset of non-overlapping jobs that maximises total profit.

Approach: Sort jobs by end time. For each job i, binary-search for the
last job that finishes at or before job i's start time (p(i)). Use a
bottom-up DP table where dp[i] = max profit using the first i jobs
(after sorting).

Recurrence:
    dp[0] = 0
    dp[i] = max( dp[i-1], profit[i] + dp[p(i)] )

dp[i-1]            -> skip job i
profit[i] + dp[p(i)] -> take job i, add best profit from jobs that
                        finish before job i starts
"""
from bisect import bisect_right
import time
import random


def latest_non_conflicting(ends, start, i):
    """Binary search over precomputed 'ends' array: last index j < i
    such that ends[j] <= start. O(log n) per call."""
    lo, hi = 0, i - 1
    pos = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if ends[mid] <= start:
            pos = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return pos


def weighted_job_scheduling(jobs):
    """
    jobs: list of (start, end, profit)
    returns: (max_profit, list_of_selected_jobs)
    Time complexity: O(n log n)  -- sorting O(n log n) + n binary searches O(log n)
    Space complexity: O(n) for dp table and p() lookups
    """
    if not jobs:
        return 0, []

    jobs = sorted(jobs, key=lambda x: x[1])  # sort by end time
    n = len(jobs)
    dp = [0] * (n + 1)          # dp[i] = best profit using first i jobs (1-indexed)
    choice = [False] * (n + 1)  # did we take job i?
    p = [0] * n
    ends = [job[1] for job in jobs]

    for i in range(n):
        p[i] = latest_non_conflicting(ends, jobs[i][0], i)

    for i in range(1, n + 1):
        job = jobs[i - 1]
        include_profit = job[2] + (dp[p[i - 1] + 1] if p[i - 1] != -1 else 0)
        exclude_profit = dp[i - 1]
        if include_profit > exclude_profit:
            dp[i] = include_profit
            choice[i] = True
        else:
            dp[i] = exclude_profit
            choice[i] = False

    # backtrack to find selected jobs
    selected = []
    i = n
    while i > 0:
        if choice[i]:
            selected.append(jobs[i - 1])
            i = p[i - 1] + 1
        else:
            i -= 1
    selected.reverse()
    return dp[n], selected


def generate_jobs(n, seed=0):
    random.seed(seed)
    jobs = []
    for _ in range(n):
        s = random.randint(0, 1000)
        e = s + random.randint(1, 50)
        profit = random.randint(1, 100)
        jobs.append((s, e, profit))
    return jobs


if __name__ == "__main__":
    # Worked example (classic textbook instance)
    example = [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4), (5, 8, 11), (7, 9, 2)]
    profit, chosen = weighted_job_scheduling(example)
    print("Example jobs (start, end, profit):", example)
    print("Max profit:", profit)
    print("Selected jobs:", chosen)
    print()

    # Empirical timing
    for n in [100, 1000, 10000]:
        jobs = generate_jobs(n)
        t0 = time.perf_counter()
        weighted_job_scheduling(jobs)
        t1 = time.perf_counter()
        print(f"n={n:6d}  time={t1 - t0:.6f}s")
