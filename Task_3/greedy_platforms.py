"""
Task 3 - Greedy Algorithm
Minimum Number of Platforms

Problem: Given arrival and departure times of trains at a station,
find the minimum number of platforms needed so that no train waits.

Greedy choice: sort arrivals and departures separately. Walk through
events in time order; every arrival that occurs before the earliest
still-active departure needs one more platform, every departure frees
a platform. Track the maximum number of platforms in use at once.

This is equivalent to interval-point sweeping and is a classic case
where a greedy strategy is PROVABLY optimal (unlike Task 3's weighted
interval scheduling, where greedy is optimal only for the unweighted
version, not the weighted one).
"""
import time
import random


def min_platforms_greedy(arrivals, departures):
    """
    O(n log n): sorting dominates; the sweep itself is O(n).
    Space: O(1) extra (aside from sorted copies).
    """
    arr = sorted(arrivals)
    dep = sorted(departures)
    n = len(arr)

    platforms_needed = 0
    max_platforms = 0
    i, j = 0, 0
    while i < n and j < n:
        if arr[i] <= dep[j]:
            platforms_needed += 1
            i += 1
            max_platforms = max(max_platforms, platforms_needed)
        else:
            platforms_needed -= 1
            j += 1
    return max_platforms


def min_platforms_bruteforce(arrivals, departures):
    """
    Exact / brute-force check: for every train's arrival instant,
    count how many trains are present at that instant.
    O(n^2) - used only to verify the greedy result on small inputs.
    """
    n = len(arrivals)
    max_count = 0
    for i in range(n):
        count = 0
        t = arrivals[i]
        for j in range(n):
            if arrivals[j] <= t <= departures[j]:
                count += 1
        max_count = max(max_count, count)
    return max_count


def generate_schedule(n, seed=0):
    random.seed(seed)
    arrivals, departures = [], []
    for _ in range(n):
        a = random.randint(0, 2000)
        d = a + random.randint(1, 60)
        arrivals.append(a)
        departures.append(d)
    return arrivals, departures


if __name__ == "__main__":
    arrivals = [900, 940, 950, 1100, 1500, 1800]
    departures = [910, 1200, 1120, 1130, 1900, 2000]
    print("Arrivals:", arrivals)
    print("Departures:", departures)
    print("Greedy min platforms:", min_platforms_greedy(arrivals, departures))
    print("Brute-force min platforms:", min_platforms_bruteforce(arrivals, departures))
    print()

    # Correctness check against brute force on random small instances
    all_match = True
    for trial in range(200):
        a, d = generate_schedule(30, seed=trial)
        if min_platforms_greedy(a, d) != min_platforms_bruteforce(a, d):
            all_match = False
            print("MISMATCH on trial", trial)
    print("Greedy matches brute force on all 200 random trials:", all_match)
    print()

    for n in [100, 1000, 10000]:
        a, d = generate_schedule(n)
        t0 = time.perf_counter()
        min_platforms_greedy(a, d)
        t1 = time.perf_counter()
        print(f"n={n:6d}  greedy time={t1 - t0:.6f}s")
