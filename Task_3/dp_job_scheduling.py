"""
Task 3 - Dynamic Programming
Weighted Job Scheduling with Time Windows
"""

import time
import random


def find_previous_job(end_times, current_start):
    """Returns the index of the last non-overlapping job."""
    left = 0
    right = len(end_times) - 1
    answer = -1

    while left <= right:
        middle = (left + right) // 2

        if end_times[middle] <= current_start:
            answer = middle
            left = middle + 1
        else:
            right = middle - 1

    return answer


def weighted_job_scheduling(job_list):
    if len(job_list) == 0:
        return 0, []

    # Sort jobs by finishing time
    sorted_jobs = sorted(job_list, key=lambda job: job[1])

    total_jobs = len(sorted_jobs)
    end_times = [job[1] for job in sorted_jobs]

    # Store previous compatible job index
    previous = []
    for i in range(total_jobs):
        previous.append(find_previous_job(end_times[:i], sorted_jobs[i][0]))

    # DP arrays
    max_profit = [0] * total_jobs
    selected = [False] * total_jobs

    for i in range(total_jobs):

        current_profit = sorted_jobs[i][2]

        if previous[i] != -1:
            current_profit += max_profit[previous[i]]

        profit_without_current = max_profit[i - 1] if i > 0 else 0

        if current_profit > profit_without_current:
            max_profit[i] = current_profit
            selected[i] = True
        else:
            max_profit[i] = profit_without_current

    # Recover selected jobs
    chosen_jobs = []
    index = total_jobs - 1

    while index >= 0:

        if selected[index]:
            chosen_jobs.append(sorted_jobs[index])
            index = previous[index]
        else:
            index -= 1

    chosen_jobs.reverse()

    return max_profit[-1], chosen_jobs


def generate_jobs(n, seed=0):
    random.seed(seed)

    jobs = []

    for _ in range(n):
        start = random.randint(0, 1000)
        end = start + random.randint(1, 50)
        profit = random.randint(1, 100)

        jobs.append((start, end, profit))

    return jobs


if __name__ == "__main__":

    example_jobs = [
        (1, 3, 5),
        (2, 5, 6),
        (4, 6, 5),
        (6, 7, 4),
        (5, 8, 11),
        (7, 9, 2)
    ]

    best_profit, selected_jobs = weighted_job_scheduling(example_jobs)

    print("Example Jobs:")
    print(example_jobs)
    print("Maximum Profit:", best_profit)
    print("Chosen Jobs:", selected_jobs)
    print()

    for size in [100, 1000, 10000]:
        jobs = generate_jobs(size)

        start_time = time.perf_counter()
        weighted_job_scheduling(jobs)
        end_time = time.perf_counter()

        print(f"n = {size:5d}   Time = {end_time - start_time:.6f} seconds")