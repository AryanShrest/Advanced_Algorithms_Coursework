import random
import time


def min_platforms_greedy(arrivals, departures):
    sorted_arrivals = sorted(arrivals)
    sorted_departures = sorted(departures)

    total = len(sorted_arrivals)

    arrival_index = 0
    departure_index = 0

    platforms = 0
    maximum = 0

    while arrival_index < total and departure_index < total:

        if sorted_arrivals[arrival_index] <= sorted_departures[departure_index]:
            platforms += 1
            maximum = max(maximum, platforms)
            arrival_index += 1
        else:
            platforms -= 1
            departure_index += 1

    return maximum


def min_platforms_bruteforce(arrivals, departures):
    total = len(arrivals)
    answer = 0

    for i in range(total):
        current_time = arrivals[i]
        overlap = 0

        for j in range(total):
            if arrivals[j] <= current_time <= departures[j]:
                overlap += 1

        answer = max(answer, overlap)

    return answer


def generate_schedule(size, seed=0):
    random.seed(seed)

    arrival_times = []
    departure_times = []

    for _ in range(size):
        arrival = random.randint(0, 2000)
        departure = arrival + random.randint(1, 60)

        arrival_times.append(arrival)
        departure_times.append(departure)

    return arrival_times, departure_times