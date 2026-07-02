"""
min_heap.py

Array-based binary Min-Heap, used as a priority queue keyed on
City.distance -- e.g. "always give me the next nearest unvisited city".

Theoretical complexity (n = number of elements currently in the heap):
    push (insert)          : O(log n)
    pop (extract-min)      : O(log n)
    peek (read min)        : O(1)
    decrease_key           : O(log n)  (needed for Dijkstra-style relaxation)
    build_heap from list   : O(n)      (heapify, cheaper than n inserts)

A heap gives O(log n) insert/extract-min but, unlike a BST/AVL tree, does
NOT support O(log n) arbitrary search or in-order retrieval -- searching
for an arbitrary city is O(n). This asymmetry is important for Task 1's
"justify which structure fits which use case" discussion: heaps are ideal
for repeated "give me the nearest city" queries (Dijkstra-style routing),
not for lookups by id/name.
"""


class MinHeap:
    def __init__(self):
        self._heap = []  # list of City objects, heap-ordered by .distance
        # position index for O(log n) decrease_key: city_id -> index in heap
        self._pos = {}

    def __len__(self):
        return len(self._heap)

    def is_empty(self):
        return len(self._heap) == 0

    # ---------------------------------------------------------------- #
    # Core heap operations
    # ---------------------------------------------------------------- #
    def push(self, city):
        self._heap.append(city)
        idx = len(self._heap) - 1
        self._pos[city.city_id] = idx
        self._sift_up(idx)

    def pop(self):
        """Remove and return the city with the smallest distance."""
        if not self._heap:
            raise IndexError("pop from empty heap")

        top = self._heap[0]
        last = self._heap.pop()
        del self._pos[top.city_id]

        if self._heap:
            self._heap[0] = last
            self._pos[last.city_id] = 0
            self._sift_down(0)

        return top

    def peek(self):
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def decrease_key(self, city_id, new_distance):
        """Lower a city's distance (used when a shorter route is found)."""
        idx = self._pos.get(city_id)
        if idx is None:
            raise KeyError(f"city_id {city_id} not in heap")
        if new_distance > self._heap[idx].distance:
            raise ValueError("new_distance is greater than current distance")
        self._heap[idx].distance = new_distance
        self._sift_up(idx)

    def build_heap(self, cities):
        """O(n) heap construction from an existing list of cities."""
        self._heap = list(cities)
        self._pos = {c.city_id: i for i, c in enumerate(self._heap)}
        for i in reversed(range(len(self._heap) // 2)):
            self._sift_down(i)

    # ---------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------- #
    def _swap(self, i, j):
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._pos[self._heap[i].city_id] = i
        self._pos[self._heap[j].city_id] = j

    def _sift_up(self, idx):
        while idx > 0:
            parent = (idx - 1) // 2
            if self._heap[idx].distance < self._heap[parent].distance:
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(self, idx):
        n = len(self._heap)
        while True:
            left, right = 2 * idx + 1, 2 * idx + 2
            smallest = idx
            if left < n and self._heap[left].distance < self._heap[smallest].distance:
                smallest = left
            if right < n and self._heap[right].distance < self._heap[smallest].distance:
                smallest = right
            if smallest == idx:
                break
            self._swap(idx, smallest)
            idx = smallest


# ---------------------------------------------------------------------- #
# Demo / smoke test with clear, labelled output
# Cities are real Nepali towns, with approximate road distance (km) from
# Kathmandu used as the priority key -- simulating "next nearest city to
# visit" on a route-planning trip.
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from city import City

    print("=" * 60)
    print("  MIN-HEAP (PRIORITY QUEUE) -- DEMO")
    print("  (cities of Nepal, keyed on distance from Kathmandu)")
    print("=" * 60)

    NEPAL_SAMPLE = [
        ("Lalitpur", 5), ("Bhaktapur", 13), ("Pokhara", 200),
        ("Butwal", 278), ("Itahari", 503), ("Dharan", 525),
        ("Biratnagar", 547),
    ]
    sample = [City(i, name, 0, 0, 100_000, d)
              for i, (name, d) in enumerate(NEPAL_SAMPLE)]

    print(f"\nPushing {len(sample)} cities (name, distance from Kathmandu in km):")
    for name, d in NEPAL_SAMPLE:
        print(f"   {name:<12} {d} km")

    heap = MinHeap()
    for c in sample:
        heap.push(c)

    print("\n[1] Peek at nearest city (should not remove it):")
    print(f"      {heap.peek()}")

    print("\n[2] Popping all cities -- should come out nearest-first "
          "(i.e. the order you'd actually visit them on a route):")
    order = 1
    while not heap.is_empty():
        city = heap.pop()
        print(f"      {order}. {city}")
        order += 1

    print("\n" + "=" * 60)
    print("  END OF MIN-HEAP DEMO")
    print("=" * 60)
