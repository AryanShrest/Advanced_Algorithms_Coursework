"""
hash_table.py

Hash Table keyed on City.city_id, using SEPARATE CHAINING for collision
handling (each bucket is a Python list of (key, city) pairs). The table
auto-resizes (doubles) once the load factor exceeds a threshold, keeping
operations close to their theoretical average-case cost as n grows.

Theoretical complexity (n = number of entries, m = number of buckets,
load factor alpha = n / m):
    Average case (good hash function, alpha kept low): O(1) for
        insert / search / delete
    Worst case (all keys collide into one bucket): O(n)

For fast *city lookup by id* (the use case this coursework asks for),
a well-sized hash table beats BST/AVL's O(log n) with O(1) average time --
but offers no ordering, so it can't support "give me the nearest city"
style range/priority queries the way a tree or heap can.
"""


class HashTable:
    def __init__(self, initial_capacity=16, load_factor_threshold=0.75):
        self._capacity = initial_capacity
        self._threshold = load_factor_threshold
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0

    def __len__(self):
        return self._size

    def load_factor(self):
        return self._size / self._capacity

    def _hash(self, key):
        # Python's built-in hash() is fine for ints/strings; mod capacity
        return hash(key) % self._capacity

    # ---------------------------------------------------------------- #
    # Insert
    # ---------------------------------------------------------------- #
    def insert(self, key, city):
        """Insert or update the city stored under `key` (city_id)."""
        idx = self._hash(key)
        bucket = self._buckets[idx]

        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, city)  # update existing
                return

        bucket.append((key, city))
        self._size += 1

        if self.load_factor() > self._threshold:
            self._resize()

    # ---------------------------------------------------------------- #
    # Search
    # ---------------------------------------------------------------- #
    def search(self, key):
        idx = self._hash(key)
        for k, city in self._buckets[idx]:
            if k == key:
                return city
        return None

    # ---------------------------------------------------------------- #
    # Delete
    # ---------------------------------------------------------------- #
    def delete(self, key):
        idx = self._hash(key)
        bucket = self._buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self._size -= 1
                return True
        return False

    # ---------------------------------------------------------------- #
    # Resizing
    # ---------------------------------------------------------------- #
    def _resize(self):
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        old_size = self._size
        self._size = 0
        for bucket in old_buckets:
            for key, city in bucket:
                self.insert(key, city)
        self._size = old_size

    # ---------------------------------------------------------------- #
    # Diagnostics (useful for the Task 1 analysis section)
    # ---------------------------------------------------------------- #
    def max_bucket_length(self):
        """Longest chain -- indicates how close we are to worst case."""
        return max((len(b) for b in self._buckets), default=0)

    def collision_stats(self):
        non_empty = [len(b) for b in self._buckets if b]
        return {
            "capacity": self._capacity,
            "size": self._size,
            "load_factor": round(self.load_factor(), 3),
            "non_empty_buckets": len(non_empty),
            "max_chain_length": max(non_empty, default=0),
            "avg_chain_length_when_used": (
                round(sum(non_empty) / len(non_empty), 3) if non_empty else 0
            ),
        }


# ---------------------------------------------------------------------- #
# Demo / smoke test with clear, labelled output
# Cities are real Nepali towns; the hash table is keyed on city_id so a
# route-planning app can do instant "look up this city by ID" queries.
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from city import City

    print("=" * 60)
    print("  HASH TABLE (SEPARATE CHAINING) -- DEMO")
    print("  (cities of Nepal, keyed on city_id)")
    print("=" * 60)

    NEPAL_SAMPLE = [
        ("Lalitpur", 5), ("Bhaktapur", 13), ("Pokhara", 200),
        ("Butwal", 278), ("Itahari", 503), ("Dharan", 525),
        ("Biratnagar", 547),
    ]
    sample = [City(i, name, 0, 0, 100_000, d)
              for i, (name, d) in enumerate(NEPAL_SAMPLE)]

    print(f"\nInserting {len(sample)} cities (initial capacity = 4, "
          "so a resize should trigger):")
    table = HashTable(initial_capacity=4)
    for c in sample:
        table.insert(c.city_id, c)
        print(f"      inserted id={c.city_id} ({c.name:<12}) "
              f"-> load factor now {table.load_factor():.2f}")

    print("\n[1] Search for city_id = 3 (Butwal):")
    result = table.search(3)
    print(f"      Found -> {result}" if result else "      Not found")

    print("\n[2] Delete city_id = 3 (Butwal):")
    ok = table.delete(3)
    print(f"      Deletion successful: {ok}")

    print("\n[3] Search for city_id = 3 again (should be gone):")
    result = table.search(3)
    print(f"      Found -> {result}" if result else "      Not found (correct)")

    print("\n[4] Collision / load statistics:")
    for key, value in table.collision_stats().items():
        print(f"      {key:>26}: {value}")

    print("\n" + "=" * 60)
    print("  END OF HASH TABLE DEMO")
    print("=" * 60)
