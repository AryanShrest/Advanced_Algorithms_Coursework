"""
bst.py

Unbalanced Binary Search Tree (BST) keyed on City.distance.

Theoretical complexity (n = number of nodes):
    Average case (random insert order): O(log n) for insert/search/delete
    Worst case  (sorted/degenerate input): O(n)  -- tree degenerates to a
                                                     linked list

This class also tracks height so that the empirical benchmark can report
how skewed the tree became for the random datasets used in Task 1.
"""


class BSTNode:
    __slots__ = ("city", "left", "right")

    def __init__(self, city):
        self.city = city
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    # ---------------------------------------------------------------- #
    # Insert
    # ---------------------------------------------------------------- #
    def insert(self, city):
        self.root = self._insert(self.root, city)
        self._size += 1

    def _insert(self, node, city):
        if node is None:
            return BSTNode(city)
        if city.distance < node.city.distance:
            node.left = self._insert(node.left, city)
        else:
            node.right = self._insert(node.right, city)
        return node

    # ---------------------------------------------------------------- #
    # Search
    # ---------------------------------------------------------------- #
    def search(self, distance):
        """Return the City with the given distance key, or None."""
        node = self.root
        while node is not None:
            if distance == node.city.distance:
                return node.city
            node = node.left if distance < node.city.distance else node.right
        return None

    # ---------------------------------------------------------------- #
    # Delete
    # ---------------------------------------------------------------- #
    def delete(self, distance):
        self.root, deleted = self._delete(self.root, distance)
        if deleted:
            self._size -= 1
        return deleted

    def _delete(self, node, distance):
        if node is None:
            return node, False

        if distance < node.city.distance:
            node.left, deleted = self._delete(node.left, distance)
        elif distance > node.city.distance:
            node.right, deleted = self._delete(node.right, distance)
        else:
            deleted = True
            if node.left is None:
                return node.right, deleted
            if node.right is None:
                return node.left, deleted
            # Two children: replace with in-order successor
            successor = self._min_node(node.right)
            node.city = successor.city
            node.right, _ = self._delete(node.right, successor.city.distance)

        return node, deleted

    def _min_node(self, node):
        while node.left is not None:
            node = node.left
        return node

    # ---------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------- #
    def height(self):
        return self._height(self.root)

    def _height(self, node):
        if node is None:
            return -1
        return 1 + max(self._height(node.left), self._height(node.right))

    def inorder(self):
        """Return cities sorted by distance (in-order traversal)."""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node is not None:
            self._inorder(node.left, result)
            result.append(node.city)
            self._inorder(node.right, result)


# ---------------------------------------------------------------------- #
# Demo / smoke test with clear, labelled output
# Cities are real Nepali towns, with approximate road distance (km) from
# Kathmandu used as the ordering key.
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from city import City

    print("=" * 60)
    print("  BINARY SEARCH TREE (BST) -- DEMO")
    print("  (cities of Nepal, keyed on distance from Kathmandu)")
    print("=" * 60)

    NEPAL_SAMPLE = [
        ("Lalitpur", 5), ("Bhaktapur", 13), ("Pokhara", 200),
        ("Butwal", 278), ("Itahari", 503), ("Dharan", 525),
        ("Biratnagar", 547),
    ]
    sample = [City(i, name, 0, 0, 100_000, d)
              for i, (name, d) in enumerate(NEPAL_SAMPLE)]

    print(f"\nInserting {len(sample)} cities (name, distance from Kathmandu in km):")
    for name, d in NEPAL_SAMPLE:
        print(f"   {name:<12} {d} km")

    tree = BST()
    for c in sample:
        tree.insert(c)

    print("\n[1] In-order traversal (should be sorted by distance):")
    for c in tree.inorder():
        print(f"      {c}")

    print(f"\n[2] Tree height after inserts: {tree.height()}")
    print("      Note: these cities were listed in ascending order of "
          "distance, so inserting them in this order is the BST's WORST "
          "CASE (sorted input) -- the tree degenerates towards a linked "
          "list. Compare this height against avl_tree.py's demo, which "
          "inserts the same kind of sorted data but stays balanced.")

    print("\n[3] Search for the city at distance = 278 km (Butwal):")
    result = tree.search(278)
    print(f"      Found -> {result}" if result else "      Not found")

    print("\n[4] Delete the city at distance = 13 km (Bhaktapur):")
    ok = tree.delete(13)
    print(f"      Deletion successful: {ok}")

    print("\n[5] In-order traversal after deletion:")
    for c in tree.inorder():
        print(f"      {c}")

    print("\n" + "=" * 60)
    print("  END OF BST DEMO")
    print("=" * 60)
