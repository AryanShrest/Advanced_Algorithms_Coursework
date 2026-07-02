"""
avl_tree.py

Self-balancing AVL Tree keyed on City.distance.

Theoretical complexity (n = number of nodes):
    Insert / Delete / Search: O(log n) WORST CASE (guaranteed), because the
    AVL invariant (|balance factor| <= 1 at every node) is restored after
    every insert/delete via rotations, keeping the tree height at
    O(log n) even for adversarial/sorted input -- unlike a plain BST.

The constant factor is higher than a plain BST per-operation (extra height
bookkeeping + potential rotations), which is exactly the kind of
"hidden constant" trade-off Task 1 asks you to discuss: AVL guarantees
O(log n) but with more work per node than an unbalanced BST's *average*
case.
"""


class AVLNode:
    __slots__ = ("city", "left", "right", "height")

    def __init__(self, city):
        self.city = city
        self.left = None
        self.right = None
        self.height = 1  # height of subtree rooted here (leaf = 1)


class AVLTree:
    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    def _h(self, node):
        return node.height if node else 0

    def _balance_factor(self, node):
        return self._h(node.left) - self._h(node.right) if node else 0

    def _update_height(self, node):
        node.height = 1 + max(self._h(node.left), self._h(node.right))

    def _rotate_right(self, y):
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x):
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node):
        self._update_height(node)
        balance = self._balance_factor(node)

        # Left heavy
        if balance > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)  # Left-Right case
            return self._rotate_right(node)  # Left-Left case

        # Right heavy
        if balance < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)  # Right-Left case
            return self._rotate_left(node)  # Right-Right case

        return node

    # ---------------------------------------------------------------- #
    # Insert
    # ---------------------------------------------------------------- #
    def insert(self, city):
        self.root = self._insert(self.root, city)
        self._size += 1

    def _insert(self, node, city):
        if node is None:
            return AVLNode(city)

        if city.distance < node.city.distance:
            node.left = self._insert(node.left, city)
        else:
            node.right = self._insert(node.right, city)

        return self._rebalance(node)

    # ---------------------------------------------------------------- #
    # Search
    # ---------------------------------------------------------------- #
    def search(self, distance):
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
            successor = self._min_node(node.right)
            node.city = successor.city
            node.right, _ = self._delete(node.right, successor.city.distance)

        if node is None:
            return node, deleted

        return self._rebalance(node), deleted

    def _min_node(self, node):
        while node.left is not None:
            node = node.left
        return node

    # ---------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------- #
    def height(self):
        return self._h(self.root) - 1 if self.root else -1

    def inorder(self):
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
    print("  AVL TREE -- DEMO")
    print("  (cities of Nepal, keyed on distance from Kathmandu)")
    print("=" * 60)

    NEPAL_SAMPLE = [
        ("Kathmandu", 0), ("Lalitpur", 5), ("Bhaktapur", 13),
        ("Pokhara", 200), ("Butwal", 278), ("Itahari", 503),
        ("Dharan", 525), ("Biratnagar", 547), ("Dhangadhi", 669),
    ]
    sample = [City(i, name, 0, 0, 100_000, d)
              for i, (name, d) in enumerate(NEPAL_SAMPLE)]

    print(f"\nInserting {len(sample)} cities (name, distance from Kathmandu in km):")
    for name, d in NEPAL_SAMPLE:
        print(f"   {name:<12} {d} km")

    tree = AVLTree()
    for c in sample:
        tree.insert(c)

    print("\n[1] In-order traversal (should be sorted by distance):")
    for c in tree.inorder():
        print(f"      {c}")

    print(f"\n[2] Tree height after inserts: {tree.height()}")
    print("      (stays balanced -- compare against bst.py's height "
          "for the same-size input)")

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
    print("  END OF AVL TREE DEMO")
    print("=" * 60)
