"""
city.py

Defines the City data record used across all data structures in Task 1
(BST, AVL Tree, Min-Heap, Hash Table) for the route-planning application.
Cities are named after real places in Nepal (see nepal_cities.py).

Each city carries:
    - city_id            : unique integer identifier (used as the Hash
                            Table key)
    - name                : city name, e.g. "Kathmandu", "Dharan"
    - latitude/longitude  : coordinates
    - population          : integer population
    - distance            : distance (km) from a reference origin point
                             (e.g. Kathmandu). This field is used as the
                             ORDERING KEY for the BST / AVL Tree (so the
                             tree stores cities ordered by how far they
                             are from the route's start point) and as the
                             PRIORITY KEY for the Min-Heap (so the heap
                             always returns the next nearest unvisited
                             city).
"""

from dataclasses import dataclass


@dataclass
class City:
    city_id: int
    name: str
    latitude: float
    longitude: float
    population: int
    distance: float  # key used for BST/AVL ordering and Min-Heap priority

    def __repr__(self):
        return (f"City(id={self.city_id}, name='{self.name}', "
                f"dist={self.distance:.2f}km, pop={self.population:,})")

    def __lt__(self, other):
        # Enables direct comparison between City objects using `distance`
        return self.distance < other.distance

    def __eq__(self, other):
        if not isinstance(other, City):
            return NotImplemented
        return self.city_id == other.city_id
