"""
demo_nepal_network.py

Builds a small, realistic Nepal city network and produces the
VISUALISATIONS required by Task 2's analysis section:

    - dijkstra_tree.png            : final shortest-path tree from
                                      Kathmandu, drawn over the network
    - prim_mst_steps.png           : 2x2 grid showing the Minimum
                                      Spanning Tree being built up
                                      step-by-step (25% / 50% / 75% / 100%
                                      of edges added)
    - bellman_ford_convergence.png : line chart showing shortest-distance
                                      estimates converging round-by-round
                                      (demonstrates *why* Bellman-Ford
                                      needs up to V-1 rounds, unlike
                                      Dijkstra's single pass)

Run with:  python3 demo_nepal_network.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from graph import Graph
from dijkstra import dijkstra
from prim import prim_mst
from bellman_ford import bellman_ford

# Approximate road distances (km) between real Nepal cities/towns.
# Directed edges -- in practice most roads are bidirectional with the
# same cost, but a few asymmetric weights are included (e.g. modelling a
# toll or a longer one-way diversion) to keep this genuinely a directed
# graph, as the coursework specifies.
NEPAL_EDGES = [
    ("Kathmandu", "Lalitpur", 5),   ("Lalitpur", "Kathmandu", 5),
    ("Kathmandu", "Bhaktapur", 13), ("Bhaktapur", "Kathmandu", 13),
    ("Lalitpur", "Bhaktapur", 10),  ("Bhaktapur", "Lalitpur", 12),  # asymmetric (diversion)
    ("Kathmandu", "Pokhara", 200),  ("Pokhara", "Kathmandu", 200),
    ("Pokhara", "Butwal", 150),     ("Butwal", "Pokhara", 150),
    ("Kathmandu", "Hetauda", 135),  ("Hetauda", "Kathmandu", 135),
    ("Hetauda", "Butwal", 160),     ("Butwal", "Hetauda", 160),
    ("Butwal", "Bhairahawa", 25),   ("Bhairahawa", "Butwal", 25),
    ("Butwal", "Itahari", 400),     ("Itahari", "Butwal", 410),     # asymmetric
    ("Kathmandu", "Itahari", 503),  ("Itahari", "Kathmandu", 503),
    ("Itahari", "Dharan", 25),      ("Dharan", "Itahari", 25),
    ("Itahari", "Biratnagar", 22),  ("Biratnagar", "Itahari", 22),
    ("Dharan", "Biratnagar", 45),   ("Biratnagar", "Dharan", 45),
]


def build_graph():
    g = Graph(directed=True)
    for u, v, w in NEPAL_EDGES:
        g.add_edge(u, v, w)
    return g


def draw_dijkstra_tree(g, source="Kathmandu", path="dijkstra_tree.png"):
    dist, prev, steps = dijkstra(g, source, record_steps=True)

    G = nx.DiGraph()
    for u, v, w in g.edges():
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=7, k=1.4)

    tree_edges = {(p, c) for c, p in prev.items()}

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", arrows=True,
                            connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edges(G, pos, edgelist=list(tree_edges), edge_color="crimson",
                            width=2.5, arrows=True, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(G, pos, node_color="#4C72B0", node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color="#DD8452",
                            node_size=1000)
    labels = {n: f"{n}\n{dist[n]}km" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="white")

    plt.title(f"Dijkstra Shortest-Path Tree from {source}\n"
              "(red edges = shortest-path tree; grey = other network edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")
    return dist, steps


def draw_prim_steps(g, source="Kathmandu", path="prim_mst_steps.png"):
    undirected = g.to_undirected()
    mst_edges, total_weight, steps = prim_mst(undirected, source=source,
                                               record_steps=True)

    G = nx.Graph()
    for u, v, w in undirected.edges():
        G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, seed=7, k=1.4)

    checkpoints = [max(1, int(len(steps) * f)) for f in (0.25, 0.5, 0.75, 1.0)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, k in zip(axes.flat, checkpoints):
        edges_so_far = [(u, v) for u, v, w in steps[:k]]
        nodes_so_far = {source}
        for u, v in edges_so_far:
            nodes_so_far.add(u)
            nodes_so_far.add(v)

        nx.draw_networkx_edges(G, pos, edge_color="lightgray", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges_so_far, edge_color="seagreen",
                                width=3, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color="#4C72B0", node_size=500, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=list(nodes_so_far),
                                node_color="seagreen", node_size=550, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=6, font_color="white", ax=ax)
        weight_so_far = sum(w for _, _, w in steps[:k])
        ax.set_title(f"After {k}/{len(steps)} edges "
                      f"(tree weight so far: {weight_so_far} km)")
        ax.axis("off")

    fig.suptitle(f"Prim's MST Construction, step-by-step (source: {source})\n"
                  f"Final MST total weight: {total_weight} km", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")
    return mst_edges, total_weight


def draw_bellman_ford_convergence(path="bellman_ford_convergence.png"):
    # Small graph WITH a negative edge, so Bellman-Ford's multi-round
    # relaxation is actually necessary to show convergence.
    g = Graph(directed=True)
    edges = [
        ("Kathmandu", "Lalitpur", 5),
        ("Kathmandu", "Bhaktapur", 13),
        ("Lalitpur", "Pokhara", 210),
        ("Bhaktapur", "Pokhara", -20),   # hypothetical rebate
        ("Pokhara", "Butwal", 150),
        ("Butwal", "Hetauda", 160),
        ("Hetauda", "Kathmandu", 135),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)

    dist, prev, has_cycle, cycle_edges, history = bellman_ford(g, "Kathmandu")

    plt.figure(figsize=(8, 5.5))
    cities = sorted(g.vertices)
    for city in cities:
        ys = [snap.get(city, float("inf")) for snap in history]
        ys = [y if y != float("inf") else None for y in ys]
        xs = list(range(len(history)))
        plt.plot(xs, ys, marker="o", label=city)

    plt.xlabel("Relaxation round")
    plt.ylabel("Current shortest-distance estimate from Kathmandu")
    plt.title("Bellman-Ford: distance estimates converging round-by-round\n"
              "(estimates keep improving until no edge can be relaxed further)")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")
    return history


if __name__ == "__main__":
    print("=" * 72)
    print("  BUILDING NEPAL NETWORK VISUALISATIONS FOR TASK 2")
    print("=" * 72)

    g = build_graph()
    print(f"\nNetwork: {g}")

    print("\n[1/3] Drawing Dijkstra shortest-path tree...")
    dist, steps = draw_dijkstra_tree(g)

    print("\n[2/3] Drawing Prim's MST step-by-step construction...")
    mst_edges, total_weight = draw_prim_steps(g)

    print("\n[3/3] Drawing Bellman-Ford convergence chart...")
    history = draw_bellman_ford_convergence()

    print("\nAll visualisations generated.")
