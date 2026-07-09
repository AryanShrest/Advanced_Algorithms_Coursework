"""
Task 3 - Backtracking
Knight's Tour

Problem: find a sequence of knight moves on an n x n board that visits
every square exactly once (open tour).

Plain backtracking tries each of up to 8 moves from the current square,
recurses, and undoes ("un-visits") the square if the recursive call
fails. Worst case this explores O(8^(n^2)) partial paths.

Pruning strategy implemented (Warnsdorff's rule): at each step, among
the legal next moves, try the square with the FEWEST onward moves
first. This greedily steers the search toward corners/edges (which
have few options) and drastically cuts the branching factor in
practice, even though the worst-case bound is unchanged.
"""
import time

MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
         (-2, -1), (-1, -2), (1, -2), (2, -1)]


def is_valid(n, x, y, board):
    return 0 <= x < n and 0 <= y < n and board[x][y] == -1


def onward_degree(n, x, y, board):
    """Count legal moves from (x, y) - used for Warnsdorff ordering."""
    count = 0
    for dx, dy in MOVES:
        if is_valid(n, x + dx, y + dy, board):
            count += 1
    return count


def knights_tour(n, start=(0, 0), use_pruning=True):
    """
    Returns (board, stats) where board[x][y] = move index (0..n*n-1),
    or None if no tour was found. stats records nodes explored.
    Worst case time: O(8^(n^2))  (exponential - unavoidable in general)
    With Warnsdorff pruning the search finishes in near-linear practical
    time for boards commonly used (n <= ~50), though no polynomial
    worst-case guarantee exists.
    Space: O(n^2) for the board + O(n^2) recursion depth.
    """
    board = [[-1] * n for _ in range(n)]
    stats = {"nodes_explored": 0}
    sx, sy = start
    board[sx][sy] = 0

    def backtrack(x, y, move_count):
        stats["nodes_explored"] += 1
        if move_count == n * n:
            return True

        candidates = []
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if is_valid(n, nx, ny, board):
                candidates.append((nx, ny))

        if use_pruning:
            # Warnsdorff: try squares with fewest onward options first
            candidates.sort(key=lambda p: onward_degree(n, p[0], p[1], board))

        for nx, ny in candidates:
            board[nx][ny] = move_count
            if backtrack(nx, ny, move_count + 1):
                return True
            board[nx][ny] = -1  # undo - this is the "backtrack" step

        return False

    found = backtrack(sx, sy, 1)
    return (board if found else None), stats


def print_board(board):
    n = len(board)
    for row in board:
        print(" ".join(f"{v:3d}" for v in row))


if __name__ == "__main__":
    for n in [5, 6, 8]:
        for pruning in [True, False]:
            t0 = time.perf_counter()
            board, stats = knights_tour(n, use_pruning=pruning)
            t1 = time.perf_counter()
            label = "Warnsdorff pruning" if pruning else "plain backtracking"
            status = "found" if board else "NOT found"
            print(f"n={n} ({label}): tour {status}, "
                  f"nodes_explored={stats['nodes_explored']}, "
                  f"time={t1 - t0:.4f}s")
        print()

    print("Sample 6x6 tour (Warnsdorff pruning):")
    board, _ = knights_tour(6, use_pruning=True)
    if board:
        print_board(board)
