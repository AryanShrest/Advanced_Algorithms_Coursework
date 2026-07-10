"""
Task 3c - Backtracking
Problem: Knight's Tour

Find a sequence of moves for a chess knight on an n x n board so that it
visits every square exactly once (an "open" tour; a "closed" tour also
returns to the starting square with one final legal knight move).

--------------------------------------------------------------------------
BASIC BACKTRACKING FORMULATION
--------------------------------------------------------------------------
State: current square, set of visited squares, move count so far.
At each step, try each of the (up to 8) legal knight moves from the
current square that lands on an unvisited square still on the board.
Recurse; if a branch reaches n*n visited squares, a full tour is found.
If a branch runs out of legal moves before visiting all squares,
backtrack (undo the last move) and try the next candidate move.

--------------------------------------------------------------------------
PRUNING STRATEGY: WARNSDORFF'S RULE
--------------------------------------------------------------------------
Plain backtracking explores moves in a fixed (e.g. clockwise) order,
which on boards larger than about 5x5 becomes impractically slow because
the search tree is explored almost blindly and deep dead-ends are only
discovered after many wasted moves.

Warnsdorff's heuristic re-orders move candidates at every step: always
try the legal move that leads to the square with the FEWEST onward moves
(its "accessibility" / degree) first. Squares near the edges/corners of
the board have fewer legal knight moves, so visiting them earlier avoids
stranding the knight in a corner with no unvisited neighbours late in
the tour - this is the dominant failure mode of naive move ordering.

This is a pruning/ordering heuristic, not a change to the state space:
it does not, by itself, guarantee a solution exists on every board, but
it converts an exponential blind search into one that finds a full tour
almost immediately for the vast majority of board sizes and starting
squares in practice. We combine it with a genuine backtracking
fallback: if Warnsdorff ordering leads to a dead end, the algorithm
still backtracks and tries the next-best candidate (rather than
committing irrevocably), so correctness is preserved.

Two additional pruning rules are also applied:
  1. Immediate legality/bounds check before recursing (never enter an
     invalid state at all - avoids wasted stack frames).
  2. Early degree-zero cut: if, after the heuristic reorder, the current
     square already has zero legal onward moves and unvisited squares
     remain, we backtrack immediately rather than exploring further.

--------------------------------------------------------------------------
COMPLEXITY ANALYSIS
--------------------------------------------------------------------------
Without pruning:
  Each square offers at most 8 candidate moves, and a tour has n^2 - 1
  moves, so the naive worst-case search tree size is O(8^(n^2)) -
  genuinely exponential in the number of squares. This is why plain
  backtracking without move ordering becomes infeasible even for modest
  boards such as 6x6 or 7x7.

With Warnsdorff pruning:
  The heuristic reduces the *effective* branching factor close to 1 in
  the common case (almost always exactly one clearly-best move is
  available), so empirically the algorithm runs in close to O(n^2) time
  for most board sizes - a reduction from exponential to near-linear-in-
  squares behaviour in practice, even though no polynomial-time worst-
  case guarantee exists (adversarial boards/starting squares can still
  force backtracking). This gap between worst-case and observed
  behaviour is exactly the "hidden constant / practical significance"
  point required by the assignment: Big-O alone (O(8^(n^2)) worst case)
  drastically overstates the cost actually paid on almost all inputs
  once pruning/ordering is applied.

Space complexity: O(n^2) for the visited-board matrix and the O(n^2)-
long move list, plus O(n^2) recursion depth in the worst case.
"""

from __future__ import annotations
import time
from typing import List, Optional, Tuple

MOVES = [(1, 2), (2, 1), (2, -1), (1, -2),
         (-1, -2), (-2, -1), (-2, 1), (-1, 2)]


def _on_board(x: int, y: int, n: int) -> bool:
    return 0 <= x < n and 0 <= y < n


def _degree(x: int, y: int, n: int, visited: List[List[bool]]) -> int:
    """Number of legal onward knight moves from (x, y) - used by
    Warnsdorff's rule to rank candidate squares."""
    count = 0
    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy
        if _on_board(nx, ny, n) and not visited[nx][ny]:
            count += 1
    return count


def knights_tour(n: int, start: Tuple[int, int] = (0, 0),
                  use_warnsdorff: bool = True) -> Optional[List[Tuple[int, int]]]:
    """Backtracking search for an open knight's tour on an n x n board.

    use_warnsdorff=True applies the pruning/ordering heuristic described
    above. Set to False to see the (much slower / often infeasible)
    naive fixed-order backtracking for comparison on small boards.
    Returns the list of visited squares in order, or None if no tour
    was found from the given start (search still explores all branches
    via backtracking; None only occurs after the full space is
    exhausted).
    """
    visited = [[False] * n for _ in range(n)]
    path = [start]
    visited[start[0]][start[1]] = True

    def candidates(x: int, y: int):
        cand = []
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if _on_board(nx, ny, n) and not visited[nx][ny]:
                cand.append((nx, ny))
        if use_warnsdorff:
            # Warnsdorff's rule: try the square with fewest onward moves first
            cand.sort(key=lambda sq: _degree(sq[0], sq[1], n, visited))
        return cand

    def backtrack(x: int, y: int) -> bool:
        if len(path) == n * n:
            return True
        for nx, ny in candidates(x, y):
            visited[nx][ny] = True
            path.append((nx, ny))
            if backtrack(nx, ny):
                return True
            # undo (backtrack step)
            visited[nx][ny] = False
            path.pop()
        return False

    if backtrack(start[0], start[1]):
        return path
    return None


def print_board(path: List[Tuple[int, int]], n: int) -> None:
    board = [[-1] * n for _ in range(n)]
    for step, (x, y) in enumerate(path):
        board[x][y] = step
    width = len(str(n * n))
    for row in board:
        print(" ".join(str(v).rjust(width) for v in row))


def empirical_comparison():
    print("\n--- Empirical timing: naive order vs Warnsdorff pruning ---")
    print(f"{'n':>4} | {'naive (s)':>12} | {'naive found':>12} | "
          f"{'warnsdorff (s)':>15} | {'warnsdorff found':>17}")
    # NOTE: naive (unpruned) backtracking grows so fast that n=7 already
    # takes ~10s in pure Python; n=8 without pruning is infeasible to
    # run here (could take minutes to hours). We therefore only run the
    # naive baseline up to n=7, and show Warnsdorff up to n=12 to
    # demonstrate the practical gap described in the complexity analysis.
    for n in [5, 6, 7]:
        t0 = time.perf_counter()
        tour_naive = knights_tour(n, use_warnsdorff=False)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        tour_w = knights_tour(n, use_warnsdorff=True)
        t3 = time.perf_counter()

        print(f"{n:>4} | {t1 - t0:>12.6f} | {str(tour_naive is not None):>12} | "
              f"{t3 - t2:>15.6f} | {str(tour_w is not None):>17}")

    # Warnsdorff only, larger boards - naive is skipped (infeasible)
    for n in [8, 10, 12]:
        t2 = time.perf_counter()
        tour_w = knights_tour(n, use_warnsdorff=True)
        t3 = time.perf_counter()
        print(f"{n:>4} | {'skipped':>12} | {'-':>12} | "
              f"{t3 - t2:>15.6f} | {str(tour_w is not None):>17}")


if __name__ == "__main__":
    n = 8
    print(f"Finding a knight's tour on an {n}x{n} board using Warnsdorff pruning...")
    t0 = time.perf_counter()
    tour = knights_tour(n, start=(0, 0), use_warnsdorff=True)
    t1 = time.perf_counter()

    if tour:
        print(f"Tour found in {t1 - t0:.4f}s, visiting all {n*n} squares.\n")
        print_board(tour, n)
    else:
        print("No tour found from this starting square.")

    empirical_comparison()
