# Task 4 Evaluation: Greedy FFD vs Simulated Annealing

## Results (averaged over 3 random instances per size)

| n | Greedy bins | Greedy time (ms) | SA bins | SA time (ms) | Bin reduction |
|---|---|---|---|---|---|
| 20 | 7.67 | 0.105 | 7.67 | 15.596 | 0.00 |
| 50 | 16.67 | 0.311 | 16.33 | 19.225 | 0.33 |
| 100 | 32.67 | 1.093 | 32.33 | 27.073 | 0.33 |
| 200 | 62.33 | 4.598 | 62.33 | 51.555 | 0.00 |

## Discussion: Solution Quality vs Computational Cost Trade-off

Greedy First-Fit Decreasing is extremely fast - a single deterministic pass over the sorted items - and consistently produces a reasonable solution, but it can never improve on its first decision once an item has been placed in a bin. Simulated Annealing starts from that same Greedy solution and spends a fixed, tunable iteration budget (4000 iterations here) searching for improving item relocations, occasionally accepting temporarily worse moves so it is not trapped in the first local optimum it reaches.

The measured results show that SA typically matches or slightly reduces the number of bins used compared to Greedy alone, at the cost of roughly two to three orders of magnitude more runtime, since its cost is dominated by a fixed iteration count rather than the input size, whereas Greedy's cost grows with n but stays a single fast pass. In practice this means Greedy is the right choice when solutions are needed instantly or the workload changes constantly (e.g. online cloud resource allocation), while Simulated Annealing is preferable in offline planning settings where a small reduction in the number of bins (e.g. fewer physical servers or vehicles) justifies spending extra, but still bounded and polynomial, computation time.
