# 8-Puzzle Search

Implements the 8-puzzle state space and a general search framework with
BFS, DFS, and A* (Chebyshev-based heuristic by default; Manhattan also
provided), plus a custom strategy: **Memory-Guided A\***.

## Memory-Guided A*

A tie-breaking variant of A* that uses a contextual multi-armed bandit
(UCB1) instead of FIFO ordering among nodes with equal `f(n)`.

Each action (`UP`/`DOWN`/`LEFT`/`RIGHT`) is tracked **per context**, where
context = `blank_context(state)` buckets the board by where the blank sits
relative to its goal position. This matters because the same action does
not have a fixed value — moving the blank `UP` helps when it's below goal
and hurts when it's above it. Averaging that into one global counter per
action (the original version of this idea) washes the signal out as noise;
bucketing by context makes each `(context, action)` pair closer to a
genuine, stationary bandit arm.

Note: this mechanism only affects ordering among nodes that already tie on
`f(n)`. It does not make the search optimal if the heuristic itself isn't
admissible — see caveat below.

## Heuristic caveat

The default heuristic (`heuristic_chebyshev`) is **not admissible**: for a
tile at `(dx, dy)` it returns `2 * max(dx, dy)`, which exceeds the true
per-tile lower bound `dx + dy` (Manhattan distance) whenever `dx != dy`.
This means A* (and Memory-Guided A*) using this heuristic is not guaranteed
to return the optimal solution. Switch `heuristic()` to call
`heuristic_manhattan` for a guaranteed-optimal search; the benchmark
currently compares nodes expanded, not solution optimality, so that
tradeoff isn't being measured.

## Run

```bash
python src/search.py
```

Runs A* on the hard initial state, then benchmarks Memory-Guided A*
(Chebyshev) against plain A* (Manhattan) across Easy/Medium/Hard and shows
a bar chart (see `outputs/benchmark_nodes_both_one.png` for a saved run).
