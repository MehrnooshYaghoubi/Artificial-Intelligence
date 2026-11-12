# Traveling Salesman Problem (TSP)

- `src/tsp_solver_base.py` — assignment template with placeholders for 7
  algorithms: backtracking, forward checking (CSP), MRV/MVC heuristics,
  hill climbing, simulated annealing, and a genetic algorithm.
- `src/modified_sa.py` — a completed simulated-annealing solver, plus 10
  alternative cooling-schedule functions (geometric, linear, logarithmic,
  exponential, adaptive, reheating, etc.) for comparison.
- `data/sample.json`, `data/sample_input.json` — city/distance datasets
  used by both scripts.

## Run

```bash
python src/modified_sa.py
python src/tsp_solver_base.py
```
