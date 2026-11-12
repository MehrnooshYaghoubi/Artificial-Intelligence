# SVM Kernel Comparison

Compares linear, polynomial, and RBF SVM kernels across three synthetic
2D datasets (`make_blobs`, `make_circles`, `make_moons`).

- `src/task1_datasets.py` — generates and visualizes the three datasets.
- `src/task2_linear_svm.py` — linear SVM + shared `plot_decision_boundary` helper.
- `src/task3_polynomial_svm.py` — polynomial SVM, degrees 2 and 3.
- `src/task4_rbf_svm.py` — RBF SVM, low vs. high gamma.
- `src/task5_report.py` — runs tasks 2-4 and writes a summary table to
  `outputs/svm_summary.csv`.

## Run

```bash
python src/task5_report.py
```

Plots and the summary CSV are written to `outputs/`.
