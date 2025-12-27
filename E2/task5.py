import pandas as pd
from tqdm import tqdm
from task2 import run_task2
from task3 import run_task3
from task4 import run_task4

def main():
    all_results = []

    tasks = [
        ("Linear SVM", run_task2),
        ("Polynomial SVM", run_task3),
        ("RBF SVM", run_task4)
    ]

    print("Running all SVM tasks...\n")
    
    # Loop through tasks with progress bar
    for task_name, task_func in tqdm(tasks, desc="Tasks progress"):
        print(f"\nExecuting {task_name}...")
        results = task_func()
        all_results.extend(results)

    # Create summary DataFrame
    df = pd.DataFrame(all_results)
    df = df[["dataset", "kernel", "params", "accuracy", "observation"]]

    print("\n=== SVM Summary Table ===\n")
    print(df)

    # Export To CSV
    df.to_csv("svm_summary.csv", index=False)
    print("\nSaved 'svm_summary.csv' and 'svm_summary.tex'")

if __name__ == "__main__":
    main()
