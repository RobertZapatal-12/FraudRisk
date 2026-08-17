from concurrent.futures import ProcessPoolExecutor

from runall.runall_models import (
    run_decision_tree,
    run_logistic_regresion,
    run_random_forest,
)


def main():
    """Execute all machine learning models in parallel.
    
    Trains and evaluates three models (Logistic Regression, Decision Tree, 
    and Random Forest) concurrently using a process pool executor.
    """
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_logistic_regresion),
            executor.submit(run_decision_tree),
            executor.submit(run_random_forest),
        ]

        for future in futures:
            future.result()


if __name__ == "__main__":
    main()