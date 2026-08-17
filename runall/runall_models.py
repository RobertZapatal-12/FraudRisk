import pandas as pd

from src.evaluation.evaluate import evaluate_model
from src.preprocessing.preprocess import preprocess
from src.training.training import (
    save_model_forest,
    save_model_lr,
    save_model_tree,
    training,
    training_forest,
    training_tree,
)


def run_logistic_regresion():
    """ """
    df = pd.read_csv("data/processed/creditcard_clean.csv")
    x_train_scaled, x_test_scaled, y_train, y_test = preprocess(df)

    model_lr = training(x_train_scaled, y_train)
    evalu = evaluate_model(model_lr, x_test_scaled, y_test)
    save_model_lr(model_lr)

    print("=" * 25)
    print("Logistics Regression")
    print("=" * 25)
    print(evalu)
    return evalu


def run_decision_tree():
    """ """
    df = pd.read_csv("data/processed/creditcard_clean.csv")
    x_train_scaled, x_test_scaled, y_train, y_test = preprocess(df)

    model_tree = training_tree(x_train_scaled, y_train)
    evalu = evaluate_model(model_tree, x_test_scaled, y_test)
    save_model_tree(model_tree)

    print("=" * 25)
    print("Decision Tree")
    print("=" * 25)
    print(evalu)
    return evalu


def run_random_forest():
    """ """
    df = pd.read_csv("data/processed/creditcard_clean.csv")
    x_train_scaled, x_test_scaled, y_train, y_test = preprocess(df)

    model_forest = training_forest(x_train_scaled, y_train)
    evalu = evaluate_model(model_forest, x_test_scaled, y_test)
    save_model_forest(model_forest)

    print("=" * 25)
    print("Random Forest")
    print("=" * 25)
    print(evalu)
    return evalu
