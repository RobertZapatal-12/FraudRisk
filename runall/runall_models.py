from src.preprocessing.preprocess import preprocess
from src.training.training import training, training_tree, save_model_lr, save_model_tree
from src.evaluation.evaluate import evaluate_model
import pandas as pd

def run_logistic_regresion():

    df = pd.read_csv("data/processed/creditcard_clean.csv")

    x_train_scaled, x_test_scaled, y_train, y_test = preprocess(df)

    model_lr = training(x_train_scaled, y_train)

    evalu = evaluate_model(model_lr)

    save_model_lr(model_lr)

    print("=" *25)
    print("Logistics Regression")
    print("=" *25)
    print(evalu)

def run_decision_tree():
    df = pd.read_csv("data/processed/creditcard_clean.csv")

    x_train_scaled, x_test_scaled, y_train, y_test = preprocess(df)

    model_tree = training_tree(x_train_scaled, y_train)

    evalu = evaluate_model(model_tree)

    save_model_tree(model_tree)

    print("=" *25)
    print("Decision Tree")
    print("=" *25)
    print(evalu)

    