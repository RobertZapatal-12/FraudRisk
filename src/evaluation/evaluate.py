import pandas as pd
from src.preprocessing.preprocess import preprocess
from src.training.training import training
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


df = pd.read_csv("data/processed/creditcard_clean.csv")

def evaluate_model(model):
    x_train_scaled, x_test_scaled, y_train, y_test = preprocess(df)
    model = training(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall =  recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return f"Acurracy = {accuracy}\n f1_score = {f1}\n Recall Score = {recall}\n Precision = {precision} "


    







