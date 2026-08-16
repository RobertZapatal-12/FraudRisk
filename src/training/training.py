from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

def training(x_train_scaled, y_train):
    model = LogisticRegression()
    model.fit(x_train_scaled, y_train)
    return model

def training_tree(x_train_scaled, y_train):
    model = DecisionTreeClassifier(max_leaf_nodes= 50)
    model.fit(x_train_scaled, y_train)
    return model


def save_model_lr(model):
    joblib.dump(model, "models/logistic_creditcard_model.pkl")

def save_model_tree(model):
    joblib.dump(model, "models/tree_creditcard_model.pkl")




    