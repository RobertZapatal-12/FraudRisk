from sklearn.linear_model import LogisticRegression
import joblib

def training(x_train_scaled, y_train):
    model = LogisticRegression()
    model.fit(x_train_scaled, y_train)
    return model

def save_model(model):
    joblib.dump(model, "models/logistic_creditcard_model.pkl")




    