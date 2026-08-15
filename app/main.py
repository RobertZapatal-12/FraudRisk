from src.preprocessing.preprocess import preprocess
from src.training.training import training, save_model 
from src.evaluation.evaluate import evaluate_model
import pandas as pd

df = pd.read_csv("data/processed/creditcard_clean.csv")

x_train_scaled, x_test_scaled, y_train, y_test = preprocess(df)

model_lr = training(x_train_scaled, y_train)

evalu = evaluate_model(model_lr)

save_model(model_lr)

print(evalu)