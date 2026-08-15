import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/creditcard_clean.csv")

def preprocess(df):

    min_count = df["Class"].value_counts().min()

    df = df.groupby("Class").sample(n=min_count, random_state=42)

    x = df.drop(columns=["Class"])
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test






