import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/creditcard_clean.csv")

def preprocess(df):
    """Preprocess and balance credit card fraud detection data.
    
    Performs class balancing by downsampling to the minority class size, 
    splits data into train/test sets, and scales features using StandardScaler.

    Args:
        df (pd.DataFrame): Input dataframe with features and 'Class' column for labels.
    
    Returns:
        tuple: Contains four arrays:
            - x_train_scaled (np.ndarray): Scaled training features of shape (n_samples, n_features).
            - x_test_scaled (np.ndarray): Scaled test features of shape (n_samples, n_features).
            - y_train (pd.Series): Training labels.
            - y_test (pd.Series): Test labels.
    """
    
    min_count = df["Class"].value_counts().min()

    df = df.groupby("Class").sample(n=min_count, random_state=42)

    x = df.drop(columns=["Class"])
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test






