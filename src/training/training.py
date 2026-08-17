from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

def training(x_train_scaled, y_train):
    """Train a Logistic Regression model.

    Args:
        x_train_scaled (np.ndarray): Training features scaled array of shape (n_samples, n_features).
        y_train (pd.Series or np.ndarray): Training labels.
    
    Returns:
        LogisticRegression: Fitted logistic regression model.
    """
    model = LogisticRegression()
    model.fit(x_train_scaled, y_train)
    return model

def training_tree(x_train_scaled, y_train):
    """Train a Decision Tree Classifier model.

    Args:
        x_train_scaled (np.ndarray): Training features scaled array of shape (n_samples, n_features).
        y_train (pd.Series or np.ndarray): Training labels.

    Returns:
        DecisionTreeClassifier: Fitted decision tree classifier model with max_leaf_nodes=50.
    """
    model = DecisionTreeClassifier(max_leaf_nodes=50)
    model.fit(x_train_scaled, y_train)
    return model

def training_forest(x_train_scaled, y_train):
    """Train a Random Forest Classifier model.

    Args:
        x_train_scaled (np.ndarray): Training features scaled array of shape (n_samples, n_features).
        y_train (pd.Series or np.ndarray): Training labels.

    Returns:
        RandomForestClassifier: Fitted random forest classifier model.
    """
    model = RandomForestClassifier()
    model.fit(x_train_scaled, y_train)
    return model


def save_model_lr(model):
    """Save the Logistic Regression model to disk.

    Args:
        model (LogisticRegression): Fitted logistic regression model to save.
    """
    joblib.dump(model, "models/logistic_creditcard_model.pkl")

def save_model_tree(model):
    """Save the Decision Tree Classifier model to disk.

    Args:
        model (DecisionTreeClassifier): Fitted decision tree classifier model to save.
    """
    joblib.dump(model, "models/tree_creditcard_model.pkl")

def save_model_forest(model):
    """Save the Random Forest Classifier model to disk.

    Args:
        model (RandomForestClassifier): Fitted random forest classifier model to save.
    """
    joblib.dump(model, "models/forest_creditcard_model.pkl")





    