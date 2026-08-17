from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _format_metrics(y_true, y_pred):
    """Format evaluation metrics into a readable string.
    
    Args:
        y_true (pd.Series or np.ndarray): True labels.
        y_pred (pd.Series or np.ndarray): Predicted labels.
    
    Returns:
        str: Formatted string with accuracy, F1-score, recall, and precision metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    return (
        f"Acurracy = {accuracy}\n"
        f"f1_score = {f1}\n"
        f"Recall Score = {recall}\n"
        f"Precision = {precision}"
    )


def evaluate_model(model, x_test, y_test):
    """Evaluate a trained machine learning model on test data.
    
    Args:
        model (BaseEstimator): A scikit-learn fitted model object with predict method.
                              Can be LogisticRegression, DecisionTreeClassifier, or RandomForestClassifier.
        x_test (pd.DataFrame or np.ndarray): Test features.
        y_test (pd.Series or np.ndarray): True test labels.
    
    Returns:
        str: Formatted string with evaluation metrics (Accuracy, F1-Score, Recall, Precision).
    """
    y_pred = model.predict(x_test)
    return _format_metrics(y_test, y_pred)


def evaluate_model_tree(model, x_test, y_test):
    """Evaluate a Decision Tree model on test data.
    
    This is a wrapper function that delegates to evaluate_model().
    
    Args:
        model (DecisionTreeClassifier): A fitted decision tree classifier model.
        x_test (pd.DataFrame or np.ndarray): Test features.
        y_test (pd.Series or np.ndarray): True test labels.
    
    Returns:
        str: Formatted string with evaluation metrics (Accuracy, F1-Score, Recall, Precision).
    """
    return evaluate_model(model, x_test, y_test)


def evaluate_model_forest(model, x_test, y_test):
    """Evaluate a Random Forest model on test data.
    
    This is a wrapper function that delegates to evaluate_model().
    
    Args:
        model (RandomForestClassifier): A fitted random forest classifier model.
        x_test (pd.DataFrame or np.ndarray): Test features.
        y_test (pd.Series or np.ndarray): True test labels.
    
    Returns:
        str: Formatted string with evaluation metrics (Accuracy, F1-Score, Recall, Precision).
    """
    return evaluate_model(model, x_test, y_test)





    







