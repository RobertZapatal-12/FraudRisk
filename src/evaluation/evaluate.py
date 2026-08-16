from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _format_metrics(y_true, y_pred):
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
    y_pred = model.predict(x_test)
    return _format_metrics(y_test, y_pred)


def evaluate_model_tree(model, x_test, y_test):
    return evaluate_model(model, x_test, y_test)


def evaluate_model_forest(model, x_test, y_test):
    return evaluate_model(model, x_test, y_test)





    







