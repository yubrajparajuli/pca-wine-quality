import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_evaluate(X, y, label=""):
    """
    Splits data, trains Logistic Regression,
    evaluates and returns results dictionary.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)

    start = time.time()
    model.fit(X_train, y_train)
    end   = time.time()

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    duration = end - start

    print(f"\nModel Results {label}:")
    print(f"  Features used:  {X_train.shape[1]}")
    print(f"  Training time:  {duration:.5f} seconds")
    print(f"  Accuracy:       {accuracy*100:.2f}%")
    print(f"\nDetailed Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Bad Wine', 'Good Wine']
    ))

    return {
        "model"    : model,
        "accuracy" : accuracy,
        "duration" : duration,
        "y_test"   : y_test,
        "y_pred"   : y_pred
    }