from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from my_package import load_data


def train_iris():

    X_train, X_test, y_train, y_test = load_data.load_data()
    # Train a Random Forest classifier on the training set
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the testing set and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the trained model to disk
    dump(clf, "data/iris_model.joblib")