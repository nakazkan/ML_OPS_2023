from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



def load_data():

    # Load the Iris dataset
    iris = load_iris()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

