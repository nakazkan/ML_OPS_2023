from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

clf = load('data/iris_model.joblib')
df = pd.read_csv('data/test_data.csv')
X_test = np.array(df)
y_test = X_test[:,0]
X_test = X_test[:,1:]

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

out = pd.DataFrame(y_pred)
out.to_csv('data/out.csv')