"""
Check training time using FHE for Wine database
"""

import time
import pandas as pd

from concrete.ml.sklearn import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

column_names = [
    "Class label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]

wine_data = pd.read_csv(URL, header=None, names=column_names)


mapping = {1: 0, 2: 0, 3: 1}
y = y.map(mapping)

parameters_range = (-1.0, 1.0)

model = SGDClassifier(
    random_state=42,
    max_iter=50,
    fit_encrypted=True,
    parameters_range=parameters_range,
)

X = X.to_numpy()
y = y.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

###########################################
start_time = time.time()
model.fit(X_train, y_train, fhe="execute")
end_time = time.time()
elapsed_time = end_time - start_time
print(f'FHE Model built! Training time: {elapsed_time:.6f} seconds')
###########################################


###########################################
start_time = time.time()
fhe_circuit = model.compile(X_train)
y_pred = model.predict(X_test, fhe="execute")
end_time = time.time()
elapsed_time = end_time - start_time
accuracy = accuracy_score(y_test, y_pred)
print(f'FHE Model tested! Prediction time: {elapsed_time:.6f}. Accuracy: {accuracy:.4f} seconds')
###########################################


###########################################
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Clear Model built! Training time: {elapsed_time:.6f} seconds')
###########################################


###########################################
start_time = time.time()
fhe_circuit = model.compile(X_train)
y_pred = model.predict(X_test)
end_time = time.time()
elapsed_time = end_time - start_time
accuracy = accuracy_score(y_test, y_pred)
print(f'Clear Model tested! Prediction time: {elapsed_time:.6f}. Accuracy: {accuracy:.4f} seconds')
###########################################
