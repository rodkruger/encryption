"""
Test Dataframe encryption using concrete machine learning
"""
from concrete.ml.pandas import ClientEngine
from concrete.ml.sklearn import SGDClassifier
from concrete.ml.common.utils import FheMode

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
import time

# =============> THIS CODE SNIPPET SHOULD RUN ON THE CLIENT SIDE <=============

# First step: get wine data and normalize it.
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

column_names = [
    "Class label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
    "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols",
    "Proanthocyanins", "Color intensity", "Hue",
    "OD280/OD315 of diluted wines", "Proline"
]

data = pd.read_csv(URL, header=None, names=column_names)

X = data.drop("Class label", axis=1)
y = data["Class label"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Second step: encrypt the data using TFHE
client = ClientEngine(keys_path="my_keys")
X_encrypted = client.encrypt_from_pandas(pd.DataFrame(X))
y_encrypted = client.encrypt_from_pandas(pd.DataFrame(y))

# =============> THIS CODE SNIPPET SHOULD RUN ON THE SERVER SIDE <=============

parameters_range = (-1.0, 1.0)

model = SGDClassifier(
    random_state=42,
    max_iter=50,
    fit_encrypted=True,
    parameters_range=parameters_range,
)

X = X_encrypted.encrypted_values
y = y_encrypted.encrypted_values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

start_time = time.time()
model.fit(X_train, y_train, fhe=FheMode.SIMULATE)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'FHE Model built! Training time: {elapsed_time:.6f} seconds')

start_time = time.time()
fhe_circuit = model.compile(X_train)
y_pred = model.predict(X_test)
end_time = time.time()
elapsed_time = end_time - start_time
accuracy = accuracy_score(y_test, y_pred)
print(f'FHE Model tested! Prediction time: {elapsed_time:.6f}. Accuracy: {accuracy:.4f} seconds')
