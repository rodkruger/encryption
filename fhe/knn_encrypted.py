from concrete import fhe

from concrete.ml.sklearn.neighbors import KNeighborsClassifier
from concrete.ml.quantization import QuantizedModule

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def add(x, y):
    return x + y

compiler = fhe.Compiler(add, {"x": "encrypted", "y": "encrypted"})

inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]

print(f"Compilation...")
circuit = compiler.compile(inputset)

print(f"Key generation...")
circuit.keygen()

print(f"Homomorphic evaluation...")
encrypted_x, encrypted_y = circuit.encrypt(2, 6)
encrypted_result = circuit.run(encrypted_x, encrypted_y)
result = circuit.decrypt(encrypted_result)

assert result == add(2, 6)

#------------------------------------------------------------------------------

import numpy as np
from collections import Counter

l = []

@fhe.compiler({"x1": "encrypted", "x2": "encrypted"})
def e_manhattan_distance(x1, x2):

    if len(l) < 20:
        l.append( (x1, x2) )

    return np.sum(np.abs(x1 - x2))

class KNN:
    def __init__(self, circuit, k=3):
        self.k = k
        self.circuit = circuit

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _manhattan_distance(self, x1, x2):
        return e_manhattan_distance(x1, x2)
   
    def _predict(self, x):
        # Compute the Euclidean distances between the test point and all training points
        #distances = [self._manhattan_distance(x_train, x) for x_train in self.X_train]
        distances = [self.circuit.run(x_train, x) for x_train in self.X_train]
        
        # Sort by distance and return the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

#------------------------------------------------------------------------------

# Fetching data
data = load_breast_cancer()
X = data.data
y = data.target

X = StandardScaler().fit_transform(X)

#! This is totally wrong! Use quantization instead!
X = X.astype(int)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compile the function using sample data
l = []
for i in range(0, 10):
    l.append( ( x_train[i], x_test[i] ) )

circuit = e_manhattan_distance.compile(l)

#! Generate Keys - we are generating the key in the server side. This is wrong!
circuit.keygen()

# Encrypt data
e_x_train = circuit.encrypt(x_train, x_test)

# Initialize KNN with k=3
knn = KNN(circuit, k=3)

# Train the model
knn.fit(x_train, y_train)

# Predict the class labels for the test set
predictions = knn.predict(x_test)

# Calculate the accuracy score using scikit-learn
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.6f}')

#! Problem: concrete only implements basic arithmetic operations
# https://docs.zama.ai/concrete/core-features/fhe_basics
# Possible conclusion: concrete is inappropriate to be used in data stream