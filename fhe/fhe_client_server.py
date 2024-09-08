from concrete.ml.sklearn import DecisionTreeClassifier
from concrete.ml.sklearn import SGDClassifier
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import os
import shutil

# Define the directory for FHE client/server files
fhe_directory = '/tmp/fhe_client_server_files/'

# Check if the directory exists
if os.path.exists(fhe_directory):
    # Remove the directory and its contents
    shutil.rmtree(fhe_directory)

# Initialize the Decision Tree model
parameters_range = (-1.0, 1.0)

model = SGDClassifier(
    random_state=42,
    max_iter=50,
    fit_encrypted=False,
    parameters_range=parameters_range)

# Generate some random data for training
data = load_wine()
X = data.data
y = data.target

X = StandardScaler().fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and compile the model
model.fit(x_train, y_train)
model.compile(x_train)

# Setup the development environment
dev = FHEModelDev(path_dir=fhe_directory, model=model)
dev.save()

# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

# Client pre-processes new data
X_new = np.array([ x_test[0] ])
encrypted_data = client.quantize_encrypt_serialize(X_new)

# Setup the server
server = FHEModelServer(path_dir=fhe_directory)
server.load()

# Server processes the encrypted data
encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)

# Client decrypts the result
result = client.deserialize_decrypt_dequantize(encrypted_result)
print(result)

# SGD
# KNN
# Implementar sobre dados criptografados
