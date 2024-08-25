"""
Evaluate different datasets on ML models trained with encrypted data
"""

import numpy as np
import pandas as pd
import time

from concrete.ml.sklearn import SGDClassifier

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------------------------

def train_model(x_train, y_train, p_fit_encrypted=True, p_fhe_mode="execute"):
    """
    Build and train a SGDClassifier applying FHE
    """

    start_time = time.time()
    parameters_range = (-1.0, 1.0)

    model = SGDClassifier(
        random_state=42,
        max_iter=50,
        fit_encrypted=p_fit_encrypted,
        parameters_range=parameters_range,
    )

    model.fit(x_train, y_train, fhe=p_fhe_mode)
    model.compile(x_train)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'Training time: {elapsed_time:.6f} seconds')

    return model

#------------------------------------------------------------------------------

def test_model(x_test, y_test, p_model, p_fhe_mode="execute"):
    """
    Tests the model with the test dataset
    """
    start_time = time.time()
    y_pred = p_model.predict(x_test, fhe=p_fhe_mode)
    accuracy = accuracy_score(y_test, y_pred)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'Testing time: {elapsed_time:.6f} seconds. Accuracy: {accuracy:.4f}')

#------------------------------------------------------------------------------

def get_wine_data():
    """
    Read wine dataset from UCI. Transform data into a binary problem and 
    scaled by the StandardScaler.
    """

    wine_data = load_wine()
    x = wine_data.data
    y = wine_data.target

    # Roughly transform the database into a binary classification, because
    # concrete-ml can only work on binary problems. This is not correct, is
    # just to test concrete-ml
    mapping = {0: 0, 1: 0, 2: 1}
    y = np.array([mapping[x] for x in y])

    # Scale data
    x = StandardScaler().fit_transform(x)

    return x, y

#------------------------------------------------------------------------------

def get_cancer_data():
    """
    Read wine dataset from UCI. Transform data into a binary problem and 
    scaled by the StandardScaler.
    """

    cancer_data = load_breast_cancer()
    x = cancer_data.data
    y = cancer_data.target

    # Scale data
    x = StandardScaler().fit_transform(x)

    return x, y

#------------------------------------------------------------------------------

print(">>> Wine dataset ...")
x, y = get_wine_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print("# Building models on clear data")
model = train_model(x_train, y_train, p_fit_encrypted=False, p_fhe_mode=None)
test_model(x_test, y_test, model, p_fhe_mode="disable")

print("# Building models on encrypted data")
model = train_model(x_train, y_train)
test_model(x_test, y_test, model)

print(">>> Cancer dataset ...")
x, y = get_cancer_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print("# Building models on clear data")
model = train_model(x_train, y_train, p_fit_encrypted=False, p_fhe_mode=None)
test_model(x_test, y_test, model, p_fhe_mode="disable")

print("# Building models on encrypted data")
model = train_model(x_train, y_train)
test_model(x_test, y_test, model)

# TODO
# 1. Two examples of regression
# 2. Think to adapt concrete to work as data stream
# 2. Implement something with CKKS