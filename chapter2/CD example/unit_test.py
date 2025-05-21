import pytest
from typing import Union
import numpy as np
from sklearn.datasets import load_wine
import sklearn.ensemble
from sklearn.model_selection import train_test_split
import sklearn
import joblib

@pytest.fixture
def test_dataset() -> Union[np.array, np.array]:

    X, y = load_wine(return_X_y=True)

    y = y == 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_test, y_test 

@pytest.fixture
def model() -> sklearn.ensemble._forest.RandomForestClassifier:
    REPO_ID = "electricweegie/mlewp-sklearn-wine"
    FILENAME = "rfc.joblib"
    model = joblib.load(hf_hub_download(REPO_ID, FILENAME))
    return model

# test type returns
def test_model_inference_types(model, test_dataset):
    assert isinstance(model.predict(test_dataset[0],), np.ndarray)
    ...

# assert the performance is standard
def test_model_performacne(model, test_dataset):
    metrics = ...
    assert metrics['False']['f1-score'] > .95