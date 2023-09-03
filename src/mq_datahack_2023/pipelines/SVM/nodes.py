import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import json


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data.loc[:, data.columns != 'target']  # Taking only the petal length and petal width
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = parameters["test_size"], random_state = parameters["random_state"])
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> SVC:
    """Trains the C-Support Vector Classification model. For more information about the model, visit https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained SVM Classifier model.
    """

    svm_clf = SVC(kernel=parameters['kernel'], C=parameters['C'])
    svm_clf.fit(X_train, y_train)  #
    return svm_clf


def evaluate_model(
        data: pd.DataFrame, model: SVC, X_test: pd.DataFrame, y_test: pd.Series
) -> str:
    """Calculates and logs the accuracy of the SVC classification

    Args:
        data: Raw data.
        model: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    # Testing and checking the accuracy

    y_pred = model.predict(X_test)

    labels = data['target'].unique()

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    accuracy = accuracy_score(y_test, y_pred)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
    confusion = confusion_matrix(y_test, y_pred, labels=labels)

    logger = logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.3f on test data.", accuracy)
    logger.info("Confusion matrix:\n%s", confusion)
    logger.info("F1 score: %.3f", f1)

    return json.dumps({"accuracy": accuracy, "f1": f1, "confusion": confusion.tolist()})
