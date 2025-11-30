import os
import sys
import dill
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import CustomException


def save_object(file_path, obj):
    """Save a Python object using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """Train & evaluate multiple models."""
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            report[name] = r2_score(y_test, y_test_pred)

        return report

    except Exception as e:
        raise CustomException(e, sys)
