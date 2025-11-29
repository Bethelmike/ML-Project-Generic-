import sys
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging
import pickle

# -----------------------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# -----------------------------

# -----------------------------
# Custom Exception
class CustomException(Exception):
    def __init__(self, error, sys_info):
        super().__init__(f"{error} | sys info: {sys_info}")
# -----------------------------

# -----------------------------
# Utility function to save object
def save_object(file_path: str, obj: object) -> None:
    """Save a Python object to a file using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Preprocessor saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
# -----------------------------

# -----------------------------
# Data Transformation Classes
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = r"C:\Users\HIGH HORIZON\Desktop\ML Project 2\artifacts\preprocessor.pkl"

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """Build ColumnTransformer for preprocessing."""
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Read train/test CSVs, apply preprocessing, return arrays and preprocessor path."""
        try:
            # Check CSV existence
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Train CSV not found: {train_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test CSV not found: {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data successfully")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # convert sparse to dense if necessary
            if issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if issparse(input_feature_test_arr):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # combine features and target
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)
            ]

            # save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info(f"Data transformation completed. Train shape: {train_arr.shape}, Test shape: {test_arr.shape}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

# -----------------------------
# Main Execution
if __name__ == "__main__":
    train_csv = r"C:\Users\HIGH HORIZON\Desktop\ML Project 2\artifacts\train.csv"
    test_csv = r"C:\Users\HIGH HORIZON\Desktop\ML Project 2\artifacts\test.csv"

    dt = DataTransformation()
    train_arr, test_arr, preprocessor_path = dt.initiate_data_transformation(train_csv, test_csv)

    logging.info(f"Preprocessor file created at: {preprocessor_path}")
    print("Done! Preprocessor pickle is saved.")
