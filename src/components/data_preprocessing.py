import warnings
import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from src.components.feature_cleaning import FeatureCleaner
from src.components.feature_engineering import FeatureEngineer
from src.components.feature_scaling import FeatureScaler
from src.utils.logger import get_logger
from src.utils.data_loader import load_csv
from src.utils.data_splitter import DataSplitter
from src.utils.data_saver import save_split_data, save_split_processed_transformed_data, save_csv
from src.utils.config import (
    DATA_PREPROCESSING_PIPELINE_FILE,
    X_TRAIN_FILE, Y_TRAIN_FILE,
    X_VAL_FILE, Y_VAL_FILE,
    X_TEST_FILE, Y_TEST_FILE,
    )
from src.utils.params import load_params
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = get_logger(name="data_preprocessor", log_file="data_preprocessor.log")


class DataPreprocessor:
    """
    The DataPreprocessor class orchestrates the data preprocessing workflow, encapsulating
    a modular pipeline composed of:

    - Feature Cleaning via `FeatureCleaner`
    - Feature Engineering via `FeatureEngineer` (with optional VIF analysis)
    - Feature Scaling via `FeatureScaler`

    Upon fitting, the pipeline is serialized and persisted to disk to enable reproducible
    transformations on future datasets.

    Attributes:
        run_vif (bool): Flag to enable or disable Variance Inflation Factor check.
        pipeline (Pipeline): Scikit-learn pipeline comprising cleaner, engineer, and scaler.

    Methods:
        fit_transform(X): Fits and transforms the dataset, saving the pipeline to disk.
        transform(X): Transforms dataset using the saved pipeline.
        load_pipeline(): Loads a previously saved pipeline from disk.
    """
    
    def __init__(self, run_vif=False):
        try:            
            self.pipeline = Pipeline([
                ("cleaner", FeatureCleaner()),
                ("engineer", FeatureEngineer(run_vif=run_vif)),
                ("scaler", FeatureScaler())
            ])
            logger.info("Initialized DataPreprocessor pipeline successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize preprocessing pipeline: {e}")
            raise
        
    def fit(self, X: pd.DataFrame):
        try:
            logger.info("Fitting DataPreprocessor pipeline...")
            self.pipeline.fit(X)
            joblib.dump(self.pipeline, Path(DATA_PREPROCESSING_PIPELINE_FILE))
            logger.info(f"Pipeline saved to: {DATA_PREPROCESSING_PIPELINE_FILE}")
            return self
        except Exception as e:
            logger.exception(f"Error during fit: {e}")
            raise

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting fit_transform on DataPreprocessor...")
            X_processed = self.pipeline.fit_transform(X)
            
            # Correctly save the trained pipeline
            joblib.dump(self.pipeline, Path(DATA_PREPROCESSING_PIPELINE_FILE))
            logger.info(f"Pipeline saved to: {DATA_PREPROCESSING_PIPELINE_FILE}")
            
            return X_processed
        except Exception as e:
            logger.exception(f"Error during fit_transform: {e}")
            raise


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting transform on DataPreprocessor...")
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                raise ValueError("Pipeline is not initialized or not loaded.")
            return self.pipeline.transform(X)
        except Exception as e:
            logger.exception(f"Error during transform: {e}")
            raise

    def load_pipeline(self):
        try:
            logger.info(f"Loading pipeline from: {DATA_PREPROCESSING_PIPELINE_FILE}")
            loaded_obj = joblib.load(Path(DATA_PREPROCESSING_PIPELINE_FILE))

            if not isinstance(loaded_obj, Pipeline):
                raise TypeError("Loaded object is not a scikit-learn Pipeline.")

            self.pipeline = loaded_obj
            logger.info("Pipeline loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Pipeline file not found at: {DATA_PREPROCESSING_PIPELINE_FILE}")
            raise
        except Exception as e:
            logger.exception(f"Error loading pipeline: {e}")
            raise



def main():
    try:
        params = load_params()
        feature_engineering_params = params['feature_engineering']
        run_vif = feature_engineering_params['run_vif']
        
        # Preprocess the data
        X_train = load_csv(X_TRAIN_FILE)
        y_train = load_csv(Y_TRAIN_FILE)
        X_val = load_csv(X_VAL_FILE)
        y_val = load_csv(Y_VAL_FILE)
        X_test = load_csv(X_TEST_FILE)
        y_test = load_csv(Y_TEST_FILE)
        

        processor = DataPreprocessor(run_vif=run_vif)
        X_train_processed = processor.fit_transform(X_train)
        X_val_processed = processor.transform(X_val)
        X_test_processed = processor.transform(X_test)
        
        # Apply log transformation on target variable
        y_train_transformed = np.log1p(y_train) 
        y_val_transformed = np.log1p(y_val) 
        y_test_transformed = np.log1p(y_test)
       
        # Save the final Processed and Transformed data
        save_split_processed_transformed_data(
        X_train_processed, y_train_transformed,
        X_val_processed, y_val_transformed,
        X_test_processed, y_test_transformed
        )
            
        logger.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logger.exception("Data preprocessing failed: %s", e)
        raise



if __name__ == "__main__":
    main()
