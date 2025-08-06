import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler
from pathlib import Path
from src.utils.data_saver import save_split_data, save_csv
from sklearn.pipeline import Pipeline
import joblib
from src.utils.logger import get_logger
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.config import BASE_DIR, PROCESSED_FEATURES_FILE, SCALER_PIPELINE_FILE, X_TRAIN_FILE, SCALED_DATA_FILE
from src.utils.data_splitter import DataSplitter

logger = get_logger(name='feature_scaling', log_file='feature_scaling.log')


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features=None):
        if numerical_features is None:
            numerical_features = [
                'temperature', 'humidity', 'wind_speed', 'visibility',
                'solar_radiation', 'rainfall', 'snowfall'
            ]
        self.numerical_features = numerical_features
        self.pipeline = Pipeline([
            ('power', PowerTransformer()),
            ('robust', RobustScaler(unit_variance=True))
        ])
        self.fitted = False

    def _validate_columns(self, X: pd.DataFrame):
        """Check if all required numerical features are present in the input data."""
        missing = set(self.numerical_features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns in input data: {missing}")


    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self._validate_columns(X)
        logger.info(f"Fitting scaler pipeline data with shape: {X.shape}")
        self.pipeline.fit(X[self.numerical_features])
        self.fitted = True
        logger.info("Scaler pipeline fitted successfully.")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before calling transform.")
        try:
            logger.info("Running transform...")
            logger.info(f"Transforming data with shape: {X.shape}")
            self._validate_columns(X)
            X_scaled = X.copy()
            X_scaled[self.numerical_features] = self.pipeline.transform(X[self.numerical_features])
            logger.info(f"Transformed data successfully. Output shape: {X_scaled.shape}")
            return X_scaled
        except Exception as e:
            logger.exception("Error during transform.")
        raise      
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        try:
            logger.info("Running fit_transform...")
            self._validate_columns(X)
            X_scaled = X.copy()
            X_scaled[self.numerical_features] = self.pipeline.fit_transform(X[self.numerical_features])
            self.fitted = True
            logger.info("Scaler pipeline fit_transform completed successfully.")
            self.save_pipeline()
            return X_scaled
        except Exception as e:
            logger.exception("Error during fit_transform.")
            raise

    def save_pipeline(self, filepath: str = SCALER_PIPELINE_FILE):
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            rel_path = filepath.relative_to(BASE_DIR)
            joblib.dump(self.pipeline, filepath)
            logger.info(f"Scaler pipeline saved to {rel_path}")
        except Exception as e:
            logger.exception(f"Error while saving scaler pipeline. {e}")


def main():
    df = pd.read_csv(PROCESSED_FEATURES_FILE) 
    scaler = FeatureScaler()
    df_scaled = scaler.fit_transform(df)
    save_csv(df_scaled, SCALED_DATA_FILE)


 
if __name__ == "__main__":
    main()



