import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler
from pathlib import Path
from src.utils.data_splitter import DataSplitter
from src.utils.data_saver import save_split_data
from sklearn.pipeline import Pipeline
import joblib
from src.utils.logger import get_logger
from src.utils.config import BASE_DIR, PROCESSED_FEATURES_FILE, SCALER_PIPELINE_FILE

logger = get_logger(name='feature_scaling', log_file='feature_scaling.log')

class FeatureScaler:
    def __init__(self, target_column: str = 'rented_bike_count'):
        self.numerical_features = ['temperature', 'humidity', 'wind_speed', 'visibility',
                                    'solar_radiation', 'rainfall', 'snowfall']
        
        self.target_column = target_column
        self.pipeline = Pipeline([
            ('power', PowerTransformer()),
            ('robust', RobustScaler(unit_variance=True))
        ])
        self.fitted = False

    def check_skewness(self, y: pd.Series):
        try:
            original_skew = y.skew()
            logger.info(f"Original skewness of {self.target_column}: {original_skew:.4f}")
            if (y <= 0).any():
                logger.warning(f"Skipped log/sqrt skewness — {self.target_column} contains zero or negative values.")
            else:
                logger.info(f"Log skew: {np.log(y).skew():.4f}")
                logger.info(f"Log1p skew: {np.log1p(y).skew():.4f}")
                logger.info(f"Sqrt skew: {np.sqrt(y).skew():.4f}")
        except Exception as e:
            logger.exception(f"Error while checking skewness: {e}")

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        try:
            X_scaled = X.copy()
            logger.info("Fitting and transforming numerical features...")
            X_scaled[self.numerical_features] = self.pipeline.fit_transform(X[self.numerical_features])
            self.fitted = True

            if y is not None:
                self.check_skewness(y)
                y_transformed = np.log1p(y)
                return X_scaled, y_transformed
            return X_scaled
        except Exception as e:
            logger.exception("Error during fit_transform.")
            raise

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before calling transform.")
        try:
            X_scaled = X.copy()
            logger.info("Transforming data using existing scaler pipeline...")
            X_scaled[self.numerical_features] = self.pipeline.transform(X[self.numerical_features])
            if y is not None:
                y_transformed = np.log1p(y)
                return X_scaled, y_transformed
            return X_scaled
        except Exception as e:
            logger.exception("Error during transform.")
            raise

    def inverse_transform_target(self, y_transformed: pd.Series):
        try:
            return np.expm1(y_transformed)
        except Exception as e:
            logger.exception("Error during inverse transform of target.")
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
            raise

    def load_pipeline(self, filepath: str = SCALER_PIPELINE_FILE):
        try:
            rel_path = filepath.relative_to(BASE_DIR)
            self.pipeline = joblib.load(filepath)
            self.fitted = True
            logger.info(f"Scaler pipeline loaded from {rel_path}")
        except Exception as e:
            logger.exception(f"Error while loading scaler pipeline.{e}")
            raise


def main():
    df = pd.read_csv(PROCESSED_FEATURES_FILE)
    #splitter = DataSplitter(mode='timeseries')
    splitter = DataSplitter(mode='random')
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(df, target_column='rented_bike_count')

    scaler = FeatureScaler()
    X_train_scaled, y_train_scaled = scaler.fit_transform(X = X_train, y= y_train)
    X_val_scaled, y_val_scaled = scaler.transform(X_val, y_val)
    X_test_scaled, y_test_scaled = scaler.transform(X_test, y_test)
    
    # Save scaled features
    save_split_data(
        X_train = X_train_scaled,
        y_train = y_train_scaled,
        X_val = X_val_scaled,
        y_val = y_val_scaled,
        X_test = X_test_scaled,
        y_test = y_test_scaled
        )

    scaler.save_pipeline()


if __name__ == "__main__":
    main()

# # for prediction
# from src.components.feature_scaling import FeatureScaler
# scaler = FeatureScaler(numerical_features=[...])
# scaler.load_pipeline()

# X_user_input_scaled = scaler.transform(X_user_input)
