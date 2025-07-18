import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from sklearn.preprocessing import PowerTransformer, RobustScaler
from src.utils.config import PROCESSED_FEATURES_FILE, SCALED_DATA_FILE, X_FEATURES_FILE, Y_TARGET_FILE

logger = get_logger(name='feature_scaling', log_file='feature_scaling.log')

def check_skewness(df, target_col):
    """
    Logs skewness of different transformations on target variable.
    """
    try:
        original_skew = df[target_col].skew()
        logger.info(f"Original skewness of {target_col}: {original_skew:.4f}")

        log_skew = np.log(df[target_col]).skew()
        logger.info(f"Log Transformation skewness: {log_skew:.4f}")

        log1p_skew = np.log1p(df[target_col]).skew()
        logger.info(f"Log1p Transformation skewness: {log1p_skew:.4f}")

        sqrt_skew = np.sqrt(df[target_col]).skew()
        logger.info(f"Sqrt Transformation skewness: {sqrt_skew:.4f}")

    except Exception as e:
        logger.exception(f"Error while checking skewness: {e}")

def feature_scaling():
    logger.info('\n' + '-' * 80)
    logger.info('🚀 Starting Data Transformation and Feature Scaling...')

    try:
        # Load feature engineered data
        df = pd.read_csv(PROCESSED_FEATURES_FILE)
        logger.info(f"Loaded feature-engineered data from {PROCESSED_FEATURES_FILE}")

        # Numerical Features
        numerical_features = [
            'temperature', 'humidity', 'wind_speed', 'visibility',
            'solar_radiation', 'rainfall', 'snowfall'
        ]
        target_col = 'rented_bike_count'

        # Scale numerical features using PowerTransformer
        pt = PowerTransformer()
        df[numerical_features] = pt.fit_transform(df[numerical_features])
        logger.info("Applied PowerTransformer to numerical features.")

        # Log skewness of target variable
        check_skewness(df, target_col)

        # Apply log1p transformation to target variable
        df[target_col] = np.log1p(df[target_col])
        logger.info(f"Applied log1p transformation to {target_col}.")

        # Split into features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Scale features using RobustScaler
        rs = RobustScaler(unit_variance=True)
        X_scaled = rs.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        logger.info("Applied RobustScaler to features.")

        # Save scaled data
        df.to_csv(SCALED_DATA_FILE, index=False)
        X_scaled_df.to_csv(X_FEATURES_FILE, index=False)
        y.to_csv(Y_TARGET_FILE, index=False)

        logger.info(f"Saved scaled data to {SCALED_DATA_FILE}")
        logger.info(f"Saved X features to {X_FEATURES_FILE}")
        logger.info(f"Saved y target to {Y_TARGET_FILE}")

    except Exception as e:
        logger.exception("Failed during data scaling: %s", e)

    logger.info('Completed Feature Scaling.')

if __name__ == '__main__':
    feature_scaling()
