import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.data_loader import load_csv
from src.utils.data_splitter import split_data, save_split_data
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.pipeline import Pipeline
import joblib
from src.utils.config import (
    PROCESSED_FEATURES_FILE,
    SCALED_DATA_FILE,
)

logger = get_logger(name='feature_scaling', log_file='feature_scaling.log')

def check_skewness(df, target_col):
    try:
        original_skew = df[target_col].skew()
        logger.info(f"Original skewness of {target_col}: {original_skew:.4f}")

        if (df[target_col] <= 0).any():
            logger.warning(f"Skipped log/sqrt skewness — {target_col} contains zero or negative values.")
        else:
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
    logger.info('Starting Data Transformation and Feature Scaling...')

    try:
        # Load feature-engineered data
        df = load_csv(PROCESSED_FEATURES_FILE)
        logger.info(f"Loaded feature-engineered data from {PROCESSED_FEATURES_FILE}")

        # Define numerical features
        numerical_features = ['temperature', 'humidity', 'wind_speed', 'visibility',
                              'solar_radiation', 'rainfall', 'snowfall']
        target_column = 'rented_bike_count'
            
        # Split into X/y
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, target_column=target_column)

        # Fit transform on training set
        pipeline = Pipeline([
            ('power', PowerTransformer(),),
            ('robust', RobustScaler(unit_variance=True))
        ])
        X_train[numerical_features] = pipeline.fit_transform(X_train[numerical_features])
        X_val[numerical_features] = pipeline.transform(X_val[numerical_features])
        X_test[numerical_features] = pipeline.transform(X_test[numerical_features])
        logger.info("Applied PowerTransformer + RobustScaler pipeline on numerical features.")
        
        # After fitting the pipeline
        joblib.dump(pipeline, SCALED_DATA_FILE)  # Save pipeline
        logger.info("Saved transformation pipeline to %s", SCALED_DATA_FILE)

        # Log and transform the target
        check_skewness(y_train.to_frame(), target_column)
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
        y_test = np.log1p(y_test)
        logger.info("Applied log1p transformation on target: %s", target_column)

        # Save transformed splits if needed
        save_split_data(X_train, y_train, X_val, y_val, X_test, y_test)


        logger.info("Saved all transformed splits.")

    except Exception as e:
        logger.exception("Failed during feature scaling: %s", e)

    logger.info('Completed Feature Scaling.')
    
if __name__ == '__main__':
    feature_scaling()
