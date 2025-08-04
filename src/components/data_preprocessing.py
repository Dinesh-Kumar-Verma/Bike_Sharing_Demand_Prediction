import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

# Local module imports
from src.components.feature_cleaning import FeatureCleaner
from src.components.feature_engineering import FeatureEngineer
from src.components.feature_scaling import FeatureScaler
from src.utils.data_loader import load_csv
from src.utils.config import X_TRAIN_FILE, DATA_PREPROCESSING_PIPELINE_FILE, RAW_DATA_FILE
from src.utils.data_splitter import DataSplitter
from src.utils.data_saver import save_split_data




logger = get_logger(name='data_preprocessing', log_file='data_preprocessing.log')

def run_data_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs and applies the data preprocessing pipeline.

    Parameters:
        df (pd.DataFrame): Raw input data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    logger.info('-' * 80)
    logger.info("Starting data preprocessing pipeline...")
    logger.info(f"Input columns: {df.columns.tolist()}")

    # Create instances of the transformers
    cleaner = FeatureCleaner()
    engineer = FeatureEngineer(run_vif=False)
    scaler = FeatureScaler()

    # Step 1: Cleaning
    logger.info("Applying FeatureCleaner...")
    df_cleaned = cleaner.fit_transform(df)
    logger.info(f"Columns after cleaning: {df_cleaned.columns.tolist()}")

    # Step 2: Engineering
    logger.info("Applying FeatureEngineer...")
    df_engineered = engineer.fit_transform(df_cleaned)
    logger.info(f"Columns after engineering: {df_engineered.columns.tolist()}")

    # Step 3: Scaling
    logger.info("Applying FeatureScaler...")
    df_scaled = scaler.fit_transform(df_engineered)
    logger.info(f"Columns after scaling: {df_scaled.columns.tolist()}")

    # Original pipeline for saving
    preprocessing_pipeline = Pipeline([
        ("cleaning", cleaner),
        ("engineering", engineer),
        ("scaling", scaler)
    ])
    
    # Save the fitted pipeline
    joblib.dump(preprocessing_pipeline, DATA_PREPROCESSING_PIPELINE_FILE)
    logger.info(f"Data preprocessing pipeline saved to {DATA_PREPROCESSING_PIPELINE_FILE}")

    return df_scaled


if __name__ == "__main__":
    splitter = DataSplitter(mode='random')
    df = load_csv(RAW_DATA_FILE)
    df1 = df.copy()
    df1 = df1.drop(columns=['Rented Bike Count'], axis=1, inplace= True)
    # X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(df, target_column='Rented Bike Count')
    # save_split_data(X_train, y_train, X_val, y_val, X_test, y_test)
    # df = load_csv(X_TRAIN_FILE)
    df_processed = run_data_preprocessing_pipeline(df1)