import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import X_TRAIN_FILE
from src.utils.data_loader import load_csv

logger = get_logger(name='data_validation', log_file='data_validation.log')

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_missing_values(self):
        logger.info("Checking for missing values...")
        missing = self.df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values detected:\n{missing}")
        else:
            logger.info("No missing values found.")

    def check_duplicates(self):
        logger.info("Checking for duplicate rows...")
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows.")
        else:
            logger.info("No duplicate rows found.")

    def check_data_types(self):
        logger.info("Checking data types of columns...")
        dtypes = self.df.dtypes
        logger.info(f"Data types:\n{dtypes}")

    def check_expected_columns(self, expected_columns):
        logger.info("Validating expected columns...")
        actual_columns = set(self.df.columns)
        missing_cols = set(expected_columns) - actual_columns
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
        else:
            logger.info("All expected columns are present.")

    def run_all_checks(self):
        self.df
        self.check_expected_columns([
            'Date', 'Hour', 'Temperature(°C)',
            'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
            'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)',
            'Rainfall(mm)', 'Snowfall (cm)', 'Seasons',
            'Holiday', 'Functioning Day'
        ])
        self.check_missing_values()
        self.check_duplicates()
        self.check_data_types()


def data_validation():
    logger.info('\n' + '-' * 80)
    logger.info('Starting data_validation...')
    try:
        df = load_csv(X_TRAIN_FILE)
        validator = DataValidator(df)
        validator.run_all_checks()
        logger.info('Completed data_validation.')
    except Exception as e:
        logger.exception("Data validation failed: %s", e)
        raise

if __name__ == '__main__':
    data_validation()