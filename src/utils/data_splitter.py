import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from src.utils.logger import get_logger
from src.utils.data_loader import load_csv
from src.utils.data_saver import save_split_data
from src.utils.config import PROCESSED_FEATURES_FILE



logger = get_logger(name='data_splitter', log_file='data_splitter.log')


class DataSplitter:
    """
    Versatile data splitter for both Time Series and Non-Time Series problems.
    """

    def __init__(self, 
                 train_size=0.7, 
                 val_size=0.15, 
                 test_size=0.15,
                 mode='random',  # 'random' or 'timeseries'
                 time_series_splits=5, 
                 random_state=42):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.mode = mode
        self.time_series_splits = time_series_splits
        self.random_state = random_state

        if round(train_size + val_size + test_size, 2) != 1.0:
            raise ValueError("Train + Val + Test sizes must sum to 1.0")

        if self.mode not in ['random', 'timeseries']:
            raise ValueError("Mode must be 'random' or 'timeseries'")

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Split the data according to the selected mode.
        Returns: X_train, y_train, X_val, y_val, X_test, y_test
        """
        logger.info('\n' + '-' * 80)
        logger.info(f"Splitting data in '{self.mode}' mode...")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        try:
            if self.mode == 'random':
                
                return self._random_split(X, y)
            else:
                return self._time_series_split(X, y)
        except Exception as e:
            logger.exception("Data splitting failed.")
            raise

    def _random_split(self, X, y):
        """
        Random splitting with optional shuffling.
        """
        logger.info("Performing random train/val/test split...")

        # Train vs Temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=self.train_size,
            shuffle=True,
            random_state=self.random_state
        )

        # Val vs Test
        val_relative_size = self.val_size / (self.val_size + self.test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_relative_size,
            shuffle=True,
            random_state=self.random_state
        )

        logger.info("Random splitting complete.")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _time_series_split(self, X, y):
        """
        Time series split: no shuffling, order preserved.
        """
        logger.info("Performing time series train/val/test split...")

        n = len(X)
        train_end = int(n * self.train_size)
        val_end = int(n * (self.train_size + self.val_size))

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        logger.info("Time series splitting complete.")
        return X_train, y_train, X_val, y_val, X_test, y_test


# if __name__ == '__main__':
#     df = load_csv(PROCESSED_FEATURES_FILE)
#     #  Random split
#     # splitter = DataSplitter(mode='random')
#     # X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(df, target_column='rented_bike_count')
     
#      # Timeseries Split
#     splitter = DataSplitter(mode='timeseries')
#     X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(df, target_column='rented_bike_count')
#     save_split_data(X_train, y_train, X_val, y_val, X_test, y_test)