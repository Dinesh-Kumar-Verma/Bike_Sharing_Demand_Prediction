import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from src.utils.logger import get_logger
from src.utils.data_loader import load_csv
from src.utils.config import (
    PROCESSED_FILE,
    INTERIM_DATA_DIR,
    TIMESERIES_FOLDS_DIR,
    X_TRAIN_FILE, Y_TRAIN_FILE,
    X_VAL_FILE, Y_VAL_FILE,
    X_TEST_FILE, Y_TEST_FILE
)

logger = get_logger(name='data_splitter', log_file='data_splitter.log')


def split_data(df: pd.DataFrame, target_column: str, train_size=0.7, val_size=0.15, random_state=42):
    """
    Split the dataset into Train, Validation, and Test sets.
    """
    logger.info("Starting initial 70/15/15 split...")
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # First split: Train + Temp (Val + Test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, shuffle=False, random_state=random_state
        )

        # Second split: Validation + Test
        val_relative_size = val_size / (1 - train_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_relative_size, shuffle=False, random_state=random_state
        )

        # Add assertions for integrity check
        assert len(X_train) == len(y_train), "Mismatch in training set"
        assert len(X_val) == len(y_val), "Mismatch in validation set"
        assert len(X_test) == len(y_test), "Mismatch in test set"
        logger.info("Assertion checks passed: All splits are aligned.")
        logger.info("Completed Train/Val/Test split.")
        return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        logger.exception("Error during split: %s", e)
        raise


def save_split_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Save split datasets to CSV files in interim directory.
    """
    try:
        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(X_TRAIN_FILE, index=False)
        y_train.to_csv(Y_TRAIN_FILE, index=False)
        X_val.to_csv(X_VAL_FILE, index=False)
        y_val.to_csv(Y_VAL_FILE, index=False)
        X_test.to_csv(X_TEST_FILE, index=False)
        y_test.to_csv(Y_TEST_FILE, index=False)

        logger.info("Saved Train/Val/Test splits to interim directory.")

    except Exception as e:
        logger.exception("Failed to save split data: %s", e)
        raise


# def create_timeseries_folds(X_train, y_train, n_splits=5):
#     """
#     Create TimeSeriesSplit folds for hyperparameter tuning and save them.
#     """
#     logger.info("Creating TimeSeriesSplit folds for hyperparameter tuning...")
#     try:
#         TIMESERIES_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
#         tscv = TimeSeriesSplit(n_splits=n_splits)

#         for idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
#             fold_X_train = X_train.iloc[train_idx]
#             fold_y_train = y_train.iloc[train_idx]
#             fold_X_val = X_train.iloc[val_idx]
#             fold_y_val = y_train.iloc[val_idx]

#             # Save each fold to separate CSV files
#             fold_dir = TIMESERIES_FOLDS_DIR / f"fold_{idx + 1}"
#             fold_dir.mkdir(parents=True, exist_ok=True)

#             fold_X_train.to_csv(fold_dir / "X_train.csv", index=False)
#             fold_y_train.to_csv(fold_dir / "y_train.csv", index=False)
#             fold_X_val.to_csv(fold_dir / "X_val.csv", index=False)
#             fold_y_val.to_csv(fold_dir / "y_val.csv", index=False)

#             logger.info("Saved TimeSeriesSplit Fold %d", idx + 1)

#         logger.info("All TimeSeriesSplit folds created successfully.")

#     except Exception as e:
#         logger.exception("Failed to create TimeSeriesSplit folds: %s", e)
#         raise


def main(): 
    logger.info("-" * 80)
    logger.info("Running Data Splitter Pipeline...")

    try:
        df = load_csv(PROCESSED_FILE)
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            df, target_column="rented_bike_count"
        )

        #save_split_data(X_train, y_train, X_val, y_val, X_test, y_test)

        logger.info("Data Splitter Pipeline completed successfully.")
        return X_train, y_train, X_val, y_val, X_test, y_test

    except AssertionError as ae:
        logger.error("Assertion failed: %s", ae)
        raise  # Re-raise to make it visible in CI/CD or pipeline failure

    except Exception as e:
        logger.exception("Data Splitter Pipeline failed: %s", e)
        raise

    



if __name__ == "__main__":
    main()