from pathlib import Path

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(BASE_DIR)

# Data directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw" 
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
INTERIM_DATA_DIR = BASE_DIR / "data" / "interim"
TIMESERIES_FOLDS_DIR = INTERIM_DATA_DIR / 'timeseries_folds'


# File paths
RAW_DATA_FILE = RAW_DATA_DIR / "bike_sharing_data.csv"
CLEANED_FEATURE_FILE = RAW_DATA_DIR / "bike_sharing_data_cleaned.csv"
PROCESSED_FILE = PROCESSED_DATA_DIR / "bike_sharing_data_processed.csv"
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "bike_sharing_data_features.csv" # File after feature engineering
SCALED_DATA_FILE = PROCESSED_DATA_DIR / "bike_sharing_data_scaled.csv"
X_FEATURES_FILE = PROCESSED_DATA_DIR / 'X_features.csv'
Y_TARGET_FILE = PROCESSED_DATA_DIR / 'Y_target.csv'

X_TRAIN_FILE = INTERIM_DATA_DIR / "X_train.csv"
Y_TRAIN_FILE = INTERIM_DATA_DIR / "y_train.csv"
X_VAL_FILE = INTERIM_DATA_DIR / "X_val.csv"
Y_VAL_FILE = INTERIM_DATA_DIR / "y_val.csv"
X_TEST_FILE = INTERIM_DATA_DIR / "X_test.csv"
Y_TEST_FILE = INTERIM_DATA_DIR / "y_test.csv"

# Create directories if they don’t exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
