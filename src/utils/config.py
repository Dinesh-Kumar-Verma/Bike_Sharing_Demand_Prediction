from pathlib import Path

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(BASE_DIR)

# Data directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw" 
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# File paths
RAW_DATA_FILE = RAW_DATA_DIR / "bike_sharing_data.csv"
PROCESSED_FILE = PROCESSED_DATA_DIR / "bike_sharing_data_processed.csv"
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "bike_sharing_data_features.csv"
SCALED_DATA_FILE = PROCESSED_DATA_DIR / "bike_sharing_data_scaled.csv"
X_FEATURES_FILE = PROCESSED_DATA_DIR / 'X_features.csv'
Y_TARGET_FILE = PROCESSED_DATA_DIR / 'Y_target.csv'

# Create directories if they don’t exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
