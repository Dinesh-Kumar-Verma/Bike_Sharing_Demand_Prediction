from src.utils.logger import get_logger
import pandas as pd
from src.utils.config import PROCESSED_FILE, PROCESSED_FEATURES_FILE, PROCESSED_DATA_DIR
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = get_logger(name='feature_engineering', log_file='feature_engineering.log')

def calc_vif(X, output_file):
    """
    Calculate Variance Inflation Factor (VIF) for features and save as CSV.
    """
    logger.info('Calculating VIF for numerical features...')
    try:
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Save VIF report
        vif_report_path = output_file.parent / 'vif_report.csv'
        vif.to_csv(vif_report_path, index=False)
        logger.info(f"VIF report saved at {vif_report_path.resolve()}")
        logger.info(f"\n{vif}")

        return vif

    except Exception as e:
        logger.exception(f"Error while calculating VIF: {e}")
        raise
    

def feature_engineering():
    logger.info('\n' + '-' * 80)
    logger.info('Starting feature_engineering...')

    try:
        numerical_features = [
            'temperature', 'humidity', 'wind_speed', 'visibility',
            'dew_point_temp', 'solar_radiation', 'rainfall', 'snowfall'
        ]

        # Load processed data
        df = pd.read_csv(PROCESSED_FILE, sep=',', encoding='latin')
        logger.info(f'Loaded data with shape: {df.shape}')

        # Encode categorical features
        df['holiday'] = df['holiday'].map({"Holiday": 1, "No Holiday": 0}).fillna(0).astype(int)
        df['func_day'] = df['func_day'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        logger.info('Encoded holiday and func_day columns.')

        # One-hot encode hour, month, weekday
        df = pd.concat([
            df,
            pd.get_dummies(df['hour'], prefix='hour', drop_first=True),
            pd.get_dummies(df['month'], prefix='month', drop_first=True),
            pd.get_dummies(df['weekday'], prefix='weekday', drop_first=True)
        ], axis=1)
        df.drop(['hour', 'month', 'weekday', 'year'], axis=1, inplace=True)
        logger.info(f'One-hot encoded features and dropped original columns. Data shape: {df.shape}')

        # Clean column names
        df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '')
                      .replace(',', '').replace("'", "") for col in df.columns]

        # VIF Analysis
        logger.info("calculating vif for numerical features")
        vif_df = calc_vif(df[numerical_features], PROCESSED_FEATURES_FILE)

        # Drop highly correlated feature based on VIF
        if 'dew_point_temp' in df.columns:
            df.drop(['dew_point_temp'], axis=1, inplace=True)
            logger.info("Dropped 'dew_point_temp' due to high VIF.")

        # Drop other less important features
        df.drop(['seasons', 'day'], axis=1, inplace=True)
        logger.info("Dropped 'seasons' and 'day' based on domain knowledge.")

        # Ensure output directory exists
        PROCESSED_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Save processed features
        df.to_csv(PROCESSED_FEATURES_FILE, index=False)
        logger.info(f"Feature-engineered data saved to {PROCESSED_FEATURES_FILE.resolve()}")

    except Exception as e:
        logger.exception("Failed during feature engineering: %s", e)

    logger.info('Completed feature_engineering.')
    
    return df

if __name__ == '__main__':
    feature_engineering()
