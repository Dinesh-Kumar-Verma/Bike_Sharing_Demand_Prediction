from pathlib import Path
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.utils.logger import get_logger
from src.utils.data_saver import save_csv
from src.utils.config import CLEANED_FEATURE_FILE, PROCESSED_FEATURES_FILE


logger = get_logger(name='feature_engineering', log_file='feature_engineering.log')


class FeatureEngineer:
    def __init__(self, run_vif: bool = True, drop_features: list = None):
        """
        Initialize the feature engineer.

        :param run_vif: Whether to calculate and log VIF.
        :param drop_features: List of additional features to drop.
        """
        self.run_vif = run_vif
        self.drop_features = drop_features or ['dew_point_temp', 'seasons', 'day']
        self.numerical_features = [
            'temperature', 'humidity', 'wind_speed', 'visibility',
            'dew_point_temp', 'solar_radiation', 'rainfall', 'snowfall'
        ]
    def _calculate_vif(self, X: pd.DataFrame, label: str = '', output_path: Path = None) -> pd.DataFrame:
        logger.info(f'Calculating VIF {label}...')
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        if output_path:
            file_name = f'vif_report_{label.replace(" ", "_").lower()}.csv'
            vif_report_path = output_path.parent / file_name
            vif.to_csv(vif_report_path, index=False)
            logger.info(f"VIF ({label}) saved at {vif_report_path.resolve()}")

        logger.info(f"\nVIF {label}:\n{vif}")
        return vif

    def transform(self, df: pd.DataFrame, save_to: Path = None) -> pd.DataFrame:
        """
        Perform feature engineering on input DataFrame.
        :param df: Input dataframe (raw or preprocessed).
        :param save_to: Optional path to save the transformed dataframe.
        :return: Transformed dataframe.
        """
        logger.info('-' * 80)
        logger.info('Starting feature engineering...')

        df = df.copy()

        # 1. Encode categorical features
        df['holiday'] = df['holiday'].map({"Holiday": 1, "No Holiday": 0}).fillna(0).astype(int)
        df['func_day'] = df['func_day'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        logger.info("Encoded 'holiday' and 'func_day'.")

        # 2. One-hot encode hour, month, weekday
        if {'hour', 'month', 'weekday'}.issubset(df.columns):
            df = pd.concat([
                df,
                pd.get_dummies(df['hour'], prefix='hour', drop_first=True),
                pd.get_dummies(df['month'], prefix='month', drop_first=True),
                pd.get_dummies(df['weekday'], prefix='weekday', drop_first=True)
            ], axis=1)
            df.drop(['hour', 'month', 'weekday'], axis=1, inplace=True)
            logger.info('One-hot encoded hour, month, weekday and dropped originals.')

        # 3. Clean column names
        df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '')
                      .replace(',', '').replace("'", "") for col in df.columns]

        # 4. VIF calculation
        if self.run_vif:
            try:
                vif_df = self._calculate_vif(df[self.numerical_features], output_path=save_to)
            except Exception as e:
                logger.warning(f"VIF calculation failed: {e}")

        # 5. Drop unnecessary features
        for feat in self.drop_features:
            if feat in df.columns:
                df.drop(feat, axis=1, inplace=True)
                logger.info(f"Dropped feature '{feat}'")
                
        # 6. VIF after dropping
        if self.run_vif:
            remaining_numerics = [col for col in self.numerical_features if col not in self.drop_features]
            try:
                self._calculate_vif(
                    df[[col for col in remaining_numerics if col in df.columns]], 
                    label='after_dropping',
                    output_path=save_to
                )
            except Exception as e:
                logger.warning(f"VIF after dropping failed: {e}")

        # 7. Save if path is provided
        if save_to:
            save_csv(df, save_to)
            logger.info(f"Feature-engineered data saved at {save_to.resolve()}")

        logger.info('Completed feature engineering.')
        return df


def main():
    df_raw = pd.read_csv(CLEANED_FEATURE_FILE)
    fe = FeatureEngineer(run_vif=True)
    df_transformed = fe.transform(df_raw, save_to=PROCESSED_FEATURES_FILE)
    df_transformed = fe.transform(df_raw)

if __name__ == '__main__':
    main()
    
    
# from src.feature_engineering import FeatureEngineer

# def prepare_input_for_model(user_input_df):
#     fe = FeatureEngineer(run_vif=False)  # No VIF in prod
#     return fe.transform(user_input_df)
