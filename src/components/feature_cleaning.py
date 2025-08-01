import pandas as pd
from src.utils.logger import get_logger
from src.utils.data_loader import load_csv
from src.utils.config import RAW_DATA_FILE, CLEANED_FEATURE_FILE
from src.utils.data_saver import save_csv



class FeatureCleaner:
    def __init__(self, log_file='feature_cleaning.log'):
        self.logger = get_logger(name='feature_cleaning', log_file=log_file)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame.
        Includes renaming columns and extracting date features.
        """
        try:
            self.logger.info('\n' + '-' * 80)
            self.logger.info('Starting feature cleaning...')
            self.logger.info(f'Data received with shape: {df.shape}')

            df = self._rename_columns(df)
            df = self._process_date_columns(df)

            self.logger.info(f'Data after feature cleaning with shape: {df.shape}')
            self.logger.info('Completed feature cleaning.')

            return df

        except Exception as e:
            self.logger.exception("Error during feature cleaning: %s", e)
            raise

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info('Renaming columns...')
        column_mapping = {
            'Date': 'date',
            'Rented Bike Count': 'rented_bike_count',
            'Hour': 'hour',
            'Temperature(°C)': 'temperature',
            'Humidity(%)': 'humidity',
            'Wind speed (m/s)': 'wind_speed',
            'Visibility (10m)': 'visibility',
            'Dew point temperature(°C)': 'dew_point_temp',
            'Solar Radiation (MJ/m2)': 'solar_radiation',
            'Rainfall(mm)': 'rainfall',
            'Snowfall (cm)': 'snowfall',
            'Seasons': 'seasons',
            'Holiday': 'holiday',
            'Functioning Day': 'func_day'
        }
        return df.rename(columns=column_mapping)

    def _process_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date' not in df.columns:
            self.logger.warning('No "date" column found. Skipping feature extraction.')
            return df

        self.logger.info('Processing date features...')
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        df['day'] = df['date'].dt.day
        df['year'] = df['date'].dt.year
        df['weekday'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.strftime('%b')
        
        return df.drop('date', axis=1)



if __name__ == '__main__':
    df_raw = load_csv(RAW_DATA_FILE)
    raw_FeatureCleaner = FeatureCleaner()
    df_cleaned = raw_FeatureCleaner.transform(df_raw)
    save_csv(df_cleaned, CLEANED_FEATURE_FILE)

