from src.utils.logger import get_logger
import pandas as pd
from src.utils.config import RAW_DATA_FILE, PROCESSED_FILE


logger = get_logger(name='data_preprocessing', log_file='data_preprocessing.log')

def data_preprocessing():
    logger.info('\n' + '-' * 80)
    logger.info('Starting data_preprocessing...')
    try:
        # load data
        df = pd.read_csv(RAW_DATA_FILE, sep=',',encoding='latin')
        # Rename the columns names to ensure compatibility, readability, and prevent errors.
        df.rename(
            columns= {
                'Date':'date',
                'Rented Bike Count': 'rented_bike_count',
                'Hour':'hour',
                'Temperature(°C)':'temperature',
                'Humidity(%)':'humidity',
                'Wind speed (m/s)': 'wind_speed',
                'Visibility (10m)': 'visibility',
                'Dew point temperature(°C)':'dew_point_temp',
                'Solar Radiation (MJ/m2)': 'solar_radiation',
                'Rainfall(mm)': 'rainfall',
                'Snowfall (cm)':'snowfall',
                'Seasons':'seasons',
                'Holiday':'holiday',
                'Functioning Day':'func_day'
                },
            inplace=True
            )
        
        # convert date feature into datetime data type
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        
        # Split date into year, month, day & day_name
        df['day'] = df['date'].dt.day
        df['year'] = df['date'].dt.year
        df['weekday'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.strftime('%b')

        # drop data column
        df.drop('date', axis = 1, inplace = True)
        
        # save the data
        df.to_csv(PROCESSED_FILE, index=False)
        logger.info('Processed data saved to data/processed/bike_sharing_data_processed.csv')
    
    except Exception as e:
        logger.exception("Failed during data preprocessing: %s", e)
        raise
              
    logger.info('Completed data_preprocessing.')

if __name__ == '__main__':
    data_preprocessing()
