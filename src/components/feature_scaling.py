from utils.logger import get_logger

logger = get_logger('feature_scaling')

def feature_scaling():
    logger.info('Starting feature_scaling...')
    
    # Define categorical feature.
    categorical_feature = ['hour','seasons', 'holiday', 'func_day', 'day', 'month', 'year', 'weekday']
    
    # Define numerical features
    numerical_features = ['temperature', 'humidity', 'wind_speed', 'visibility', 'dew_point_temp', 'solar_radiation', 'rainfall', 'snowfall']

    logger.info('Completed feature_scaling.')

if __name__ == '__main__':
    feature_scaling()
