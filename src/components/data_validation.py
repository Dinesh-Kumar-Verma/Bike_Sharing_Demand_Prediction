import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import RAW_DATA_FILE


logger = get_logger(name='data_validation', log_file='data_validation.log')

def data_validation():
    logger.info('\n' + '-' * 80)  # Separator line for each run
    logger.info('Starting data_validation...')
    # Load your data
    df = pd.read_csv(RAW_DATA_FILE, sep=',',encoding='latin')
    
    # Check for missing values
    missing = df.isnull().sum()
    logger.info(f'Missing values:\n{missing}')
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    logger.info(f'Duplicate rows: {duplicates}')
    
    # Check data types
    logger.info(f'Data types:\n{df.dtypes}')
    
    logger.info('Completed data_validation.')

if __name__ == '__main__':
    data_validation()