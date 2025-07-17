from src.utils.logger import get_logger
from src.utils.data_fetcher import fetch_raw_data
from src.utils.config import BASE_DIR

logger = get_logger(name='data_ingestion', log_file='data_ingestion.log')

def data_ingestion():
    logger.info('\n' + '-' * 80)  # Separator line for each run
    logger.info('Starting data_ingestion...')
    try:
        fetch_raw_data(
            source='gdrive',
            file_name='bike_sharing_data.csv',
            gdrive_url="https://drive.google.com/file/d/1Jpw26yBmQD055UkCi-eSZSb3rFicUqWr/view?usp=sharing"
        )
        logger.info('Completed data_ingestion.')
    except Exception as e:
        logger.exception("Failed during data_ingestion.")
        raise e

if __name__ == '__main__':
    data_ingestion()
