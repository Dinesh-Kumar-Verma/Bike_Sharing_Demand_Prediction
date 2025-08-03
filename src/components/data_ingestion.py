# src/components/data_ingestion.py

from src.utils.logger import get_logger
from src.utils.data_fetcher import fetch_raw_data
from src.utils.config import BASE_DIR



class DataIngestion:
    def __init__(self, log_file='data_ingestion.log'):
        self.logger = get_logger(name='data_ingestion', log_file=log_file)

    def ingest(self, source: str, file_name: str, url: str) -> str:
        """
        Ingests data from specified source. Currently supports: 'gdrive'.
        Returns path to saved file.
        """
        self.logger.info('\n' + '-' * 80)
        self.logger.info('Starting data ingestion...')

        try:
            if source == 'gdrive':
                path = fetch_raw_data(
                    source=source,
                    file_name=file_name,
                    gdrive_url=url
                )
                self.logger.info(f'Data successfully ingested from GDrive to: {path}')
            else:
                raise NotImplementedError(f"Source '{source}' not supported yet.")

            self.logger.info('Completed data ingestion.')
            return path

        except Exception as e:
            self.logger.exception("Failed during data ingestion: %s", e)
            raise


if __name__ == '__main__':
    ingestion = DataIngestion()
    file_path = ingestion.ingest(
        source='gdrive',
        file_name='bike_sharing_data.csv',
        url='https://drive.google.com/file/d/1Jpw26yBmQD055UkCi-eSZSb3rFicUqWr/view?usp=sharing'
    )
    print(f"Data saved to: {file_path}")