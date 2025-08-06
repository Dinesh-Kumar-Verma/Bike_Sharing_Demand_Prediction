# src/components/data_ingestion.py

from src.utils.logger import get_logger
from src.utils.data_fetcher import fetch_raw_data
from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_csv
from src.utils.config import RAW_DATA_FILE
from src.utils.data_saver import save_split_data
from src.utils.params import load_params


logger = get_logger(name='data_ingestion', log_file='data_ingestion.log')   



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


def main():
    try:
        params = load_params()
        data_split_params = params['data_split']
        test_size = data_split_params['test_size']
        val_size = data_split_params['val_size']
        random_state = data_split_params['random_state']
        mode = data_split_params['mode']
        time_series_splits = data_split_params['time_series_splits']

        ingestion = DataIngestion()
        file_path = ingestion.ingest(
            source='gdrive',
            file_name='bike_sharing_data.csv',
            url='https://drive.google.com/file/d/1Jpw26yBmQD055UkCi-eSZSb3rFicUqWr/view?usp=sharing'
        )
        print(f"Data saved to: {file_path}")
        
        # Split and save the splitted data
        raw_data_file = load_csv(RAW_DATA_FILE)
        splitter = DataSplitter(mode=mode, test_size=test_size, val_size=val_size, random_state=random_state, time_series_splits=time_series_splits)
        X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(raw_data_file, target_column='Rented Bike Count')
        save_split_data(X_train, y_train, X_val, y_val, X_test, y_test)
        
    except Exception as e:
        logger.exception("Data ingestion failed: %s", e)
        raise


if __name__ == '__main__':
    main()

