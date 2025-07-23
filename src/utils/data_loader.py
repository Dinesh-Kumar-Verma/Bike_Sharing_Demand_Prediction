import pandas as pd
from src.utils.logger import get_logger

# Initialize a logger specific for data loading tasks
logger = get_logger('data_loader')


def load_csv(filepath, sep=',', encoding='latin'):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath (str or Path): Path to the CSV file to load.
        sep (str): Delimiter to use. Default is ',' for standard CSVs.
        encoding (str): File encoding. Default is 'latin' to handle special characters.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        Exception: If loading fails, raises the original exception after logging the error.
    """
    try:
        # Log the start of the loading process
        logger.info("Loading data from %s", filepath)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)
        
        # Log successful loading
        logger.info("Data successfully loaded from %s", filepath)
        
        # Return the DataFrame to the caller
        return df
    
    except Exception as e:
        # Log the error with details if something goes wrong
        logger.error("Failed to load data from %s: %s", filepath, e)
        # Re-raise the exception to propagate it up the call stack
        raise
