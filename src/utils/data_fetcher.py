import os
import boto3
import gdown
from botocore.exceptions import NoCredentialsError, ClientError
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("data_fetcher", "utils_data_fetcher.log")

# Create raw data directory if not exists
RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def fetch_raw_data(source, file_name, gdrive_url=None, s3_bucket=None, s3_key=None):
    """
    Fetch raw data from Google Drive or AWS S3.

    Parameters:
    ----------
    source : str
        Data source: 'gdrive' or 's3'
    file_name : str
        Output file name (saved in data/raw/)
    gdrive_url : str, optional
        Google Drive file URL (for gdrive source)
    s3_bucket : str, optional
        S3 bucket name (for s3 source)
    s3_key : str, optional
        S3 object key/path (for s3 source)

    Returns:
    -------
    str
        Path to the downloaded raw data file
    """
    output_path = os.path.join(RAW_DATA_DIR, file_name)

    try:
        if source == "gdrive":
            if not gdrive_url:
                logger.error("Google Drive URL is missing for gdrive source")
                raise ValueError("Google Drive URL must be provided for gdrive source")

            logger.info(f"Downloading {file_name} from Google Drive...")
            gdown.download(url= gdrive_url, output= output_path, fuzzy=True)
            logger.info(f"Downloaded {file_name} to {output_path}")

        elif source == "s3":
            if not s3_bucket or not s3_key:
                logger.error("S3 bucket or key missing for s3 source")
                raise ValueError("S3 bucket and key must be provided for s3 source")

            logger.info(f"Downloading {file_name} from S3 bucket: {s3_bucket}/{s3_key}...")
            s3 = boto3.client("s3")
            s3.download_file(s3_bucket, s3_key, output_path)
            logger.info(f"Downloaded {file_name} to {output_path}")

        else:
            logger.error("Invalid source specified. Must be 'gdrive' or 's3'")
            raise ValueError("Source must be 'gdrive' or 's3'")

    except NoCredentialsError as e:
        logger.exception("AWS credentials not found. Check your .env or config.")
        raise e
    except ClientError as e:
        logger.exception("AWS S3 client error occurred.")
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error occurred while fetching data from {source}")
        raise e

    return output_path
