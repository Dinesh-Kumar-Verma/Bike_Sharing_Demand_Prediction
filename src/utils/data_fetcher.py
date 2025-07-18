import os
import boto3
import gdown
from botocore.exceptions import NoCredentialsError, ClientError
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import RAW_DATA_FILE

# Initialize logger
logger = get_logger("data_fetcher", "utils_data_fetcher.log")

def fetch_raw_data(source, file_name, gdrive_url=None, s3_bucket=None, s3_key=None):
    """
    Fetch raw data from Google Drive or AWS S3.

    Parameters
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

    Returns
    -------
    str
        Path to the downloaded raw data file
    """
    if source not in ("gdrive", "s3"):
        logger.error("Invalid source specified. Must be 'gdrive' or 's3'.")
        raise ValueError("Source must be 'gdrive' or 's3'")

    output_path = RAW_DATA_FILE

    try:
        if source == "gdrive":
            if not gdrive_url:
                logger.error("Google Drive URL is missing for gdrive source")
                raise ValueError("Google Drive URL must be provided for gdrive source")

            logger.info(f"Downloading {file_name} from Google Drive...")
            result = gdown.download(url=gdrive_url, output=str(output_path), fuzzy=True)
            if not result:
                logger.error("Failed to download from Google Drive.")
                raise RuntimeError("gdown failed to download file")
            logger.info(f"Downloaded {file_name} to {output_path}")

        elif source == "s3":
            if not s3_bucket or not s3_key:
                logger.error("S3 bucket or key missing for s3 source")
                raise ValueError("S3 bucket and key must be provided for s3 source")

            logger.info(f"Downloading {file_name} from S3 bucket: {s3_bucket}/{s3_key}...")
            s3 = boto3.client("s3")
            s3.download_file(s3_bucket, s3_key, output_path)
            logger.info(f"Downloaded {file_name} to {output_path}")

        return output_path

    except NoCredentialsError as e:
        logger.exception("AWS credentials not found. Check your .env or AWS config.")
        raise e
    except ClientError as e:
        logger.exception("AWS S3 client error occurred.")
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error occurred while fetching data from {source}")
        raise e