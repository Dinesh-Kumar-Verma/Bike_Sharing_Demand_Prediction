import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def get_logger(name: str, log_file: str = "app.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
   # Add timestamp to log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_with_time = f"{os.path.splitext(log_file)[0]}_{timestamp}.log"
   

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler
    file_handler = logging.FileHandler(os.path.join("logs", log_file))
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add Handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


