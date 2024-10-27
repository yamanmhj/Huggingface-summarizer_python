import logging
import os

from from_root import from_root
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y%m%d')}_log"
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


logging.basicConfig(
    
    level=logging.DEBUG,
    filename = LOG_FILE_PATH,
    format="%(asctime)s - %(levelname)s - %(message)s",
)