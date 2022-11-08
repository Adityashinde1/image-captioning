from dataclasses import dataclass
from from_root import from_root
import os
from src.utils.main_utils import MainUtils
from src.constant import *

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR) 
        self.ZIP_DATA_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, S3_DATA_FOLDER_NAME)
        self.UNZIP_FOLDER_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.TRAIN_TOKEN_FILE_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, TRAIN_TOKEN_FILE_NAME)
        self.TEST_TOKEN_FILE_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, TEST_TOKEN_FILE_NAME)        
        self.TRAIN_IMAGE_NAMES_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, TRAIN_IMAGE_NAMES)
        self.TEST_IMAGE_NAMES_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, TEST_IMAGE_NAMES)



