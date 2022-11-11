from dataclasses import dataclass
from from_root import from_root
import os
from models.inception import Inception
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
        self.DATA_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, UNZIP_FOLDER_NAME)

@dataclass
class DataPreprocessingConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        self.DATA_PREPROCESSING_ARTIFACTS_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_PREPROCESSING_ARTIFACTS_DIR)
        self.CLEANED_TRAIN_DESC_PATH: str = os.path.join(self.DATA_PREPROCESSING_ARTIFACTS_DIR, CLEANED_TRAIN_DESC_NAME)
        self.CLEANED_TEST_DESC_PATH: str = os.path.join(self.DATA_PREPROCESSING_ARTIFACTS_DIR, CLEANED_TEST_DESC_NAME)

        self.PREPARED_TRAIN_DESC_PATH: str = os.path.join(self.DATA_PREPROCESSING_ARTIFACTS_DIR, PREPARED_TRAIN_DESC_FILE_NAME)

        self.TRAIN_IMAGE_WITH_PATH: str = os.path.join(self.DATA_PREPROCESSING_ARTIFACTS_DIR, TRAIN_IMAGE_WITH_PATH_NAME)
        self.TEST_IMAGE_WITH_PATH: str = os.path.join(self.DATA_PREPROCESSING_ARTIFACTS_DIR, TEST_IMAGE_WITH_PATH_NAME)
        self.EMBEDDING_MATRIX_PATH: str = os.path.join(self.DATA_PREPROCESSING_ARTIFACTS_DIR, EMBEDDING_MATRIX_FILE_NAME)
        self.WORD_TO_INDEX_PATH: str = os.path.join(self.DATA_PREPROCESSING_ARTIFACTS_DIR, WORD_TO_INDEX_NAME)
        # self.TRAIN_IMAGE_WITH_CLEANED_DESC_PATH: str = os.path.join(self.DATA_PREPROCESSING_ARTIFACTS_DIR, TRAIN_IMAGE_NAME)
        # self.TEST_IMAGE_WITH_CLEANED_DESC_PATH: str = os.path.join(self.DATA_PREPROCESSING_ARTIFACTS_DIR, TEST_IMAGE_NAME)
        self.GLOVE_MODEL_PATH: str = os.path.join(from_root(), MODELS_DIR, GLOVE_MODEL_NAME)


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        self.INCEPTION = Inception(weights=INCEPTION_WEIGHT)
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAIN_FEATURE_PATH: str = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR, TRAIN_FEATURE_FILE_NAME)
        self.TEST_FEATURE_PATH: str = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR, TEST_FEATURE_FILE_NAME)