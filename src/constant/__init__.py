import os
from datetime import datetime
from from_root import from_root


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join(from_root(), "artifacts", TIMESTAMP)
LOGS_DIR = 'logs'
LOGS_FILE_NAME = 'image_caption.log' 

BUCKET_NAME = 'image-captioning-io-files'
S3_MODEL_NAME = 'model.h5'
S3_DATA_FOLDER_NAME = "data.zip"
S3_TRAIN_TOKEN_FILE_NAME = 'Flickr8k_train_token.txt'
S3_TEST_TOKEN_FILE_NAME = 'Flickr8k_test_token.txt'
S3_TRAIN_IMAGE_NAMES = 'Flickr_8k.trainImages.txt'
S3_TEST_IMAGE_NAMES = 'Flickr_8k.testImages.txt'

DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
UNZIP_FOLDER_NAME = 'data'
TRAIN_TOKEN_FILE_NAME = 'train_token.txt'
TEST_TOKEN_FILE_NAME = 'test_token.txt'
TRAIN_IMAGE_NAMES = 'train_img.txt'
TEST_IMAGE_NAMES = "test_img.txt"
