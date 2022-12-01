import os
import sys
from src.exception import CustomException
from zipfile import ZipFile, Path
import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifacts
from src.configuration.s3_opearations import S3Operation
from src.constant import *

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, S3_operations: S3Operation):
        self.data_ingestion_config = data_ingestion_config
        self.S3_operations = S3_operations


    def get_images_from_s3(self, bucket_file_name: str, bucket_name: str, output_filepath: str) -> zip:
        logger.info("Entered the get_data_from_s3 method of Data ingestion class")
        try:
            self.S3_operations.read_data_from_s3(bucket_file_name, bucket_name, output_filepath)

            logger.info("Exited the get_data_from_s3 method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e 


    def unzip_file(self, zip_data_filepath: str, unzip_dir_path: str) -> Path:
        logger.info("Entered the unzip_file method of Data ingestion class")
        try:
            with ZipFile(zip_data_filepath, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir_path)
            logger.info("Exited the unzip_file method of Data ingestion class")
            return unzip_dir_path

        except Exception as e:
            raise CustomException(e, sys) from e 


    def get_tokens_from_s3(self, bucket_name: str, output_file_path: str, key: str) -> Path:
        logger.info("Entered the get_tokens_from_s3 method of Data Ingestion class")
        try:
            self.S3_operations.download_file(bucket_name = bucket_name, output_file_path = output_file_path, 
                                                key = key)
            logger.info("Image tokens file saved") 
            logger.info("Exited the get_tokens_from_s3 method of Data Ingestion class")
            return output_file_path

        except Exception as e:
            raise CustomException(e, sys) from e


    def get_train_image_names_from_s3(self, bucket_name: str, output_file_path: str, key: str) -> Path:
        logger.info("Entered the get_train_image_names_from_s3 method of Data Ingestion class")
        try:
            self.S3_operations.download_file(bucket_name = bucket_name, output_file_path = output_file_path, 
                                                key = key) 
            logger.info("Train image name file saved") 
            logger.info("Exited the get_train_image_names_from_s3 method of Data Ingestion class")
            return output_file_path

        except Exception as e:
            raise CustomException(e, sys) from e


    def get_test_image_names_from_s3(self, bucket_name: str, output_file_path: str, key: str) -> Path:
        logger.info("Entered the get_test_image_names_from_s3 method of Data Ingestion class")
        try:
            self.S3_operations.download_file(bucket_name = bucket_name, output_file_path = output_file_path, 
                                                key = key) 
            logger.info("Test image name file saved") 
            logger.info("Exited the get_test_image_names_from_s3 method of Data Ingestion class")
            return output_file_path
            
        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            # Creating Data Ingestion Artifacts directory inside artifact folder
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            logger.info(
                f"Created {os.path.basename(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)} directory."
            )
            
            self.get_images_from_s3(bucket_file_name=S3_DATA_FOLDER_NAME, bucket_name=BUCKET_NAME,
                                               output_filepath=self.data_ingestion_config.ZIP_DATA_PATH)
            logger.info("Downloaded images zip file from S3 bucket")

            # Unzipping the file
            self.unzip_file(zip_data_filepath=self.data_ingestion_config.ZIP_DATA_PATH, unzip_dir_path=self.data_ingestion_config.UNZIP_FOLDER_PATH)
            logger.info("Extracted the images from the zip file")

            # Reading Train image name file from s3 bucket
            train_image_path = self.get_train_image_names_from_s3(bucket_name=BUCKET_NAME, output_file_path=self.data_ingestion_config.TRAIN_IMAGE_NAMES_PATH,
                                                                     key=S3_TRAIN_IMAGE_NAMES)
            logger.info("Got the train image names")
            # Reading Train image name file from s3 bucket
            test_image_path = self.get_test_image_names_from_s3(bucket_name=BUCKET_NAME, output_file_path=self.data_ingestion_config.TEST_IMAGE_NAMES_PATH, 
                                                                        key=S3_TEST_IMAGE_NAMES)
            logger.info("Got the test image names")

            # Reading train tokens from s3 bucket
            train_token_path = self.get_tokens_from_s3(bucket_name=BUCKET_NAME, output_file_path=self.data_ingestion_config.TRAIN_TOKEN_FILE_PATH, 
                                                        key=S3_TRAIN_TOKEN_FILE_NAME)
            logger.info("got train tokens file from s3 bucket")
            # Reading test tokens from s3 bucket                                                        
            test_token_path = self.get_tokens_from_s3(bucket_name=BUCKET_NAME, output_file_path=self.data_ingestion_config.TEST_TOKEN_FILE_PATH, 
                                                        key=S3_TEST_TOKEN_FILE_NAME)
            logger.info("got test tokens file from s3 bucket")

            # Saving data ingestion artifacts
            data_ingestion_artifacts = DataIngestionArtifacts(image_data_dir=self.data_ingestion_config.DATA_PATH,
                                                                train_token_file_path=train_token_path,
                                                                test_token_file_path=test_token_path,
                                                                train_image_txt_file_path=train_image_path,
                                                                test_image_txt_file_path=test_image_path)

            logger.info("Exited the initiate_data_ingestion method of Data Ingestion class ")
            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
