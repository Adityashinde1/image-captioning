import os
import sys
import string
from src.entity.artifacts_entity import DataIngestionArtifacts, DataPreprocessingArtifacts
from src.exception import CustomException
from src.entity.config_entity import DataPreprocessingConfig
from typing import List
import glob
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DataPreprocessing:
    def __init__(self, data_preprocessing_config: DataPreprocessingConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_preprocessing_config = data_preprocessing_config
        self.data_ingestion_artifacts = data_ingestion_artifacts


    @staticmethod
    def mapping_descriptions_with_image_name(doc) -> dict:
        logger.info("Entered the mapping_descriptions_with_image_name method")
        try:
            mapping = dict()
            # process lines
            for line in doc.split('\n'):
                # split line by white space
                tokens = line.split()
                if len(line) < 2:
                    continue
                    # take the first token as the image id, the rest as the description
                image_id, image_desc = tokens[0], tokens[1:]
                # extract filename from image id
                image_id = image_id.split('.')[0]
                # convert description tokens back to string
                image_desc = ' '.join(image_desc)
                # create the list if needed
                if image_id not in mapping:
                    mapping[image_id] = list()
                    # store description
                mapping[image_id].append(image_desc)
            logger.info("Exited the mapping_descriptions_with_image_name method")
            return mapping

        except Exception as e:
            raise CustomException(e, sys) from e


    @staticmethod
    def clean_descriptions(descriptions) -> dict:
        logger.info("Entered the clean_descriptions method.")
        try:
            # prepare translation table for removing punctuation
            _desc = {}
            table = str.maketrans('', '', string.punctuation)
            for key, desc_list in descriptions.items():
                for i in range(len(desc_list)):
                    desc = desc_list[i]
                    # tokenize
                    desc = desc.split()
                    # convert to lower case
                    desc = [word.lower() for word in desc]
                    # remove punctuation from each token
                    desc = [w.translate(table) for w in desc]
                    # remove hanging 's' and 'a'
                    desc = [word for word in desc if len(word)>1]
                    # remove tokens with numbers in them
                    desc = [word for word in desc if word.isalpha()]
                    # store as string
                    desc_list[i] =  ' '.join(desc)
                    _desc[key] = desc_list
            logger.info("Exited the clean_descriptions method.")  
            return _desc

        except Exception as e:
            raise CustomException(e, sys) from e


    @staticmethod
    def to_vocabulary(descriptions) -> set:
        logger.info("Entered the to_vocabulary method")
        try:
            # build a list of all description strings
            all_desc = set()
            for key in descriptions.keys():
                [all_desc.update(d.split()) for d in descriptions[key]]
            logger.info("Exited the to_vocabulary method")
            return all_desc

        except Exception as e:
            raise CustomException(e, sys) from e


    def load_image_name_set(self, filename) -> set:
        logger.info("Entered the load_image_name_set methos of Data Preprocessing class")
        try:
            doc = self.data_preprocessing_config.UTILS.read_txt_file(filename)
            dataset = list()
            # process line by line
            for line in doc.split('\n'):
                # skip empty lines
                if len(line) < 1:
                    continue
                # get the image identifier
                identifier = line.split('.')[0]
                dataset.append(identifier)
            logger.info("Exited the load_image_name_set methos of Data Preprocessing class")
            return set(dataset)

        except Exception as e:
            raise CustomException(e, sys) from e


    def get_images(self, image_path:str, img_txt_file_path: str) -> List[str]:
        logger.info("Entered the get_images method of Data Transforamtion class")
        try:
            # Create a list of all image names in the directory
            img = glob.glob(image_path + "*.jpg")
            # Reading the train image names in a set
            train_images = set(open(img_txt_file_path, 'r').read().strip().split('\n'))
            # Create a list of all the training images with their full path names
            train_img = []
            for i in img: # img is list of full path names of all images
                if i[len(image_path):] in train_images: # Check if the image belongs to training set
                    train_img.append(i) # Add it to the list of train image
            logger.info("Exited the get_images method of Data Transforamtion class")
            return train_img

        except Exception as e:
            raise CustomException(e, sys) from e


    # load clean descriptions into memory
    def prepare_descriptions(self, filename, dataset) -> dict:
        logger.info("Entered the prepare_descriptions method of Data Preprocessing class")
        try:
            # load document
            doc = self.data_preprocessing_config.UTILS.read_txt_file(filename)
            descriptions = dict()
            for line in doc.split('\n'):
                # split line by white space
                tokens = line.split()
                # split id from description
                image_id, image_desc = tokens[0], tokens[1:]
                # skip images not in the set
                if image_id in dataset:
                    # create list
                    if image_id not in descriptions:
                        descriptions[image_id] = list()
                    # wrap description in tokens
                    desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                    # store
                    descriptions[image_id].append(desc)
            logger.info("Exited the prepare_descriptions method of Data Preprocessing class")
            return descriptions

        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_data_preprocessing(self) -> DataPreprocessingArtifacts:
        logger.info("Entered the initiate_data_preprocessing method of Data Preprocessing class")
        try:
            os.makedirs(self.data_preprocessing_config.DATA_PREPROCESSING_ARTIFACTS_DIR, exist_ok=True)
            logger.info(
                f"Created {os.path.basename(self.data_preprocessing_config.DATA_PREPROCESSING_ARTIFACTS_DIR)} directory."
            )

            # Mapping the train descriptions with train image names
            train_token_file_path = self.data_ingestion_artifacts.train_token_file_path
            train_token_file = self.data_preprocessing_config.UTILS.read_txt_file(train_token_file_path)
            train_mapping = self.mapping_descriptions_with_image_name(doc=train_token_file)
            logger.info("Mapped train descriptions")

            # Mapping the test descriptions with test image names
            test_token_file_path = self.data_ingestion_artifacts.test_token_file_path
            test_token_file = self.data_preprocessing_config.UTILS.read_txt_file(test_token_file_path)
            test_mapping = self.mapping_descriptions_with_image_name(doc=test_token_file)
            logger.info("Mapped test descriptions")

            # Cleaning the descriptions
            cleaned_train_mapping = self.clean_descriptions(descriptions=train_mapping)
            cleaned_test_mapping = self.clean_descriptions(descriptions=test_mapping)
            logger.info("Cleaned the train and test descriptions")

            # Saving the train and test desacriptions to the artifacts directory
            cleaned_train_desc_path = self.data_preprocessing_config.UTILS.save_descriptions(descriptions=cleaned_train_mapping, 
                                                                                                filename=self.data_preprocessing_config.CLEANED_TRAIN_DESC_PATH)
            logger.info(f"saved the train descriptions to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.CLEANED_TRAIN_DESC_PATH)}")
            cleaned_test_desc_path = self.data_preprocessing_config.UTILS.save_descriptions(descriptions=cleaned_test_mapping, 
                                                                                                filename=self.data_preprocessing_config.CLEANED_TEST_DESC_PATH)
            logger.info(f"saved the test descriptions to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.CLEANED_TEST_DESC_PATH)}")

            # Reading Train and Test image names from .txt file
            train_image_txt_names = self.load_image_name_set(filename=self.data_ingestion_artifacts.train_token_file_path)
            test_image_txt_names = self.load_image_name_set(filename=self.data_ingestion_artifacts.test_token_file_path)  
            logger.info("Loaded train and test image names from txt file")

            # Getting Train and Test imagaes from data ingestion artifacst directory
            train_images = self.get_images(image_path=self.data_ingestion_artifacts.image_data_dir, 
                                                        img_txt_file_path=self.data_ingestion_artifacts.train_image_txt_file_path) 
            test_images = self.get_images(image_path=self.data_ingestion_artifacts.image_data_dir, 
                                                        img_txt_file_path=self.data_ingestion_artifacts.test_image_txt_file_path)
            logger.info("Got Train and test images")

            # saving the train and test images with their full path to artifacts directory
            train_image_path = self.data_preprocessing_config.UTILS.save_txt_file(output_file_path=self.data_preprocessing_config.TRAIN_IMAGE_WITH_PATH, 
                                                                                                    data=train_images) 
            logger.info(f"Saved train images to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.TRAIN_IMAGE_WITH_PATH)}")
            test_image_path = self.data_preprocessing_config.UTILS.save_txt_file(output_file_path=self.data_preprocessing_config.TEST_IMAGE_WITH_PATH, 
                                                                                                    data=test_images)
            logger.info(f"Saved test images to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.TEST_IMAGE_WITH_PATH)}")

            # Preparing the cleaned descriptions.
            train_img_with_cleaned_desc = self.prepare_descriptions(filename=self.data_preprocessing_config.CLEANED_TRAIN_DESC_PATH, dataset=train_image_txt_names)
            test_img_with_cleaned_desc = self.prepare_descriptions(filename=self.data_preprocessing_config.CLEANED_TEST_DESC_PATH, dataset=test_image_txt_names)
            logger.info("Prepared the train and test descriptions")

            # Saving the cleaned descriptions to the artifacts directory.
            train_img_with_cleaned_desc_path = self.data_preprocessing_config.UTILS.save_txt_file(output_file_path=self.data_preprocessing_config.TRAIN_IMAGE_WITH_CLEANED_DESC_PATH, 
                                                                                                    data=train_img_with_cleaned_desc)
            logger.info(f"Saved the train descriptions with image names to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.TRAIN_IMAGE_WITH_CLEANED_DESC_PATH)}")
            test_img_with_cleaned_desc_path = self.data_preprocessing_config.UTILS.save_txt_file(output_file_path=self.data_preprocessing_config.TEST_IMAGE_WITH_CLEANED_DESC_PATH, 
                                                                                                    data=test_img_with_cleaned_desc)
            logger.info(f"Saved the test descriptions with image names to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.TEST_IMAGE_WITH_CLEANED_DESC_PATH)}")

            data_transformation_artifacts = DataPreprocessingArtifacts(cleaned_train_desc_path=cleaned_train_desc_path,
                                                                        cleaned_test_desc_path=cleaned_test_desc_path,
                                                                        train_img_path=train_image_path,
                                                                        test_img_path=test_image_path,
                                                                        train_img_with_cleaned_desc=train_img_with_cleaned_desc_path,
                                                                        test_img_with_cleaned_desc=test_img_with_cleaned_desc_path)

            logger.info("Exited the initiate_data_preprocessing method of Data Preprocessing class")
            return data_transformation_artifacts      

        except Exception as e:
            raise CustomException(e, sys) from e 
