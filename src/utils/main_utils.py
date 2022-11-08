import shutil
import sys
from typing import Dict
import dill
import numpy as np
import pandas as pd
import yaml
from src.constant import *
from src.exception import CustomException
import logging
from PIL import Image

# initiatlizing logger
logger = logging.getLogger(__name__)


class MainUtils:
    def read_yaml_file(self, filename: str) -> Dict:
        logger.info("Entered the read_yaml_file method of MainUtils class")
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e


    def save_numpy_array_data(self, file_path: str, array: np.array) -> str:
        logger.info("Entered the save_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
            logger.info("Exited the save_numpy_array_data method of MainUtils class")
            return file_path

        except Exception as e:
            raise CustomException(e, sys) from e


    def load_numpy_array_data(self, file_path: str) -> np.array:
        logger.info("Entered the load_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj)

        except Exception as e:
            raise CustomException(e, sys) from e


    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logger.info("Entered the save_object method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)

            logger.info("Exited the save_object method of MainUtils class")

            return file_path

        except Exception as e:
            raise CustomException(e, sys) from e


    @staticmethod
    def load_object(file_path: str) -> object:
        logger.info("Entered the load_object method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                obj = dill.load(file_obj)
            logger.info("Exited the load_object method of MainUtils class")
            return obj

        except Exception as e:
            raise CustomException(e, sys) from e


    @staticmethod
    def create_artifacts_zip(file_name: str, folder_name: str) -> None:
        logger.info("Entered the create_artifacts_zip method of MainUtils class")
        try:
            shutil.make_archive(file_name, "zip", folder_name)
            logger.info("Exited the create_artifacts_zip method of MainUtils class")

        except Exception as e:
            raise CustomException(e, sys) from e


    @staticmethod
    def unzip_file(filename: str, folder_name: str) -> None:
        logger.info("Entered the unzip_file method of MainUtils class")
        try:
            shutil.unpack_archive(filename, folder_name)
            logger.info("Exited the unzip_file method of MainUtils class")

        except Exception as e:
            raise CustomException(e, sys) from e


    @staticmethod        
    def read_txt_file(filename: str) -> str:
        logger.info("Entered the load_doc method of MainUtils class")
        try:
            # Opening file for read only
            file1 = open(filename, 'r')
            # read all text
            text = file1.read()
            # close the file
            file1.close()
            logger.info("Exited the load_doc method of MainUtils class")
            return text

        except Exception as e:
            raise CustomException(e, sys) from e






