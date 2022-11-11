import os
import sys
import string
from src.entity.artifacts_entity import DataIngestionArtifacts, DataPreprocessingArtifacts
from src.configuration.s3_opearations import S3Operation
from src.exception import CustomException
from src.constant import *
from src.entity.config_entity import DataPreprocessingConfig
from typing import List
import glob
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DataPreprocessing:
    def __init__(self, data_preprocessing_config: DataPreprocessingConfig, data_ingestion_artifacts: DataIngestionArtifacts, s3_operations: S3Operation):
        self.data_preprocessing_config = data_preprocessing_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.s3_operations = s3_operations

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
    def to_vocabulary(descriptions: dict) -> set:
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
            # Reading the image names in a set
            _images = set(open(img_txt_file_path, 'r').read().strip().split('\n'))
            # Create a list of all the images with their full path names
            img_ = []
            for i in img: # img is list of full path names of all images
                if i[len(image_path):] in _images: # Check if the image belongs to set
                    img_.append(i) # Add it to the list of image
            logger.info("Exited the get_images method of Data Transforamtion class")
            return img_

        except Exception as e:
            raise CustomException(e, sys) from e


    # load clean descriptions into memory
    def prepare_descriptions(self, filename: str, dataset: str) -> dict:
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

    # We're creating here a list of all the captions
    def create_caption_list(self, descriptions: dict) -> list:
        try:
            all_captions = []
            for key, val in descriptions.items():
                for cap in val:
                    all_captions.append(cap)
            return all_captions

        except Exception as e:
            raise CustomException(e, sys) from e

    # Considering only those words which occur at least threshold times in the corpus
    def create_corpus(self, threshold: int, captions: list) -> list:
        try:
            word_counts = {}
            nsents = 0
            for sent in captions:
                nsents += 1
                for w in sent.split(' '):
                    word_counts[w] = word_counts.get(w, 0) + 1
            vocab = [w for w in word_counts if word_counts[w] >= threshold]
            return vocab

        except Exception as e:
            raise CustomException(e, sys) from e

    def convert_index_to_word(self, vocab: list) -> dict:
        try:
            index_to_word = {}
            ix = 1
            for w in vocab:
                index_to_word[ix] = w
                ix += 1
            return index_to_word

        except Exception as e:
            raise CustomException(e, sys) from e

    def convert_word_to_index(self, vocab: list) -> dict:
        try:
            word_to_index = {}
            ix = 1
            for w in vocab:
                word_to_index[w] = ix
                ix += 1
            return word_to_index

        except Exception as e:
            raise CustomException(e, sys) from e

    # converting a dictionary of clean descriptions to a list of descriptions
    @staticmethod
    def to_lines(descriptions: dict) -> list:
        try:
            all_desc = list()
            for key in descriptions.keys():
                [all_desc.append(d) for d in descriptions[key]]
            return all_desc

        except Exception as e:
            raise CustomException(e, sys) from e

    # calculating the length of the description with the most words
    def max_length(self, descriptions: dict) -> int:
        try:
            lines = self.to_lines(descriptions)
            return max(len(d.split()) for d in lines)

        except Exception as e:
            raise CustomException(e, sys) from e

    def generate_word_vectors(self) -> dict:
        try:
            embeddings_index = {}
            self.s3_operations.download_file(bucket_name=BUCKET_NAME, output_file_path=self.data_preprocessing_config.GLOVE_MODEL_PATH, key=S3_GLOVE_MODEL_NAME)
            f = open(self.data_preprocessing_config.GLOVE_MODEL_PATH, encoding="utf-8")
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
            return embeddings_index

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_dense_vectors(self, embedding_dim: int, vocab_size: int, vocab: list):
        try:
            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            wrd_to_idx = self.convert_word_to_index(vocab=vocab)
            emb_index = self.generate_word_vectors()
            for word, i in wrd_to_idx.items():
                #if i < max_words:
                embedding_vector = emb_index.get(word)
                if embedding_vector is not None:
                    # Words not found in the embedding index will be all zeros
                    embedding_matrix[i] = embedding_vector

            return embedding_matrix

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
            cleaned_train_desc = self.clean_descriptions(descriptions=train_mapping)
            cleaned_test_desc = self.clean_descriptions(descriptions=test_mapping)
            logger.info("Cleaned the train and test descriptions")

            # Saving the train and test desacriptions to the artifacts directory
            cleaned_train_desc_path = self.data_preprocessing_config.UTILS.save_descriptions(descriptions=cleaned_train_desc, 
                                                                                                filename=self.data_preprocessing_config.CLEANED_TRAIN_DESC_PATH)
            logger.info(f"saved the train descriptions to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.CLEANED_TRAIN_DESC_PATH)}")
            cleaned_test_desc_path = self.data_preprocessing_config.UTILS.save_descriptions(descriptions=cleaned_test_desc, 
                                                                                                filename=self.data_preprocessing_config.CLEANED_TEST_DESC_PATH)
            logger.info(f"saved the test descriptions to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.CLEANED_TEST_DESC_PATH)}")

            # Reading Train and Test image names from .txt file
            train_image_txt_names = self.load_image_name_set(filename=self.data_ingestion_artifacts.train_token_file_path)
            test_image_txt_names = self.load_image_name_set(filename=self.data_ingestion_artifacts.test_token_file_path)  
            logger.info("Loaded train and test image names from txt file")

            # Getting Train and Test imagaes from data ingestion artifacst directory
            train_img = self.get_images(image_path=self.data_ingestion_artifacts.image_data_dir, 
                                                        img_txt_file_path=self.data_ingestion_artifacts.train_image_txt_file_path) 
            test_img = self.get_images(image_path=self.data_ingestion_artifacts.image_data_dir, 
                                                        img_txt_file_path=self.data_ingestion_artifacts.test_image_txt_file_path)
            logger.info("Got Train and test images")

            # saving the train and test images with their full path to artifacts directory
            self.data_preprocessing_config.UTILS.dump_pickle_file(output_filepath=self.data_preprocessing_config.TRAIN_IMAGE_WITH_PATH, data=train_img) 
            
            logger.info(f"Saved train images to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.TRAIN_IMAGE_WITH_PATH)}")
            self.data_preprocessing_config.UTILS.dump_pickle_file(output_filepath=self.data_preprocessing_config.TEST_IMAGE_WITH_PATH, data=test_img)
            logger.info(f"Saved test images to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.TEST_IMAGE_WITH_PATH)}")

            # Preparing the cleaned descriptions.
            prepared_train_descriptions = self.prepare_descriptions(filename=self.data_preprocessing_config.CLEANED_TRAIN_DESC_PATH, dataset=train_image_txt_names)
            #test_descriptions = self.prepare_descriptions(filename=self.data_preprocessing_config.CLEANED_TEST_DESC_PATH, dataset=test_image_txt_names)
            logger.info("Prepared the train and test descriptions")

            # Saving the cleaned descriptions to the artifacts directory.
            self.data_preprocessing_config.UTILS.dump_pickle_file(output_filepath=self.data_preprocessing_config.PREPARED_TRAIN_DESC_PATH, 
                                                                                            data=prepared_train_descriptions)
            logger.info(f"Saved the train descriptions to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.PREPARED_TRAIN_DESC_PATH)}")

            # logger.info(f"Saved the test descriptions with image names to the artifacts directory. File name - {os.path.basename(self.data_preprocessing_config.TEST_IMAGE_WITH_CLEANED_DESC_PATH)}")

            all_train_captions = self.create_caption_list(descriptions=prepared_train_descriptions)

            vocab = self.create_corpus(threshold=WORD_COUNT_THRESHOLD, captions=all_train_captions)
            #print(vocab)

            vocab_dict = self.convert_index_to_word(vocab=vocab)
            #print(vocab_dict)
            word_to_index = self.convert_word_to_index(vocab=vocab)

            self.data_preprocessing_config.UTILS.dump_pickle_file(output_filepath=self.data_preprocessing_config.WORD_TO_INDEX_PATH, data=word_to_index)

            vocab_size = len(vocab_dict) + 1   # one for appended 0's

            max_length = self.max_length(descriptions=prepared_train_descriptions)

            embedding_matrix = self.get_dense_vectors(embedding_dim=EMBEDDING_DIM, vocab_size=vocab_size, vocab=vocab)

            self.data_preprocessing_config.UTILS.dump_pickle_file(output_filepath=self.data_preprocessing_config.EMBEDDING_MATRIX_PATH, 
                                                                                            data=embedding_matrix)

            data_preprocessing_artifacts = DataPreprocessingArtifacts(cleaned_train_desc_path=cleaned_train_desc_path,
                                                                        cleaned_test_desc_path=cleaned_test_desc_path,
                                                                        max_length=max_length,
                                                                        vocab_size=vocab_size,
                                                                        prepared_train_description_path=self.data_preprocessing_config.PREPARED_TRAIN_DESC_PATH,
                                                                        embedding_matrix_path=self.data_preprocessing_config.EMBEDDING_MATRIX_PATH,
                                                                        word_to_index_path=self.data_preprocessing_config.WORD_TO_INDEX_PATH,
                                                                        train_image_path=self.data_preprocessing_config.TRAIN_IMAGE_WITH_PATH,
                                                                        test_image_path=self.data_preprocessing_config.TEST_IMAGE_WITH_PATH)

            logger.info("Exited the initiate_data_preprocessing method of Data Preprocessing class")
            return data_preprocessing_artifacts      

        except Exception as e:
            raise CustomException(e, sys) from e 
