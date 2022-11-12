import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig, ModelTrainerConfig, ModelPusherConfig
from src.entity.artifacts_entity import DataIngestionArtifacts, DataPreprocessingArtifacts, ModelPusherArtifacts, ModelTrainerArtifacts
from src.entity.artifacts_entity import DataIngestionArtifacts
from src.configuration.s3_opearations import S3Operation
from src.exception import CustomException
from src.constant import *
import logging

# initializing logger
logger = logging.getLogger(__name__)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.s3_operations = S3Operation()


    # This method is used to start the data ingestion
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logger.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logger.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config, S3_operations=S3Operation())
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Got the train_set and test_set from mongodb")
            logger.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e


    # This method is used to start the data validation
    def start_data_preprocessing(
        self, data_ingestion_artifact: DataIngestionArtifacts
    ) -> DataPreprocessingArtifacts:
        logger.info("Entered the start_data_preprocessing method of TrainPipeline class")
        try:
            data_preprocessing = DataPreprocessing(data_preprocessing_config=self.data_preprocessing_config, data_ingestion_artifacts=data_ingestion_artifact,
            s3_operations=self.s3_operations)

            data_preprocessing_artifact = data_preprocessing.initiate_data_preprocessing()
            logger.info("Performed the data validation operation")
            logger.info(
                "Exited the start_data_preprocessing method of TrainPipeline class"
            )
            return data_preprocessing_artifact

        except Exception as e:
            raise CustomException(e, sys) from e


    # This method is used to start the model trainer
    def start_model_trainer(
        self, data_preprocessing_artifact: DataPreprocessingArtifacts, data_ingestion_artifact: DataIngestionArtifacts
    ) -> ModelTrainerArtifacts:
        try:
            model_trainer = ModelTrainer(
                data_preprocessing_artifacts=data_preprocessing_artifact,
                model_trainer_config=self.model_trainer_config,
                data_ingestion_artifacts=data_ingestion_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys) from e


    # This method is used to start the model pusher
    def start_model_pusher(
        self,
        model_trainer_artifacts: ModelTrainerArtifacts,
        s3_operation: S3Operation,
    ) -> ModelPusherArtifacts:
        logger.info("Entered the start_model_pusher method of TrainPipeline class")
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifacts=model_trainer_artifacts,
                S3_operations=s3_operation,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logger.info("Initiated the model pusher")
            logger.info("Exited the start_model_pusher method of TrainPipeline class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e            


    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_preprocessing_artifact = self.start_data_preprocessing(data_ingestion_artifact=data_ingestion_artifact) 
            model_trainer_artifact = self.start_model_trainer(data_preprocessing_artifact=data_preprocessing_artifact, data_ingestion_artifact=data_ingestion_artifact)
            model_pusher_artifact = self.start_model_pusher(model_trainer_artifacts=model_trainer_artifact, s3_operation=self.s3_operations)

        except Exception as e:
            raise CustomException(e, sys) from e