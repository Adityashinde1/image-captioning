import sys
from src.configuration.s3_opearations import S3Operation
from src.entity.artifacts_entity import (
    ModelTrainerArtifacts,
    ModelPusherArtifacts
)
from src.entity.config_entity import ModelPusherConfig
from src.exception import CustomException
import logging

# initializing logger
logger = logging.getLogger(__name__)


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifacts: ModelTrainerArtifacts,
        S3_operations: S3Operation
    ):

        self.model_pusher_config = model_pusher_config
        self.S3_operations = S3_operations
        self.model_trainer_artifacts = model_trainer_artifacts

    # this is method is used to initiate model pusher
    def initiate_model_pusher(self) -> ModelPusherArtifacts:

        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher. 
        
        Output      :    Model pusher artifact 
        """
        logger.info("Entered initiate_model_pusher method of ModelTrainer class")
        try:
            # Uploading the best model to s3 bucket
            self.S3_operations.upload_file(
                self.model_trainer_artifacts.trained_model_path,
                self.model_pusher_config.S3_MODEL_KEY_PATH,
                self.model_pusher_config.BUCKET_NAME,
                remove=False,
            )

            logger.info("Uploaded best model to s3 bucket")
            logger.info("Exited initiate_model_pusher method of ModelTrainer class")


            # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH,
            )

            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e