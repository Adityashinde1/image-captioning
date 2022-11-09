from dataclasses import dataclass

# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    image_data_dir: str
    train_token_file_path: str
    test_token_file_path: str
    train_image_txt_file_path: str
    test_image_txt_file_path: str

@dataclass
class DataTransformationArtifacts:
    cleaned_train_desc_path: str
    cleaned_test_desc_path: str
    train_img_path: str
    test_img_path: str
    train_img_with_cleaned_desc: str
    test_img_with_cleaned_desc: str

@dataclass
class ModelTrainerArtifacts:
    pass