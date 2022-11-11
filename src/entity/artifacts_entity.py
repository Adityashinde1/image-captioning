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
class DataPreprocessingArtifacts:
    cleaned_train_desc_path: str
    cleaned_test_desc_path: str
    max_length: int
    vocab_size: int
    prepared_train_description_path: str
    embedding_matrix_path: str
    word_to_index_path: str
    train_image_path: str
    test_image_path: str
    
@dataclass
class ModelTrainerArtifacts:
    train_image_features_path: str
    test_image_features_path: str
    trained_model_path: str    