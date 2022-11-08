from dataclasses import dataclass

# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    image_data_dir: str
    train_token_file_path: str
    test_token_file_path: str
    train_image_file_path: str
    test_image_file_path: str