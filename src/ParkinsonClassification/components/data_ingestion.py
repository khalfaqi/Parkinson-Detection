import os
import zipfile
from shutil import copyfile
from ParkinsonClassification import logger
from ParkinsonClassification.utils.common import get_size
from ParkinsonClassification.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Menggunakan file ZIP dari path lokal
        '''
        try: 
            source_zip_path = self.config.source_local_file  # Path lokal file ZIP sumber
            zip_download_dir = self.config.local_data_file  # Path tujuan untuk menyimpan ZIP
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            # Salin file ZIP dari path lokal ke tujuan
            copyfile(source_zip_path, zip_download_dir)

            logger.info(f"Copied data from {source_zip_path} to {zip_download_dir}")

            # Cek apakah file berhasil disalin
            if not os.path.exists(zip_download_dir):
                raise FileNotFoundError(f"Failed to copy the ZIP file to: {zip_download_dir}")

        except Exception as e:
            raise e

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted {self.config.local_data_file} to {unzip_path}")

