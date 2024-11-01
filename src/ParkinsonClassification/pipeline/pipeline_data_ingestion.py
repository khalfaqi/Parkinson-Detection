from ParkinsonClassification.config.configuration import ConfigurationManager
from ParkinsonClassification.components.data_ingestion import DataIngestion
from ParkinsonClassification import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_data_ingestion_config()  # Ambil konfigurasi saat inisialisasi

    def main(self):
        data_ingestion = DataIngestion(config=self.config)
        data_ingestion.extract_zip_file()  # Menyesuaikan jika Anda tidak perlu mendownload file lagi

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
