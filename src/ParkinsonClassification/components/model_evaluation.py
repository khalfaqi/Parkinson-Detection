import tensorflow as tf
from pathlib import Path
import dagshub
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from ParkinsonClassification.entity.config_entity import EvaluationConfig
from ParkinsonClassification.utils.common import read_yaml, create_directories, save_json

# Inisialisasi integrasi DagsHub dengan MLflow pada repositori Parkinson-Detection milik khalfaqi
dagshub.init(repo_owner='khalfaqi', repo_name='Parkinson-Detection', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/khalfaqi/Parkinson-Detection.mlflow')


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        # Konstruktor class untuk menerima konfigurasi evaluasi
        self.config = config

    def _valid_generator(self):
        # Membuat generator data untuk validasi
        datagenerator_kwargs = dict(
            rescale=1./255,             # Normalisasi piksel gambar (0-1)
            validation_split=0.2      # Menggunakan 20% data sebagai data validasi
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Ukuran target gambar dari konfigurasi
            batch_size=self.config.params_batch_size,       # Ukuran batch dari konfigurasi
            interpolation="bilinear"                        # Metode interpolasi gambar
        )

        # Membuat generator untuk validasi dengan ImageDataGenerator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Menghasilkan aliran data validasi dari direktori data pelatihan
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,           # Direktori data pelatihan
            subset="validation",                           # Menggunakan subset validasi
            shuffle=False,                                 # Data tidak diacak
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        # Memuat model yang disimpan dari path yang diberikan
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        # Memuat model dan menjalankan evaluasi pada data validasi
        self.model = self.load_model(self.config.path_of_model)  # Memuat model
        self._valid_generator()                                  # Membuat generator validasi
        self.score = self.model.evaluate(self.valid_generator)   # Mengevaluasi model pada data validasi
        self.save_score()                                        # Menyimpan skor evaluasi

    def save_score(self):
        # Menyimpan hasil evaluasi sebagai file JSON
        scores = {"loss": self.score[0], "accuracy": self.score[1]}  # Menyimpan loss dan akurasi
        save_json(path=Path("scores.json"), data=scores)             # Menyimpan skor dalam scores.json

    
    def log_into_mlflow(self):
        # Logging hasil evaluasi dan model ke MLflow
        mlflow.set_registry_uri(self.config.mlflow_uri)              # Set URI registri MLflow
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme  # Mendapatkan tipe URL untuk pelacakan MLflow
        
        with mlflow.start_run():                                     # Memulai log run di MLflow
            mlflow.log_params(self.config.all_params)                # Logging parameter dari konfigurasi
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})  # Logging loss dan akurasi sebagai metrik
            
            # Melakukan log model ke registry MLflow jika URL tracking bukan lokal
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")