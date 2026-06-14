import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

MODEL_BUCKET_PATH = os.getenv("MODEL_BUCKET_PATH")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

LOCAL_ARTIFACT_DIR = os.getenv("LOCAL_ARTIFACT_DIR", "/app/artifacts")
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "20"))
SEQUENCE_STRIDE = int(os.getenv("SEQUENCE_STRIDE", "5"))
FEATURE_SIZE = int(os.getenv("FEATURE_SIZE", "426"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
