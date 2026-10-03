import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# -----------------------------
# Models
# -----------------------------

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


YUNET_MODEL = os.path.join(
    MODEL_DIR,
    "face_detection_yunet_2023mar.onnx"
)


SFACE_MODEL = os.path.join(
    MODEL_DIR,
    "face_recognition_sface_2021dec.onnx"
)



# -----------------------------
# Dataset
# -----------------------------

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)


DATABASE_DIR = os.path.join(
    BASE_DIR,
    "database"
)


DATABASE_FILE = os.path.join(
    DATABASE_DIR,
    "faces.pkl"
)



# -----------------------------
# Camera
# -----------------------------

CAMERA_INDEX = 0

FRAME_WIDTH = 640

FRAME_HEIGHT = 480

FPS = 30



# -----------------------------
# Face
# -----------------------------

FACE_SIZE = (112,112)


COSINE_THRESHOLD = 0.363