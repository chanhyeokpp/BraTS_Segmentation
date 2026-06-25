import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "unet_epoch_1.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

_DATA_DIR_CANDIDATES = [
    os.environ.get("BRATS2021_DATA_DIR"),
    PROJECT_ROOT / "data" / "BraTS2021_Training_Data",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "BraTS2021_Training_Data",
]


def _has_patient_dirs(path):
    if not path or not Path(path).expanduser().is_dir():
        return False
    path = Path(path).expanduser()
    return any(child.is_dir() and child.name.startswith("BraTS2021") for child in path.iterdir())


def get_data_dir():
    for candidate in _DATA_DIR_CANDIDATES:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if _has_patient_dirs(path):
            return str(path)
    return str(PROJECT_ROOT / "data" / "BraTS2021_Training_Data")


DATA_DIR = get_data_dir()
