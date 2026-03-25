import os
from pathlib import Path

# Root Directory
ROOT_DIR = Path(__file__).parent

URBAN100_NAME = "Urban100"
BSD100_NAME = "BSD100"
COCO_NAME = "COCO"

# Dataset Paths
DATASET_DIR = ROOT_DIR / "datasets"
URBAN100_DIR = DATASET_DIR / URBAN100_NAME
BSD100_DIR = DATASET_DIR / BSD100_NAME
COCO_DIR = DATASET_DIR / COCO_NAME

# ZIP Paths
URBAN100_ZIP = DATASET_DIR / f"{URBAN100_NAME}.zip"
BSD100_ZIP = DATASET_DIR / f"{BSD100_NAME}.zip"
COCO_ZIP = DATASET_DIR / f"{COCO_NAME}.zip"

# Output Paths
OUTPUT_DIR = ROOT_DIR / "outputs"
ZSSR_OUTPUT_DIR = OUTPUT_DIR / "zssr"
SRRESNET_OUTPUT_DIR = OUTPUT_DIR / "srresnet"

# Checkpoints Paths
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZSSR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SRRESNET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)