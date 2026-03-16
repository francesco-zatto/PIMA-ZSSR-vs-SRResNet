import os
from pathlib import Path

# Root Directory
ROOT_DIR = Path(__file__).parent

# Dataset Paths
DATASET_DIR = ROOT_DIR / "datasets"
URBAN100_DIR = DATASET_DIR / "Urban100"
BSD100_DIR = DATASET_DIR / "BSD100"

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