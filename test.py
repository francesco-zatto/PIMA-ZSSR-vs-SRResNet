import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from runner.runners import ZSSRRunner
from eval.pipeline import SRPipeline

import config


NUM_PATCHES = 64
BATCH_SIZE = 1
SCALE_FACTOR = 4
N_SCALE_FACTORS = 3
N_EPOCHS = 20

if __name__ == "__main__":

    zssr_runner = ZSSRRunner()
    pipeline = SRPipeline(
        runner=zssr_runner,
        dataset_zip_path=config.URBAN100_ZIP,
        datasets_dir=config.DATASET_DIR,
        output_dir=config.ZSSR_OUTPUT_DIR + config.URBAN100_NAME,
        scale_factor=4.0
    )

    pipeline.run(n_epochs=1)