from train_config import get_config
from pathlib import Path
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
import logging
from accelerate.logging import get_logger
import transformers
import diffusers
import os


def main():
    train_config = get_config("lcm_config.json")
    
    logging_dir = Path(train_config.output_dir, train_config.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=logging_dir.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
 gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        #log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True, 
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=False)
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
    set_seed(train_config.seed)
    os.makedirs(train_config.output_dir, exist_ok=True)

