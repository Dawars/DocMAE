#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import argparse
import torch
import datasets
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
    ViTMAEConfig,
    ViTMAEForPreTraining,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from docmae import setup_logging

""" Pre-training a 🤗 ViT model as an MAE (masked autoencoder), as proposed in https://arxiv.org/abs/2111.06377."""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.29.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    image_column_name: Optional[str] = field(default=None, metadata={"help": "The column name of the images in the files."})
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this " "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        data_files = {}
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": ("The model checkpoint for weights initialization. "
                     "Don't set if you want to train a model from scratch.")
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    mask_ratio: float = field(
        default=0.75, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    norm_pix_loss: bool = field(
        default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=1e-3, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="config file for training parameters")
    parser.add_argument("-ll", "--log-level", type=str, default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        help="config file for training parameters")
    parser.add_argument("-l", "--log-dir", type=str, default="",
                        help="folder to store log files")
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="",
                        help="folder to store tensorboard logs")
    parser.add_argument("-m", "--model-output-dir", type=str, default="model",
                        help="folder to store trained models")
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging(log_level=args.log_level, log_dir=args.log_dir)

    assert args.config.endswith(".json")

    # Save config for training traceability and load config parameters
    config_file = Path(args.model_output_dir) / "config.json"
    config = json.loads(Path(args.config).read_text())

    config["logging_dir"] = args.tensorboard_dir
    config["output_dir"] = args.model_output_dir

    config_file.write_text(json.dumps(config))
    train(config_file)


def train(config_file: str):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=config_file)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_mae", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:  # ignore config.json which is always there
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset.
    ds_train = datasets.Dataset.from_dict(
        {"image": sorted(glob.glob(os.path.join(data_args.data_files["train"], "*.jpg")))},
        split=datasets.Split.TRAIN,
    ).cast_column("image", datasets.Image(decode=True))

    ds_val = datasets.Dataset.from_dict(
        {"image": sorted(glob.glob(os.path.join(data_args.data_files["val"], "*.jpg")))},
        split=datasets.Split.VALIDATION,
    ).cast_column("image", datasets.Image(decode=True))

    # combine dataset splits
    ds = datasets.DatasetDict()
    ds["train"] = ds_train
    ds["validation"] = ds_val

    # Load pretrained model and image processor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "size": {"height": 288, "width": 288},
    }
    if model_args.config_name:
        config = ViTMAEConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ViTMAEConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = ViTMAEConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
        }
    )

    # create image processor
    if model_args.image_processor_name:
        image_processor = ViTImageProcessor.from_pretrained(model_args.image_processor_name, **config_kwargs)
    elif model_args.model_name_or_path:
        image_processor = ViTImageProcessor.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        image_processor = ViTImageProcessor()

    # create model
    if model_args.model_name_or_path:
        model = ViTMAEForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = ViTMAEForPreTraining(config)

    if training_args.do_train:
        column_names = ds_train.column_names
    else:
        column_names = ds_val.column_names

    if data_args.image_column_name is not None:
        image_column_name = data_args.image_column_name
    elif "image" in column_names:
        image_column_name = "image"
    elif "img" in column_names:
        image_column_name = "img"
    else:
        image_column_name = column_names[0]

    # transformations as done in original MAE paper
    # source: https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    transforms = Compose(
        [
            Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            # RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            # RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms."""

        examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
        return examples

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Set the training transforms
        ds["train"].set_transform(preprocess_images)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
        # Set the validation transforms
        ds["validation"].set_transform(preprocess_images)

    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "tasks": "masked-auto-encoding",
        "dataset": data_args.dataset_name,
        "tags": ["masked-auto-encoding"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
