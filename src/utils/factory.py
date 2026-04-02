import os

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from safetensors.torch import load_file

from dataset.dataloader import CocoVidDataLoader
from models.backbone import ResNet
from models.transformer import VideoModel


class BaseConfig:
    def __init__(self, config_path, config_name):
        self.config_name = config_name
        with open(os.path.join(config_path, config_name), "r") as f:
            config = yaml.safe_load(f)

        self._dict_to_attr(config)

    def _dict_to_attr(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries to BaseConfig objects
                setattr(self, key, BaseConfig.from_dict(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, dictionary):
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance._dict_to_attr(dictionary)
        return instance

    def __repr__(self):
        return f"{self.__dict__}"

    def to_dict(self):
        """Recursively convert Config object to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, BaseConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def save_config(self, output_dir):
        """Save Config object to a yaml file"""

        output_dir = os.path.join(output_dir.split("training")[0], "configs")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, self.config_name), "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class TrainingConfig(BaseConfig):
    def __init__(self, config_path, model_name=None):
        super().__init__(config_path, "training.yml")

        if "output_training" not in self.folder.to_dict():
            self.folder.output_training = os.path.join(
                self.folder.output_dir, model_name, "training"
            )


class BackboneConfig(BaseConfig):
    def __init__(self, config_path):
        super().__init__(config_path, "backbone.yml")


class TransformerConfig(BaseConfig):
    def __init__(self, config_path):
        super().__init__(config_path, "transformer.yml")


def create_backbone(backbone_config):
    """Creates a backbone CNN for feature extraction."""

    model_name = backbone_config.architecture.model_name
    if model_name.lower() == "resnet":
        return ResNet(
            backbone_config.architecture.variant,
            backbone_config.training.pretrained,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def create_model(
    transformer_config, backbone_config, training_config, save_attention_weights=False
):
    """Creates a Transformer model for sequence modeling."""

    backbone = create_backbone(backbone_config)

    return VideoModel(
        backbone=backbone,
        batch_size=training_config.training.batch_size,
        learning_rate=training_config.training.learning_rate,
        num_channels=transformer_config.architecture.num_channels,
        nhead=transformer_config.architecture.nhead,
        nlayers=transformer_config.architecture.nlayers,
        hidden_dim=transformer_config.architecture.hidden_dim,
        dropout=transformer_config.architecture.dropout,
        pos_dropout=transformer_config.architecture.pos_dropout,
        save_attention_weights=save_attention_weights,
        weight_decay=training_config.training.weight_decay,
        max_len=transformer_config.architecture.pos_max_len,
    )


def create_dataloader(
    training_config,
    truncation=None,
    use_truncation=False,
    debug=False,
):
    """Creates a DataLoader for the training dataset."""

    trunc = (
        truncation if truncation is not None else training_config.training.truncation
    )

    return CocoVidDataLoader(
        data_dir=training_config.folder.data_dir,
        batch_size=training_config.training.batch_size,
        num_workers=training_config.training.num_workers if not debug else 0,
        mean=training_config.misc.mean,
        std=training_config.misc.std,
        min_seq_length=training_config.training.min_seq_length,
        truncation=trunc,
        use_truncation=use_truncation,
    )


def create_trainer(training_config, callbacks, experiment_path, logger):
    return Trainer(
        max_epochs=training_config.training.max_epochs,
        default_root_dir=experiment_path,
        accelerator="gpu",
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        benchmark=False,
        enable_model_summary=True,
    )


def create_callbacks(training_config):
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(training_config.folder.output_training, "checkpoints"),
        filename="epoch-{epoch}_val_BAC-{Val/Accuracy:.3f}",
        monitor="Val/Accuracy",
        auto_insert_metric_name=False,
        mode="max",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    if training_config.training.early_stop_patience:
        early_stop_callback = EarlyStopping(
            monitor="Val/Loss",
            patience=training_config.training.early_stop_patience,
        )
        callbacks.append(early_stop_callback)

    return callbacks


def load_checkpoint(checkpoint_path, device):
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path, device="cpu")
    elif checkpoint_path.endswith(".ckpt"):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        else:
            raise ValueError(
                f"Checkpoint file {checkpoint_path} does not contain 'state_dict' key."
            )
    return state_dict
