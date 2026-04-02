import argparse
import os
from datetime import datetime

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

import utils.factory as Factory


def train(name, logging=False):
    # configs
    training_config = Factory.TrainingConfig("src/configs", name)
    backbone_config = Factory.BackboneConfig("src/configs")
    transformer_config = Factory.TransformerConfig("src/configs")

    config = {
        "training_config": training_config,
        "backbone_config": backbone_config,
        "transformer_config": transformer_config,
    }

    # logger
    logger = None
    if logging:
        wandb_logger = WandbLogger(
            project="cell_fate_prediction",
            name=f"{name}_{datetime.now()}",  # , log_model="all"
        )

        wandb_logger.log_hyperparams(config)
        logger = [wandb_logger]

    os.makedirs(training_config.folder.output_training, exist_ok=True)

    callbacks = Factory.create_callbacks(training_config)

    data_loader = Factory.create_dataloader(training_config)

    model = Factory.create_model(transformer_config, backbone_config, training_config)

    print(model)

    experiment_path = os.path.join(training_config.folder.output_dir, name)
    trainer = Factory.create_trainer(
        training_config, callbacks, experiment_path, logger
    )

    # save configs
    training_config.save_config(training_config.folder.output_training)
    backbone_config.save_config(training_config.folder.output_training)
    transformer_config.save_config(training_config.folder.output_training)

    print("Starting training at {}".format(datetime.now()))
    trainer.fit(model, data_loader)
    print("Finished training at {}".format(datetime.now()))

    trainer.test(
        model, data_loader, ckpt_path=trainer.checkpoint_callback.best_model_path
    )
    trainer.logger.finalize("success")
    print("Training finished at {}".format(datetime.now()))


if __name__ == "__main__":
    SEED = 42
    seed_everything(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--no-log", action="store_false")

    args = parser.parse_args()

    train(args.name, args.no_log)
