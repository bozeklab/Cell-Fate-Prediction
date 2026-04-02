import argparse
import os

import pandas as pd
import torch
from lightning.pytorch import seed_everything

import utils.factory as Factory


def test(
    model_path,
    model_checkpoint,
    truncation=None,
    use_truncation=False,
    experiment_path="overall",
    debug=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(model_path, "configs")
    training_config = Factory.TrainingConfig(config_path)
    backbone_config = Factory.BackboneConfig(config_path)
    transformer_config = Factory.TransformerConfig(config_path)

    # dataloader
    data_loader = Factory.create_dataloader(
        training_config,
        truncation=truncation,
        use_truncation=use_truncation,
        debug=debug,
    )

    # model
    model = Factory.create_model(transformer_config, backbone_config, training_config)
    state_dict = Factory.load_checkpoint(model_checkpoint, device)

    model.load_state_dict(state_dict, device)

    # callbacks, only used to get best model path
    callbacks = Factory.create_callbacks(training_config)

    # trainer
    trainer = Factory.create_trainer(
        training_config, callbacks, experiment_path=experiment_path, logger=None
    )

    trainer.test(model, data_loader)

    logged = trainer.logged_metrics

    result = {
        "Truncation": truncation,
        "F1": logged["Test/F1score"].item(),
        "Accuracy": logged["Test/Accuracy"].item(),
        "Precision_macro": logged["Test/Precision"].item(),
        "Recall_macro": logged["Test/Recall"].item(),
        "Loss": logged["Test/Loss"].item(),
    }

    for i in range(2):
        label = "Apoptosis" if i == 0 else "Mitosis"
        result[f"Precision_class_{label}"] = logged[
            f"Test/Precision_per_class_{label}"
        ].item()
        result[f"Recall_class_{label}"] = logged[
            f"Test/Recall_per_class_{label}"
        ].item()

    return result


if __name__ == "__main__":
    SEED = 42
    seed_everything(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="src/output/best")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="src/output/best/training/checkpoints/model_checkpoint.safetensors",
    )
    parser.add_argument("--output_dir", type=str, default="src/output/best/evaluation")
    parser.add_argument("--experiment_name", type=str, default="overall")
    parser.add_argument("--truncation", type=int, default=0)
    parser.add_argument("--use_truncation", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    results = test(
        args.model_path,
        args.model_checkpoint,
        truncation=args.truncation,
        use_truncation=args.use_truncation,
        experiment_path=os.path.join(args.output_dir, args.experiment_name),
        debug=args.debug,
    )

    # log experiment
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    results = pd.DataFrame([results])
    results.to_csv(os.path.join(experiment_dir, "results.csv"), index=False, sep=";")
