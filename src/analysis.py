import argparse
import csv
import os

import torch

import utils.factory as Factory
from utils.cell_analysis import analyse_cell
from utils.dose_mapping import DOSE_LOOKUP


def analysis(
    model_path, model_checkpoint, dataset_path, cell_analysis=False, truncation=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(model_path, "configs")
    training_config = Factory.TrainingConfig(config_path)
    backbone_config = Factory.BackboneConfig(config_path)
    transformer_config = Factory.TransformerConfig(config_path)

    training_config.folder.data_dir = dataset_path

    # dataloader
    data_loader = Factory.create_dataloader(
        training_config, truncation=truncation, debug=False
    )

    # model
    model = Factory.create_model(
        transformer_config,
        backbone_config,
        training_config,
        save_attention_weights=True,
    )

    state_dict = Factory.load_checkpoint(model_checkpoint, device)

    model.load_state_dict(state_dict, device)
    model.to(device)
    model.eval()

    data_loader.setup("analysis")
    dataloader = data_loader.analysis_dataloader()

    output_file = "cell_analysis"

    if truncation is not None:
        output_file += f"_trunc_{truncation}.csv"
    else:
        output_file += ".csv"

    columns = ["label", "output_label", "time_step", "dosage", "attention_weights"]
    if cell_analysis:
        output_dir = "src/data/analysis"

        columns += [
            "area",
            "circularity",
            "eccentricity",
            "equivalent_diameter_area",
            "perimeter",
            "solidity",
            "mean_intensity",
            "n_neighbors",
        ]

    with open(os.path.join(output_dir, output_file), mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(columns)

    buffer = []
    n_steps = 200
    counter = 0

    with open(os.path.join(output_dir, output_file), "a", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        with torch.no_grad():
            for batch in dataloader:
                sequence, mask, label, time_step, video_name = batch

                dosage = DOSE_LOOKUP[int(video_name[0])]

                if cell_analysis:
                    cell_data = analyse_cell(sequence, counter)

                output = model(sequence.to(device), mask.to(device))

                output_label = int((torch.sigmoid(output) >= 0.5).float().item())
                attention_weights = model.get_attention_weights()

                output = [
                    label.item(),
                    output_label,
                    time_step[0],
                    dosage,
                    attention_weights.cpu().tolist(),
                ]

                if cell_analysis:
                    output += cell_data.tolist()

                buffer.append(output)

                if len(buffer) >= n_steps:
                    print("Writing buffer to file at step: ", counter)
                    writer.writerows(buffer)
                    f.flush()
                    buffer.clear()

                counter += 1

            if buffer:
                writer.writerows(buffer)
                f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="src/output/best",
        help="Path to the model directory containing configs and checkpoints.",
    )

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="src/output/best/training/checkpoints/model_checkpoint.safetensors",
        help="Path to the model checkpoint file.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory containing the sequences.",
    )

    parser.add_argument(
        "--cell_analysis",
        action="store_true",
        help="Whether to perform cell analysis and include it in the output CSV.",
    )

    parser.add_argument(
        "--truncation",
        type=int,
        default=None,
        help="Number of frames to truncate the sequences to. If not set, uses full sequences.",
    )

    args = parser.parse_args()

    print(f"Running analysis with truncation: {args.truncation}")

    analysis(
        args.model_path,
        args.model_checkpoint,
        args.dataset_path,
        args.cell_analysis,
        args.truncation,
    )
