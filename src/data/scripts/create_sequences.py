import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torchvision.transforms.functional import crop
from tqdm import tqdm

sys.path.append(sys.path[0] + "/../..")
from utils.parser import CocoVidParser
from utils.transforms import Rescale


def load_image(path):
    image = cv2.imread(path, -1).astype(np.float32)
    image = image[:, :, ::-1].copy()  # BGR to RGB
    image = T.ToTensor()(image)
    return image


def write_csv(data, seq_lengths, labels, name, output_dir):
    combined_data = [x + [y] for x, y in zip(data, labels)]
    with open(os.path.join(output_dir, f"{name}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "cell", "sequence_length", "label"])
        for entry in combined_data:
            writer.writerow(
                [entry[0], entry[1], seq_lengths[entry[0]][entry[1]], entry[2]]
            )


def split_dataset(videos, coco):
    cell_list = []
    labels = []

    for vid_id, vid_name in videos:
        cells = coco.get_labels_from_vid_id(vid_id)

        for cell, anns in cells.items():
            annotation = coco.anns[anns[0]]
            if annotation["time_of_death"] != None:
                label = 0
            elif annotation["time_of_division"] != None:
                label = 1
            else:
                continue

            cell_list.append([vid_name, cell])
            labels.append(label)

    (
        train_cells,
        tmp_cells,
        train_labels,
        tmp_labels,
    ) = train_test_split(
        cell_list,
        labels,
        test_size=VALID_RATIO + TEST_RATIO,
        random_state=42,
        stratify=labels,
    )

    (
        val_cells,
        test_cells,
        val_labels,
        test_labels,
    ) = train_test_split(
        tmp_cells,
        tmp_labels,
        test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),
        random_state=42,
        stratify=tmp_labels,
    )
    return train_cells, val_cells, test_cells, train_labels, val_labels, test_labels


def save_sequences(videos, coco, patch_size, train_cells, data_dir, output_dir):
    h_ = patch_size[0] // 2
    w_ = patch_size[1] // 2

    channel_sums = torch.zeros(3)
    channel_sums_squared = torch.zeros(3)
    total_pixels = 0

    seq_length_dict = {}

    for vid_id, vid_name in tqdm(videos):
        cells = coco.get_labels_from_vid_id(vid_id)
        video_name = coco.load_vids(vid_id)[0]["name"]
        output_folder = os.path.join(output_dir, video_name)
        os.makedirs(output_folder, exist_ok=True)

        seq_length_dict[vid_name] = {}

        for cell, anns in cells.items():
            frames = []

            # check, if the cell divides or dies at some point
            tmp_ann = coco.anns[anns[0]]

            if tmp_ann["time_of_death"] != None:
                label = 0
            elif tmp_ann["time_of_division"] != None:
                label = 1
            else:
                continue

            # create numpy array of the sequence of the cell
            raw_annotations = [
                coco.anns[ann_id]
                for ann_id in sorted(anns)
                if coco.anns[ann_id]["category_id"] == 1
            ]

            for annotation in raw_annotations:
                # load image
                img_path = coco.loadImgs(annotation["image_id"])[0]["file_name"]

                img_path = os.path.join(data_dir, img_path)

                image = load_image(img_path)

                bounding_box = annotation["bbox"]
                bounding_box = [x * 1024 for x in bounding_box]

                centroid = bounding_box[:2]

                top = int(centroid[1] - h_)
                left = int(centroid[0] - w_)

                patch = crop(
                    image,
                    top,
                    left,
                    patch_size[0],
                    patch_size[1],
                )
                frames.append(patch)

            seq_length_dict[vid_name][cell] = len(frames)

            video = torch.stack(frames)

            # calc mean
            rescaled_video = Rescale()(video)

            if [vid_name, cell] in train_cells:
                channel_sums = rescaled_video.sum(dim=[0, 2, 3])
                channel_sums_squared = torch.square(rescaled_video.sum(dim=[0, 2, 3]))
                total_pixels += (
                    rescaled_video.size(0)
                    * rescaled_video.size(2)
                    * rescaled_video.size(3)
                )

            torch.save(
                [
                    vid_id,
                    video_name,
                    cell,
                    label,
                    tmp_ann["time_step"],
                    rescaled_video,
                ],
                os.path.join(output_folder, f"{cell}.pt"),
            )

    mean = channel_sums / total_pixels
    std = torch.sqrt((channel_sums_squared / total_pixels) - torch.square(mean))

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    with open(os.path.join(output_dir, "mean_std.txt"), "w") as f:
        f.write(f"Mean: {mean}\nStd: {std}")

    return seq_length_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the raw data.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the output.",
    )

    parser.add_argument(
        "--ann_file", type=str, required=True, help="Path to the annotation file."
    )

    parser.add_argument(
        "--patch_size", type=int, default=112, help="Size of the output patches."
    )

    args = parser.parse_args()

    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.1
    TEST_RATIO = 0.1

    ann_file = os.path.join(args.data_dir, args.ann_file)

    coco = CocoVidParser(ann_file)

    videos = [[vid["id"], vid["name"]] for vid in coco.dataset["videos"]]

    patch_size = (args.patch_size, args.patch_size)

    train_cells, val_cells, test_cells, train_labels, val_labels, test_labels = (
        split_dataset(videos, coco)
    )

    seq_length_dict = save_sequences(
        videos=videos,
        coco=coco,
        patch_size=patch_size,
        train_cells=train_cells,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    # save split to csv
    write_csv(
        train_cells,
        seq_length_dict,
        train_labels,
        "train_cells",
        output_dir=args.output_dir,
    )
    write_csv(
        val_cells, seq_length_dict, val_labels, "val_cells", output_dir=args.output_dir
    )
    write_csv(
        test_cells,
        seq_length_dict,
        test_labels,
        "test_cells",
        output_dir=args.output_dir,
    )
