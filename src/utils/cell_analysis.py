import logging

import cv2
import numpy as np
from cellpose import models
from skimage.measure import label, regionprops

MODEL = models.CellposeModel(gpu=True, model_type="cyto3")
DIAMETERS = [55, 50, 45, 30]
CHANNELS = [0, 1, 2, 3]
VERBOSE = False

if not VERBOSE:
    logging.getLogger("cellpose").setLevel(logging.ERROR)


def analyse_cell(sequence, counter=None):
    seq = normalize_image(sequence)

    cell_data = CellData()

    for frame in seq:
        found = False
        for channel in CHANNELS:
            for diameter in DIAMETERS:
                mask = get_mask(frame, diameter, channel)
                mask, prop, n_neighbors = get_region(mask)
                if prop:
                    found = True
                    break
            if found:
                break
        if not found:
            cell_data.add_data([None] * 8)
            if VERBOSE:
                print("No cell detected in frame: ", counter)

            continue

        cropped_frame = get_masked_image(frame, mask)
        cell_information = get_cell_information(prop, cropped_frame, mask, n_neighbors)

        cell_data.add_data(cell_information)

    return cell_data


def get_mask(frame, diameter, channel=0):
    mask, _, _ = MODEL.eval(frame, diameter=diameter, channels=[channel, 0])
    mask = label(mask, connectivity=2)
    return mask


def get_region(mask):
    if mask.max() == 0:
        return False, False, False
    elif mask.max() == 1:
        return mask, regionprops(mask)[0], 0
    elif mask.max() > 1:
        props = regionprops(mask)
        mask, props = filter_mask(mask, props)

        if len(props) == 0:
            return False, False, False
        elif len(props) == 1:
            return mask, props[0], 0
        elif len(props) > 1:
            mask, prop = get_best_region(props, mask)
            return mask, prop, len(props) - 1


def filter_mask(mask, props, min_size=20):
    filtered_mask = np.zeros_like(mask)
    filtered_props = []

    for region in props:
        if region.area >= min_size:
            filtered_mask[mask == region.label] = region.label
            filtered_props.append(region)
    return filtered_mask, filtered_props


def get_masked_image(frame, mask):
    masked_image = np.zeros_like(frame)
    masked_image[mask == 1] = frame[mask == 1]
    return masked_image


def normalize_image(sequence):
    seq = sequence.squeeze(0)
    return cv2.normalize(
        np.transpose(seq.numpy(), (0, 2, 3, 1)), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def get_cell_information(prop, cropped_frame, mask, n_neighbors):
    area = prop.area
    perimeter = prop.perimeter
    eccentricity = prop.eccentricity
    equivalent_diameter_area = prop.equivalent_diameter_area
    solidity = prop.solidity

    circularity = 4 * np.pi * area / (perimeter**2)

    # calculate mean intensity per channel
    mean_intensity = np.mean(cropped_frame[mask == 1], axis=0)

    return [
        area,
        perimeter,
        eccentricity,
        equivalent_diameter_area,
        solidity,
        circularity,
        mean_intensity,
        n_neighbors,
    ]


def get_best_region(props, cell_mask):
    # print("Multiple possible cells detected")
    image_center = np.array(cell_mask.shape) / 2
    img_diag = np.linalg.norm(image_center)

    max_area = max(r.area for r in props)

    best_score = -np.inf
    best_prop = None

    for prop in props:
        centroid = np.array(prop.centroid)
        dist = np.linalg.norm(centroid - image_center)
        dist_score = 1 - (dist / img_diag)

        area_score = prop.area / max_area if max_area > 0 else 0

        score = dist_score + area_score

        if score > best_score:
            best_score = score
            best_prop = prop

    best_mask = (cell_mask == best_prop.label).astype(np.uint8)
    return best_mask, best_prop


def save_as_image(inp):
    cv2.imwrite("output.png", inp)


class CellData:
    def __init__(self):
        self.area = []
        self.perimeter = []
        self.eccentricity = []
        self.equivalent_diameter_area = []
        self.solidity = []
        self.circularity = []
        self.mean_intensity = []
        self.n_neighbors = []

    def add_data(self, data):
        self.area.append(data[0].item() if data[0] is not None else None)
        self.perimeter.append(data[1].item() if data[1] is not None else None)
        self.eccentricity.append(data[2])
        self.equivalent_diameter_area.append(
            data[3].item() if data[3] is not None else None
        )
        self.solidity.append(data[4].item() if data[4] is not None else None)
        self.circularity.append(data[5].item() if data[5] is not None else None)
        self.mean_intensity.append(
            data[6].tolist() if data[6] is not None else [None] * 3
        )
        self.n_neighbors.append(data[7])

    def get_data(self):
        return {
            "area": self.area,
            "perimeter": self.perimeter,
            "eccentricity": self.eccentricity,
            "equivalent_diameter_area": self.equivalent_diameter_area,
            "solidity": self.solidity,
            "circularity": self.circularity,
            "mean_intensity": self.mean_intensity,
            "n_neighbors": self.n_neighbors,
        }

    def tolist(self):
        return [
            self.area,
            self.circularity,
            self.eccentricity,
            self.equivalent_diameter_area,
            self.perimeter,
            self.solidity,
            self.mean_intensity,
            self.n_neighbors,
        ]
