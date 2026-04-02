import ast
import multiprocessing as mp
import os
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import permutation_test as scipy_permutation_test
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

TRUNCATE = False
TRUNCATION = 86

CELL_FEATURES = [
    "area",
    "circularity",
    "eccentricity",
    "equivalent_diameter_area",
    "perimeter",
    "solidity",
    "mean_intensity",
    "n_neighbors",
]
INTERPRETATION = {
    "Negligible": 0.147,
    "Small": 0.33,
    "Medium": 0.474,
    "Large": float("inf"),
}

dose_map = {"Control": 1, "High": 2, "Medium": 3, "Low": 4, "Unknown": 5}


def read_cell_analysis(file_path, left_padding=True):
    file = pd.read_csv(file_path, delimiter=";", header=0)
    file["attention_weights"] = file["attention_weights"].apply(
        lambda x: save_literal_eval(x)
    )

    file = file.map(lambda x: save_literal_eval(x) if isinstance(x, str) else x)

    max_len = 0

    for column in CELL_FEATURES:
        file, curr_max_len = pad_column(file, left_padding=left_padding, column=column)
        if curr_max_len > max_len:
            max_len = curr_max_len

    return file, max_len


def read_analysis_results(file_path):
    file = pd.read_csv(file_path, delimiter=";", header=0)
    # file["attention_weights"] = file["attention_weights"].apply(
    #     lambda x: ast.literal_eval(x) if pd.notna(x) else np.nan
    # )
    file["attention_weights"] = file["attention_weights"].apply(
        lambda x: save_literal_eval(x)
    )

    if "time_step" in file.columns:
        if not pd.api.types.is_integer_dtype(file["time_step"]):
            file["time_step"] = file["time_step"].apply(lambda x: ast.literal_eval(x))
        if isinstance(file["time_step"][0], list):
            file["time_step"] = file["time_step"].apply(lambda x: x[0])

    return file


# def read_analysis_results(file_path):
#     file = pd.read_csv(file_path, delimiter=";", header=0)
#     file["attention_weights"] = file["attention_weights"].apply(
#         lambda x: ast.literal_eval(x)
#     )

#     return file


def get_ind_above_quantile(data, p=0.9):
    quantiles = np.nanpercentile(data, p * 100, axis=1, keepdims=True)

    mask = data >= quantiles

    return mask


def split_channels(mean_intensity):
    mean_r = [frame[0] for frame in mean_intensity]
    mean_g = [frame[1] for frame in mean_intensity]
    mean_b = [frame[2] for frame in mean_intensity]

    return pd.Series([mean_r, mean_g, mean_b], index=["mean_r", "mean_g", "mean_b"])


def pad_column(data, left_padding=True, column="attention_weights", max_len=None):
    max_len = data[column].apply(len).max() if max_len is None else max_len
    if (
        isinstance(data[column][0][0], float)
        or isinstance(data[column][0][0], int)
        or data[column][0][0] is None
    ):
        num_entries = 1
    else:
        num_entries = max(len(x) for x in data[column][0])

    copy_df = data.copy()
    copy_df[column] = copy_df[column].apply(
        lambda x: pad_sequence(x, max_len, num_entries, left_padding)
    )

    return copy_df, max_len


def pad_sequence(seq, max_len, num_entries, left_padding=True):
    num_missing = max_len - len(seq)
    if num_entries == 1:
        padding = [0.0] * num_missing
    elif num_entries == 3:
        padding = [[0.0] * num_entries] * num_missing

    if left_padding:
        return padding + seq
    else:
        return seq + padding


def save_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val.replace("nan", "None"))
        except (ValueError, SyntaxError):
            return val
    return val


def get_feature(data, feature_name, label=None, truncate=False):
    if label is not None:
        data = data[data["label"] == label]

    output = np.array(data[feature_name].values.tolist())
    doses = data["dosage"].values
    doses = np.array([dose_map[x] for x in doses]).reshape(-1, 1)

    output = np.concatenate([output, doses], axis=1)

    if truncate:
        output = output[:, :, -TRUNCATION:]

    return output


def substitute_vals(data, sub):
    try:
        mask_nonzero = data != 0
        first_nz = mask_nonzero.argmax(axis=1)
        last_nz = data.shape[1] - 1 - mask_nonzero[:, ::-1].argmax(axis=1)

        cols = np.arange(data.shape[1])

        replace_mask = (data == 0) & (
            (cols < first_nz[:, None]) | (cols > last_nz[:, None])
        )

        data[replace_mask] = sub
        # data[data == 0] = sub

    except ValueError:
        stop = 0
    if type(data[0][0]) is np.ndarray and data[0][0].shape[0] > 1:
        data = np.array(
            [
                [[sub if v is None else v for v in channels] for channels in row]
                for row in data
            ]
        )
    else:
        data = np.array([[sub if v is None else v for v in row] for row in data])
    return data


def permute_masks_within_sequences(masks):
    permuted = []
    for mask in masks:
        n_top = np.sum(mask)
        perm = np.zeros_like(mask, dtype=bool)
        perm[np.random.choice(len(mask), n_top, replace=False)] = True
        permuted.append(perm)

    return np.array(permuted)


def get_column_aggregation(
    data,
    sub,
    label=None,
    column="attention_weights",
    aggregate="mean",
    normalize=True,
    truncate=False,
):
    if label is not None:
        data = data[data["label"] == label]

    vals = np.vstack(data[column].values)

    if truncate:
        vals = vals[:, -TRUNCATION:]

    if aggregate == "mean":
        vals = substitute_vals(vals, sub)

        vals = np.nanmean(vals, axis=0)
        if np.isnan(sub):
            vals = np.nan_to_num(vals, nan=0.0)

    vals = substitute_vals(vals, sub)

    if normalize:
        if sub is np.nan:
            vals = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
        else:
            vals = (vals - vals.min()) / (vals.max() - vals.min())

    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    return vals


def combine_features(
    dataframe,
    feature_names,
    label=None,
    normalize=False,
    aggregate=True,
    truncate=False,
    sub=0,
):
    feature_list = []
    for feature in feature_names:
        if aggregate:
            feature_data = get_column_aggregation(
                dataframe,
                sub=np.nan,
                label=label,
                column=feature,
                aggregate="mean",
                normalize=False,
                truncate=truncate,
            )
        else:
            feature_data = get_feature(
                dataframe, feature, label=label, truncate=truncate
            )
            feature_data = substitute_vals(feature_data, sub=sub)
            if normalize:
                scaler = StandardScaler()
                feature_data = scaler.fit_transform(feature_data)
        feature_list.append(feature_data)

    combined_features = np.stack(feature_list, axis=0)

    if aggregate:
        combined_features = combined_features.squeeze(axis=1)
        combined_features = combined_features.T

        if normalize:
            scaler = StandardScaler()
            combined_features = scaler.fit_transform(combined_features)
    else:
        combined_features = combined_features.transpose(1, 2, 0)
        frames = combined_features[:, :186, :]
        doses = combined_features[:, 186, 0]
        doses = doses[:, None, None]
        doses = np.repeat(doses, 186, axis=1)

        combined_features = np.concatenate([frames, doses], axis=2)

    return combined_features


def cliffs_delta_perm(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    n_x = len(x)
    n_y = len(y)

    greater = sum(1 for a, b in product(x, y) if a > b)
    less = sum(1 for a, b in product(x, y) if a < b)

    delta = (greater - less) / (n_x * n_y)

    if delta > 1 or delta < -1:
        print("Warning: delta out of bounds:", delta)
        print("greater:", greater, "less:", less, "n_x:", n_x, "n_y:", n_y)
    return delta


def compute_cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n_x, n_y = len(x), len(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))

    d = (mean_x - mean_y) / pooled_std
    return d


def scipy_statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def permutation_test(
    feature, quantile_data, statistic, n_permutations, feature_name, dose
):
    """Permutation test within a single sequence"""
    results = []

    skip_counter = 0

    for row, mask, dose in tqdm(zip(feature, quantile_data, dose)):
        top = row[mask][~np.isnan(row[mask])]
        other = row[~mask][~np.isnan(row[~mask])]

        if len(top) < 2 or len(other) < 2:
            skip_counter += 1
            continue

        observed_stat = np.abs(statistic(top) - statistic(other))
        cliffs_d = cliffs_delta_perm(top, other)
        cohens_d = compute_cohens_d(top, other)

        perm_stats = []
        for _ in range(n_permutations):
            perm_mask = permute_masks_within_sequences([mask])[0]

            grp1 = row[perm_mask][~np.isnan(row[perm_mask])]
            grp2 = row[~perm_mask][~np.isnan(row[~perm_mask])]

            if len(grp1) > 0 and len(grp2) > 0:
                perm_stat = np.abs(statistic(grp1) - statistic(grp2))
                perm_stats.append(perm_stat)

        perm_stats = np.array(perm_stats)

        p_value_scipy = scipy_permutation_test(
            (top, other),
            statistic=scipy_statistic,
            n_resamples=n_permutations,
            alternative="two-sided",
        ).pvalue

        results.append([dose, p_value_scipy, cliffs_d, cohens_d])

    print(f"Skipped {skip_counter} sequences for feature {feature_name}")

    return np.array(results)


def classifiy_effect_size(value, interpretation):
    for label, threshold in interpretation.items():
        if abs(value) < threshold:
            return label


def is_significant(df):
    df["Significance"] = df["Cliff's Delta"].apply(
        lambda x: classifiy_effect_size(x, INTERPRETATION)
    )
    df["Effect"] = df["Cliff's Delta"].apply(
        lambda x: "Top > Other" if x > 0 else "Other > Top"
    )
    return df


def parallel_worker(args):
    (
        function,
        feature,
        state,
        feature_name,
        quantile_data,
        idy,
        n_permutations,
        statistic,
    ) = args

    return (
        state,
        feature_name,
        *function(
            feature, state, feature_name, quantile_data, idy, n_permutations, statistic
        ),
    )


def permutation_test_within_sequence(
    features,
    state,
    feature_name,
    quantile_data,
    feature_idx,
    n_permutations=10000,
    statistic=np.mean,
):
    np.random.seed(42)

    dose = features[:, 0, 10]
    feature = features[:, :, feature_idx]

    results = permutation_test(
        feature, quantile_data, statistic, n_permutations, feature_name, dose
    )

    results = pd.DataFrame(
        results,
        columns=[
            "Dosage",
            "p-value",
            "Cliff's Delta",
            "Cohen's D",
        ],
        index=None,
    )

    reverse_map = {v: k for k, v in dose_map.items()}

    results["Dosage"] = results["Dosage"].map(reverse_map)

    results.to_csv(os.path.join(OUTPUT_DIR, f"{state}_{feature_name}.csv"), sep=";", index=False)

    return (0, 0)


def all_in_all_per_sequence(
    features_die_all,
    features_div_all,
    die_quantile_data,
    div_quantile_data,
    n_permutations=50000,
    debug=False,
):
    tasks = []

    if debug:
        n_permutations = 10

    feats = {
        "Death": (features_die_all, die_quantile_data),
        "Division": (features_div_all, div_quantile_data),
    }

    if debug:
        for state, (feature, quantile_data) in feats.items():
            for idy, feature_name in enumerate(feature_list):
                print(f"Permutation test for {state} - {feature_name}")
                permutation_test_within_sequence(
                    feature,
                    state,
                    feature_name,
                    quantile_data,
                    idy,
                    n_permutations,
                    np.mean,
                )
    else:
        for state, (feature, quantile_data) in feats.items():
            for idy, feature_name in enumerate(feature_list):
                tasks.append(
                    (
                        permutation_test_within_sequence,
                        feature,
                        state,
                        feature_name,
                        quantile_data,
                        idy,
                        n_permutations,
                        np.mean,
                    )
                )

        with mp.Pool(mp.cpu_count()) as pool:
            for output in tqdm(
                pool.imap_unordered(parallel_worker, tasks),
                total=len(tasks),
                desc="Running all tests",
            ):
                pass


if __name__ == "__main__":
    QUANTILE = 0.90
    MODEL = "best"
    LEFT_PADDING = True
    ONLY_CORRECT_PREDICTIONS = True
    MIN_SEQ_LENGTH = 50
    ATTENTION_SUB = np.nan

    # folder creation
    model_postfix = MODEL if MODEL == "best" else "rerun"
    quantile_postfix = str(int(QUANTILE * 100))
    seq_length_postifx = f"_min_seq_{MIN_SEQ_LENGTH}" if MIN_SEQ_LENGTH else ""
    correct_postfix = "_correct" if ONLY_CORRECT_PREDICTIONS else ""
    sub_postfix = "_nanSub" if ATTENTION_SUB is np.nan else ""

    OUTPUT_DIR = "src/data/analysis/statistics"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results, max_len = read_cell_analysis(
        "src/data/analysis/cell_analysis.csv",
        left_padding=LEFT_PADDING,
    )

    results[["circadian", "p53", "cell_cycle"]] = results["mean_intensity"].apply(
        split_channels
    )
    results = results.drop(columns=["mean_intensity"])

    if ONLY_CORRECT_PREDICTIONS:
        results = results[results["label"] == results["output_label"]].reset_index(
            drop=True
        )

    if MIN_SEQ_LENGTH:
        results["sequence_length"] = results["attention_weights"].apply(len)
        results = results[results["sequence_length"] >= MIN_SEQ_LENGTH]
        results = results.drop(columns=["sequence_length"])
        results = results.reset_index(drop=True)

    print(f"Number of sequences: {len(results)}")

    feature_list = results.columns[5:]

    features_die_all = combine_features(
        results,
        feature_list,
        label=0,
        aggregate=False,
        normalize=False,
        truncate=TRUNCATE,
        sub=np.nan,
    )
    features_div_all = combine_features(
        results,
        feature_list,
        label=1,
        aggregate=False,
        normalize=False,
        truncate=TRUNCATE,
        sub=np.nan,
    )

    results, _ = pad_column(results, left_padding=LEFT_PADDING, max_len=max_len)

    attentions_die = get_column_aggregation(
        results,
        label=0,
        sub=ATTENTION_SUB,
        aggregate=False,
        truncate=TRUNCATE,
        normalize=True,
    )

    attentions_div = get_column_aggregation(
        results,
        label=1,
        sub=ATTENTION_SUB,
        aggregate=False,
        truncate=TRUNCATE,
        normalize=True,
    )

    die_quantile_data = get_ind_above_quantile(attentions_die, p=QUANTILE)
    div_quantile_data = get_ind_above_quantile(attentions_div, p=QUANTILE)

    ITERATIONS = 50000

    all_in_all_per_sequence(
        features_die_all,
        features_div_all,
        die_quantile_data,
        div_quantile_data,
        n_permutations=ITERATIONS,
        debug=False,
    )
