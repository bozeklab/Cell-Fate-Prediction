import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

TITLE_FONT = dict(size=36, family="Times New Roman", color="black")
TICK_FONT = dict(size=30, family="Times New Roman", color="black")
LEGEND_FONT = dict(size=22, family="Times New Roman", color="black")
ANNOTATION_FONT_SIZE = 22

REGIONS = [
    (-1.0, -0.474, "red", "Large Negative"),
    (-0.474, -0.33, "orange", "Medium Negative"),
    (-0.33, -0.147, "yellow", "Small Negative"),
    (-0.147, 0.147, "lightgray", "Negligible"),
    (0.147, 0.33, "lightgreen", "Small Positive"),
    (0.33, 0.474, "green", "Medium Positive"),
    (0.474, 1.0, "darkgreen", "Large Positive"),
]

SCALE = False

states = ["Death", "Division"]
features = [
    "area",
    "circularity",
    "eccentricity",
    "equivalent_diameter_area",
    "perimeter",
    "solidity",
    "circadian",
    "p53",
    "cell_cycle",
]

feature_states = {
    "Division": [
        "p53",
        "area",
        "equivalent_diameter_area",
        "circularity",
        "perimeter",
        "solidity",
        "cell_cycle",
        "circadian",
        "eccentricity",
    ],
    "Death": [
        "p53",
        "perimeter",
        "equivalent_diameter_area",
        "area",
        "circadian",
        "cell_cycle",
        "solidity",
        "eccentricity",
        "circularity",
    ],
}

dosage_groups = ["Low", "Medium", "High"]


def bootstrap_ci_median(data, n_boot=2000, alpha=0.05):
    rng = np.random.default_rng()
    n = len(data)
    boot_medians = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_medians[i] = np.median(sample)

    lower = np.percentile(boot_medians, 100 * alpha / 2)
    upper = np.percentile(boot_medians, 100 * (1 - alpha / 2))

    return lower, upper


def signed_log(x, c=0.05):
    return np.sign(x) * np.log10(1 + np.abs(x) / c)


def ypaper_range(fig, y_internal_vals):
    """
    Returns (y0, y1) in paper coordinates for a set of categorical y-values
    """

    cats = fig.layout.yaxis.categoryarray

    # Convert to list safely
    if isinstance(cats, np.ndarray):
        cats = cats.tolist()
    else:
        cats = list(cats)

    idx = [cats.index(y) for y in y_internal_vals]
    n = len(cats)

    y0 = 1 - (max(idx) + 0.5) / n
    y1 = 1 - (min(idx) + 0.5) / n

    return y0, y1


def add_super_category_annotation(fig, df, super_cat, x_pos=-1.15):
    """
    Adds a 90° rotated multirow-style annotation for a super-category
    """
    sub = df[df["State"] == super_cat]
    labels = sub["y_internal"].tolist()

    y_mid = labels[len(labels) // 2]
    print(y_mid)

    fig.add_annotation(
        x=x_pos,
        y=y_mid,
        text=f"<b>{super_cat}</b>",
        showarrow=False,
        textangle=-90,
        xref="paper",
        yref="y",
        align="center",
        valign="middle",
        font=dict(size=38, family="Times New Roman", color="black"),
        # bgcolor="rgba(240,240,240,0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )


rows = []

for state in states:
    for feature in feature_states[state]:
        filename = f"{state}_{feature}.csv"

        state_name = "Apoptosis" if state == "Death" else "Mitosis"

        data = pd.read_csv(os.path.join("src/data/analysis/statistics",filename), delimiter=";", header=0)

        data = data[(data["Dosage"] != "Control") & (data["Dosage"] != "Unknown")]

        deltas = data["Cliff's Delta"].values

        data_thresholded = data[data["p-value"] < 0.05]

        n = len(deltas)
        if n == 0:
            continue

        deltas_thresholded = data_thresholded["Cliff's Delta"].values

        n_pos = np.sum(deltas > 0)

        median_delta = np.median(deltas)
        median_delta_thresholded = np.median(deltas_thresholded)
        mean_delta_thresholded = np.mean(deltas_thresholded)
        q1, q3 = np.percentile(deltas, [25, 75])

        ci_low, ci_high = bootstrap_ci_median(deltas)

        # frac_pos as percentage
        frac_pos_percent = (n_pos / n) * 100

        rows.append(
            {
                "State": state_name,
                "Feature": feature,
                "median_delta": median_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

df = pd.DataFrame(rows)


df["y_label"] = df["Feature"]
df["y_internal"] = df["State"] + " - " + df["Feature"]


c = 0.3  # controls how strong the compression is
df["median_scaled"] = np.sign(df["median_delta"]) * np.log10(
    1 + np.abs(df["median_delta"]) / c
)
df["ci_low_scaled"] = np.sign(df["ci_low"]) * np.log10(1 + np.abs(df["ci_low"]) / c)
df["ci_high_scaled"] = np.sign(df["ci_high"]) * np.log10(1 + np.abs(df["ci_high"]) / c)

if SCALE:
    median = df["median_scaled"]
    ci_low = df["ci_low_scaled"]
    ci_high = df["ci_high_scaled"]
else:
    median = df["median_delta"]
    ci_low = df["ci_low"]
    ci_high = df["ci_high"]

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=median,
        y=df["y_internal"],
        mode="markers",
        marker=dict(size=15, color="black"),
        error_x=dict(
            type="data",
            symmetric=False,
            array=ci_high - median,
            arrayminus=median - ci_low,
            thickness=4,
            width=10,
            color="blue",
        ),
        name="Median ± 95% CI",
        showlegend=False,
    )
)

for start, end, color, label in REGIONS:
    x0_scaled = signed_log(start, c)
    x1_scaled = signed_log(end, c)

    if SCALE:
        start = x0_scaled
        end = x1_scaled

    fig.add_shape(
        type="rect",
        x0=start,
        x1=end,
        y0=-0.5,
        y1=18,
        fillcolor=color,
        opacity=0.35,
        layer="below",
        line_width=0,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],  # invisible
            mode="markers",
            marker=dict(size=10, color=color),
            name=label,
            showlegend=True,
        )
    )

for y in df["y_internal"]:
    fig.add_shape(
        type="line",
        x0=-1,
        x1=1,
        y0=y,
        y1=y,
        xref="x",
        yref="y",
        line=dict(
            color="gray",
            width=2,
        ),
        layer="below",
    )


fig.add_shape(
    type="line",
    xref="paper",
    yref="y",
    x0=1,
    x1=0.025,
    y0=8.5,
    y1=8.5,
    line=dict(color="black", width=3),
    layer="above",
)

fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="gray")

for state in df["State"].unique():
    add_super_category_annotation(fig, df, state, 0)  # -0.22

tick_vals = np.array([-1.0, -0.5, 0, 0.5, 1.0])
tick_vals_scaled = signed_log(tick_vals, c)
tick_text = ["-1.0", "-0.5", "0", "0.5", "1.0"]


if SCALE:
    x_min, x_max = signed_log(-1, c), signed_log(1, c)
    tick_vals = tick_vals_scaled
else:
    x_min, x_max = -1.0, 1.0

fig.update_layout(
    xaxis=dict(
        title="Cliff's delta",
        range=[x_min, x_max],
        zeroline=False,
        title_font=TITLE_FONT,
        tickfont=TICK_FONT,
        domain=[0.2, 1.0],
        tickvals=tick_vals,
        ticktext=tick_text,
    ),
    yaxis=dict(
        title="",
        automargin=True,
        range=[-0.5, len(df) - 0.5],
        title_font=TITLE_FONT,
        tickfont=TICK_FONT,
        categoryorder="array",
        categoryarray=df["y_internal"],
        tickvals=df["y_internal"],
        ticktext=df["y_label"],
    ),
    legend=dict(font=dict(size=34, family="Times New Roman", color="black")),
    template="simple_white",
    margin=dict(r=10, t=10, b=10, l=10),
    height=800,
    width=2600,
)


fig.update_xaxes(title_font_size=36, tickfont_size=30)
fig.update_yaxes(title_font_size=36, tickfont_size=30)

fig.write_image("src/data/analysis/features_result.eps")
