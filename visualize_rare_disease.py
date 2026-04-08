"""
Per-class F1 visualization for the rare-disease analysis.

Generates Figure 4 of the paper, comparing the per-class F1 scores of
APA-RRG and the PromptMRG baseline on the MIMIC-CXR test split. The
left panel is a radar chart over the seven low-prevalence pathologies;
the right panel is a horizontal lollipop chart of per-class F1 deltas
sorted by absolute improvement, with categories grouped by anatomical
region.

Usage:
    python visualize_rare_disease.py \
        --baseline results/baseline/model_best_rare_eval.json \
        --ours results/apa_rrg/model_best_rare_eval.json \
        --output fig_rare_disease.pdf
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 10,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "mathtext.default": "regular",
    }
)

DISEASES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

PATHOLOGIES = [d for d in DISEASES if d != "No Finding"]

# Seven low-prevalence pathologies (training-set positive rate <= 5.5%).
RARE_DISEASES = [
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Consolidation",
    "Pneumonia",
    "Pneumothorax",
    "Pleural Other",
    "Fracture",
]

SHORT_NAMES = {
    "Enlarged Cardiomediastinum": "Enl. Card.",
    "Cardiomegaly": "Cardiomegaly",
    "Lung Opacity": "Lung Opacity",
    "Lung Lesion": "Lung Lesion",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Pl. Effusion",
    "Pleural Other": "Pl. Other",
    "Fracture": "Fracture",
    "Support Devices": "Sup. Devices",
}

# Anatomical region mapping used by APG (matches models/apg.py and the
# paper's Figures 4 and 5).
REGION_MAP = {
    "Enlarged Cardiomediastinum": "Cardiac",
    "Cardiomegaly": "Cardiac",
    "Lung Opacity": "Pulmonary",
    "Lung Lesion": "Pulmonary",
    "Edema": "Pulmonary",
    "Consolidation": "Pulmonary",
    "Pneumonia": "Pulmonary",
    "Atelectasis": "Pulmonary",
    "Pneumothorax": "Pulmonary",
    "Pleural Effusion": "Pleural",
    "Pleural Other": "Pleural",
    "Fracture": "Skeletal",
    "Support Devices": "Other",
}

REGION_COLORS = {
    "Cardiac": "#E57373",
    "Pulmonary": "#64B5F6",
    "Pleural": "#81C784",
    "Skeletal": "#FFD54F",
    "Other": "#B0BEC5",
}

REGION_COLORS_DARK = {
    "Cardiac": "#C62828",
    "Pulmonary": "#1565C0",
    "Pleural": "#2E7D32",
    "Skeletal": "#F57F17",
    "Other": "#546E7A",
}


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_combined_figure(baseline, ours, output_path="fig_rare_disease.pdf"):
    fig = plt.figure(figsize=(10.5, 4.8))

    # ----- Left: radar chart over low-prevalence pathologies -----
    ax_radar = fig.add_subplot(121, polar=True)

    rare = [
        d
        for d in PATHOLOGIES
        if d in RARE_DISEASES and d in baseline["per_class"] and d in ours["per_class"]
    ]
    labels_r = [SHORT_NAMES.get(d, d) for d in rare]
    b_f1 = [baseline["per_class"][d]["f1"] for d in rare]
    o_f1 = [ours["per_class"][d]["f1"] for d in rare]

    n = len(rare)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    b_f1_c = b_f1 + [b_f1[0]]
    o_f1_c = o_f1 + [o_f1[0]]
    angles_c = angles + [angles[0]]

    ax_radar.fill(angles_c, b_f1_c, alpha=0.10, color="#78909C")
    ax_radar.plot(
        angles_c, b_f1_c, "o-", linewidth=1.2, markersize=4, color="#78909C", label="PromptMRG"
    )

    ax_radar.fill(angles_c, o_f1_c, alpha=0.18, color="#E53935")
    ax_radar.plot(
        angles_c, o_f1_c, "o-", linewidth=1.5, markersize=5, color="#E53935", label="APA-RRG (Ours)"
    )

    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(labels_r, fontsize=8.5)
    for label, angle in zip(ax_radar.get_xticklabels(), angles):
        if angle == 0:
            label.set_horizontalalignment("center")
            label.set_verticalalignment("bottom")
        elif 0 < angle < np.pi:
            label.set_horizontalalignment("left")
        elif angle == np.pi:
            label.set_horizontalalignment("center")
            label.set_verticalalignment("top")
        else:
            label.set_horizontalalignment("right")

    y_max = max(max(b_f1), max(o_f1)) * 1.3
    ax_radar.set_ylim(0, y_max)
    ax_radar.set_yticks([0.1, 0.2, 0.3])
    ax_radar.set_yticklabels(["0.1", "0.2", "0.3"], fontsize=7, color="#757575")

    ax_radar.set_title("Low-prevalence Diseases", fontsize=11, fontweight="medium", pad=18)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=8, framealpha=0.7)

    ax_radar.yaxis.grid(True, linewidth=0.3, alpha=0.5)
    ax_radar.xaxis.grid(True, linewidth=0.3, alpha=0.5)
    ax_radar.spines["polar"].set_linewidth(0.4)

    # ----- Right: horizontal lollipop of per-class F1 deltas -----
    ax_lp = fig.add_subplot(122)

    all_diseases = [
        d for d in PATHOLOGIES if d in baseline["per_class"] and d in ours["per_class"]
    ]
    common_sorted = sorted(
        [d for d in all_diseases if d not in RARE_DISEASES],
        key=lambda d: ours["per_class"][d]["f1"] - baseline["per_class"][d]["f1"],
    )
    rare_sorted = sorted(
        [d for d in all_diseases if d in RARE_DISEASES],
        key=lambda d: ours["per_class"][d]["f1"] - baseline["per_class"][d]["f1"],
    )
    ordered = common_sorted + rare_sorted

    y_pos = np.arange(len(ordered))
    deltas, pcts = [], []
    for d in ordered:
        delta = ours["per_class"][d]["f1"] - baseline["per_class"][d]["f1"]
        deltas.append(delta)
        base_f1 = baseline["per_class"][d]["f1"]
        pcts.append(delta / base_f1 * 100 if base_f1 > 0.001 else 999)

    sorted_deltas = sorted(deltas)
    clip_threshold = sorted_deltas[-2] * 1.6 if len(sorted_deltas) > 1 else max(deltas) * 1.2

    for i, (d, delta, pct) in enumerate(zip(ordered, deltas, pcts)):
        region = REGION_MAP.get(d, "Other")
        c = REGION_COLORS.get(region, "#B0BEC5")
        dark_c = REGION_COLORS_DARK.get(region, "#546E7A")

        display_delta = min(delta, clip_threshold)
        is_clipped = delta > clip_threshold

        ax_lp.hlines(y=i, xmin=0, xmax=display_delta, color=c, linewidth=2.8, alpha=0.85)
        ax_lp.scatter(
            display_delta, i, color=dark_c, s=50, zorder=5, edgecolors="white", linewidth=0.5,
            marker="o",
        )

        if is_clipped:
            bx = clip_threshold * 0.92
            for offset in [-0.12, 0.12]:
                ax_lp.plot(
                    [bx - 0.003, bx + 0.003],
                    [i + offset - 0.08, i + offset + 0.08],
                    color="#424242",
                    linewidth=0.8,
                    clip_on=False,
                )

        label_text = f"+{pct:.0f}%" if pct >= 0 else f"{pct:.0f}%"
        text_x = display_delta + clip_threshold * 0.04
        ax_lp.text(text_x, i, label_text, va="center", ha="left", fontsize=7.5, color=dark_c,
                   fontweight="medium")

    ylabels = [SHORT_NAMES.get(d, d) for d in ordered]
    ax_lp.set_yticks(y_pos)
    ax_lp.set_yticklabels(ylabels, fontsize=8.5)

    sep_y = len(common_sorted) - 0.5
    ax_lp.axhline(y=sep_y, color="#9E9E9E", linewidth=0.6, linestyle="--", alpha=0.6)

    common_mid = (len(common_sorted) - 1) / 2
    rare_mid = len(common_sorted) + (len(rare_sorted) - 1) / 2

    ax_lp.text(
        0.96, rare_mid / (len(ordered) - 1), "Low-prev.",
        fontsize=7, color="#616161", ha="center", va="center", fontstyle="italic",
        transform=ax_lp.get_yaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF3E0", edgecolor="#FFB74D",
                  linewidth=0.4, alpha=0.85),
    )
    ax_lp.text(
        0.96, common_mid / (len(ordered) - 1), "High-prev.",
        fontsize=7, color="#616161", ha="center", va="center", fontstyle="italic",
        transform=ax_lp.get_yaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F5F5F5", edgecolor="#BDBDBD",
                  linewidth=0.4, alpha=0.85),
    )

    ax_lp.axvline(x=0, color="#424242", linewidth=0.5, alpha=0.4)
    ax_lp.set_xlim(-0.005, clip_threshold * 1.25)

    ax_lp.set_xlabel("$\\Delta$F1 (APA-RRG $-$ PromptMRG)", fontsize=9.5)
    ax_lp.set_title("Per-class F1 Improvement", fontsize=11, fontweight="medium")
    ax_lp.spines["top"].set_visible(False)
    ax_lp.spines["right"].set_visible(False)
    ax_lp.spines["left"].set_linewidth(0.4)
    ax_lp.spines["bottom"].set_linewidth(0.4)
    ax_lp.tick_params(axis="y", length=0)
    ax_lp.grid(axis="x", alpha=0.15, linewidth=0.4)

    legend_elements = [
        mpatches.Patch(facecolor=REGION_COLORS[r], edgecolor=REGION_COLORS_DARK[r],
                       linewidth=0.4, label=r)
        for r in ["Cardiac", "Pulmonary", "Pleural", "Skeletal", "Other"]
    ]
    leg = ax_lp.legend(
        handles=legend_elements, loc="lower right", fontsize=6.5, ncol=3, framealpha=0.6,
        columnspacing=0.6, handletextpad=0.3, handlelength=1.2, borderpad=0.4,
    )
    leg.set_title("Anatomical Region", prop={"size": 6.5})

    plt.tight_layout(w_pad=2.5)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=str, required=True,
        help="Path to baseline (PromptMRG) per-class evaluation JSON.",
    )
    parser.add_argument(
        "--ours", type=str, required=True,
        help="Path to APA-RRG per-class evaluation JSON.",
    )
    parser.add_argument("--output", type=str, default="fig_rare_disease.pdf")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    ours = load_results(args.ours)
    plot_combined_figure(baseline, ours, args.output)


if __name__ == "__main__":
    main()
