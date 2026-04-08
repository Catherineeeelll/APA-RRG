"""
Prototype activation deviation heatmap for the Pathology-Anchored
Representation Calibration analysis (Section 4.5 of the paper).

For each disease, the average prototype selection weight is computed
over the test-set samples that are positive for that disease, then the
uniform baseline (1/14) is subtracted to highlight per-disease selection
preferences. Only the 14 CheXpert-aligned prototype slots are shown.

Usage:
    python fig_prototype_final.py \
        --input results/apa_rrg/prototype_activations.npz \
        --output fig_prototype_analysis.pdf
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

DIS14_Y = [
    "Enl. Cardio.",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pl. Effusion",
    "Pl. Other",
    "Fracture",
    "Sup. Devices",
    "No Finding",
]

# Indices of low-prevalence pathologies (italicised in the figure).
RARE_INDICES = [0, 3, 5, 6, 8, 10, 11]

# Anatomical region groups (matches models/apg.py and visualize_rare_disease.py).
# 0,1: Cardiac. 2,3,4,5,6,7: Pulmonary. 8,9,10: Pleural.
# 11: Skeletal. 12,13: Other (devices + no finding).
_GROUPS = {
    "Cardiac": ([0, 1], "#D4524B"),
    "Pulmonary": ([2, 3, 4, 5, 6, 7], "#4878A6"),
    "Pleural": ([8, 9, 10], "#4FA04A"),
    "Skeletal": ([11], "#E49629"),
    "Other": ([12, 13], "#777777"),
}
LABEL_COLORS = []
for i in range(14):
    for _name, (indices, color) in _GROUPS.items():
        if i in indices:
            LABEL_COLORS.append(color)
            break

# Horizontal separators are drawn above these row indices.
BOUNDARIES = [2, 8, 11, 12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="results/apa_rrg/prototype_activations.npz",
        help="NPZ file with arrays 'proto_weights' [N, 14] and 'labels' [N, 14].",
    )
    parser.add_argument("--output", type=str, default="fig_prototype_analysis.pdf")
    args = parser.parse_args()

    data = np.load(args.input)
    proto_w = data["proto_weights"]
    labels = data["labels"]

    uniform = 1.0 / 14
    avg_proto = np.zeros((14, 14))
    for d in range(14):
        mask = labels[:, d] == 1
        if mask.sum() > 0:
            avg_proto[d] = proto_w[mask].mean(axis=0)
    proto_dev = avg_proto - uniform

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "mathtext.fontset": "dejavuserif",
        }
    )

    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    fig.subplots_adjust(left=0.28, right=0.84, top=0.97, bottom=0.12)

    pmax = max(abs(proto_dev.min()), abs(proto_dev.max()))
    im = ax.imshow(proto_dev, cmap="RdBu_r", vmin=-pmax, vmax=pmax, aspect="equal")

    ax.set_yticks(range(14))
    ax.set_yticklabels(DIS14_Y, fontsize=7)
    for ti, yt in enumerate(ax.get_yticklabels()):
        yt.set_color(LABEL_COLORS[ti])
        yt.set_fontweight("semibold")
        if ti in RARE_INDICES:
            yt.set_fontstyle("italic")

    ax.set_xticks(range(14))
    ax.set_xticklabels([str(k + 1) for k in range(14)], fontsize=7)
    ax.set_xlabel("Prototype slot $\\mathbf{p}_k$", fontsize=8, labelpad=4)

    ax.tick_params(axis="both", length=2, width=0.3, pad=2)

    for b in BOUNDARIES:
        ax.axhline(y=b - 0.5, color="white", linewidth=0.7)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.5, width=0.3, length=2)
    cb.set_label("$\\Delta w$", fontsize=9, labelpad=3)
    cb.outline.set_linewidth(0.3)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        args.output.replace(".pdf", ".png"),
        format="png",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=300,
    )
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
