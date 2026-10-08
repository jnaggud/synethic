"""Compose labeled figures from the project's unchanged historical output grids.

Install requirements-figures.txt. Run from any directory in the repository.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
NAVY, TEAL, MUTED = "#10232f", "#6ae2bc", "#a9bcc7"


def build_social_preview():
    """Compose a repository link card using an unaltered historical sample grid."""
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=NAVY)
    fig.text(0.055, 0.83, "SynEthic", color="white", fontsize=48, weight="bold")
    fig.text(
        0.058,
        0.74,
        "MACHINE LEARNING / RESEARCH PROTOTYPE",
        color=TEAL,
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.055,
        0.49,
        "Synthetic chest X-ray\ngeneration",
        color="white",
        fontsize=29,
        linespacing=1.35,
    )
    fig.text(0.058, 0.34, "Python · TensorFlow · DCGAN", color=MUTED, fontsize=15)
    ax = fig.add_axes([0.625, 0.245, 0.33, 0.63])
    ax.imshow(Image.open(ASSETS / "epoch-500.png"))
    ax.set_axis_off()
    fig.text(
        0.632,
        0.205,
        "HISTORICAL SAMPLES / EPOCH 500",
        color=MUTED,
        fontsize=9,
    )
    fig.text(
        0.058,
        0.11,
        "CONFIGURABLE TRAINING  /  CHECKPOINT RECOVERY  /  MODEL EXPORT",
        color=TEAL,
        fontsize=10,
        weight="bold",
    )
    fig.savefig(ASSETS / "social-preview.png", dpi=100)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(10, 7.5), dpi=160, facecolor=NAVY)
    fig.text(0.06, 0.90, "SynEthic", color="white", fontsize=36, weight="bold")
    fig.text(
        0.06, 0.85, "COMPUTER VISION / RESEARCH PROTOTYPE", color=TEAL, fontsize=9.5, weight="bold"
    )
    fig.text(
        0.06,
        0.60,
        "Synthetic\nchest X-ray\ngeneration",
        color="white",
        fontsize=28,
        weight="bold",
        linespacing=1.25,
    )
    fig.text(
        0.06,
        0.37,
        "Python + TensorFlow\nDeep Convolutional GAN\nExperimental SSIM regularization",
        color=MUTED,
        fontsize=12,
        linespacing=1.8,
    )
    card = FancyBboxPatch(
        (0.48, 0.245),
        0.47,
        0.55,
        boxstyle="round,pad=0.009,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="none",
        zorder=0,
    )
    fig.add_artist(card)
    ax = fig.add_axes([0.49, 0.258, 0.45, 0.51], zorder=1)
    ax.imshow(Image.open(ASSETS / "epoch-500.png"))
    ax.set_axis_off()
    fig.text(0.5, 0.22, "HISTORICAL MODEL OUTPUT / EPOCH 500", color=MUTED, fontsize=8.5)
    fig.text(0.06, 0.145, "CONFIGURABLE RUNS", color=TEAL, fontsize=10, weight="bold")
    fig.text(0.37, 0.145, "CHECKPOINT RECOVERY", color=TEAL, fontsize=10, weight="bold")
    fig.text(0.72, 0.145, "AUTOMATED TESTS", color=TEAL, fontsize=10, weight="bold")
    fig.text(
        0.06,
        0.065,
        "Prototype samples. Clinical utility and formal privacy protection are unvalidated.",
        color=MUTED,
        fontsize=8.7,
    )
    fig.savefig(ASSETS / "portfolio-cover.png", dpi=100)
    fig.savefig(ASSETS / "portfolio-cover.svg")
    svg = ASSETS / "portfolio-cover.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="white")
    for ax, epoch in zip(axes, (1, 250, 500)):
        ax.imshow(Image.open(ASSETS / f"epoch-{epoch:03d}.png"))
        ax.set_title(f"Epoch {epoch}", fontsize=16, color=NAVY, pad=0)
        ax.set_axis_off()
    fig.suptitle(
        "SynEthic / historical training snapshots",
        x=0.035,
        ha="left",
        fontsize=23,
        weight="bold",
        color=NAVY,
    )
    fig.text(
        0.035,
        0.045,
        "Unmodified saved grids from the original prototype. Epoch labels come from filenames; full run logs are unavailable.",
        fontsize=10,
        color="#516572",
    )
    fig.subplots_adjust(left=0.025, right=0.975, top=0.84, bottom=0.12, wspace=0.025)
    fig.savefig(ASSETS / "training-progression.png", dpi=120)
    plt.close(fig)
    build_social_preview()


if __name__ == "__main__":
    main()
