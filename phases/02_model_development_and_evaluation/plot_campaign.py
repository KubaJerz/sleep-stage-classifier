#!/usr/bin/env python3
"""Plot key metrics across experiments.

Usage:
    python plot_campaign.py                        # all campaigns combined
    python plot_campaign.py campaigns/apr03        # single campaign
    python plot_campaign.py campaigns/apr03 campaigns/apr10  # specific campaigns
"""

import sys
import csv
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CAMPAIGNS_DIR = pathlib.Path("campaigns")
METRICS = [
    ("accuracy", "Accuracy"),
    ("kappa", "Cohen's Kappa"),
    ("macro_f1", "Macro F1"),
]


def load_campaign(campaign_dir):
    path = pathlib.Path(campaign_dir) / "results.tsv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows


def running_best(values):
    best, out = 0, []
    for v in values:
        best = max(best, v)
        out.append(best)
    return out


def style_ax(ax):
    """Apply clean minimal style to an axis."""
    ax.grid(True, color="#E0E0E0", linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#555555", labelsize=9)


def plot_single(campaign_dir):
    """Plot one campaign with per-metric panels, values annotated."""
    campaign_dir = pathlib.Path(campaign_dir)
    rows = load_campaign(campaign_dir)
    name = campaign_dir.name
    n = len(rows)
    n_kept = sum(1 for r in rows if r["status"] == "keep")
    xs = list(range(1, n + 1))

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (key, label) in zip(axes, METRICS):
        style_ax(ax)
        vals = [float(r[key]) for r in rows]
        best = running_best(vals)

        # discarded as faded dots
        disc_x = [xs[i] for i in range(n) if rows[i]["status"] != "keep"]
        disc_v = [vals[i] for i in range(n) if rows[i]["status"] != "keep"]
        ax.scatter(disc_x, disc_v, color="#B0C0B0", s=20, zorder=2, alpha=0.5)

        # keep points
        keep_x = [xs[i] for i in range(n) if rows[i]["status"] == "keep"]
        keep_v = [vals[i] for i in range(n) if rows[i]["status"] == "keep"]
        ax.scatter(keep_x, keep_v, color="#2CA02C", s=40, zorder=4)

        # running best staircase
        stair_x, stair_y = [xs[0]], [best[0]]
        for i in range(1, n):
            if best[i] > best[i - 1]:
                stair_x.extend([xs[i], xs[i]])
                stair_y.extend([best[i - 1], best[i]])
            else:
                stair_x.append(xs[i])
                stair_y.append(best[i])
        ax.plot(stair_x, stair_y, "-", color="#2CA02C", linewidth=1.8, alpha=0.8, zorder=3)

        # annotate keep points with value
        for i in range(n):
            if rows[i]["status"] == "keep":
                ax.annotate(
                    f"{vals[i]:.3f}",
                    (xs[i], vals[i]),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=7.5, color="#2CA02C", rotation=30,
                )

        ax.set_ylabel(label, fontsize=11, color="#333333")

        # tighter y range: pad 5% around actual data range
        vmin, vmax = min(vals), max(vals)
        pad = (vmax - vmin) * 0.15 if vmax > vmin else 0.01
        ax.set_ylim(vmin - pad, vmax + pad)

    axes[0].legend(
        handles=[
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="#B0C0B0", markersize=5, label="Discarded"),
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="#2CA02C", markersize=6, label="Kept"),
            plt.Line2D([], [], color="#2CA02C", linewidth=1.8, label="Running best"),
        ],
        loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#CCCCCC",
    )

    axes[-1].set_xlabel("Experiment #", fontsize=11, color="#333333")
    axes[0].set_title(
        f"{name}: {n} Experiments, {n_kept} Kept Improvements",
        fontsize=13, color="#333333", pad=12,
    )
    fig.tight_layout()

    out = campaign_dir / "campaign_metrics.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"saved → {out}")
    plt.close(fig)


def plot_all(campaign_dirs):
    """Karpathy-style wide single-panel plot, all campaigns combined."""
    # collect all data
    all_rows = []
    boundaries = []  # (x_position, campaign_name)
    offset = 0
    for campaign_dir in campaign_dirs:
        campaign_dir = pathlib.Path(campaign_dir)
        rows = load_campaign(campaign_dir)
        for i, r in enumerate(rows):
            r["_x"] = offset + i + 1
            r["_campaign"] = campaign_dir.name
        all_rows.extend(rows)
        if offset > 0:
            boundaries.append(offset + 0.5)
        offset += len(rows)

    n = len(all_rows)
    n_kept = sum(1 for r in all_rows if r["status"] == "keep")
    names = ", ".join(pathlib.Path(d).name for d in campaign_dirs)

    fig, axes = plt.subplots(3, 1, figsize=(max(18, n * 0.2), 10), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (key, label) in zip(axes, METRICS):
        style_ax(ax)
        xs = [r["_x"] for r in all_rows]
        vals = [float(r[key]) for r in all_rows]
        best = running_best(vals)

        # discarded
        disc_x = [r["_x"] for r in all_rows if r["status"] != "keep"]
        disc_v = [float(r[key]) for r in all_rows if r["status"] != "keep"]
        ax.scatter(disc_x, disc_v, color="#B0C0B0", s=18, zorder=2, alpha=0.4)

        # kept
        keep_x = [r["_x"] for r in all_rows if r["status"] == "keep"]
        keep_v = [float(r[key]) for r in all_rows if r["status"] == "keep"]
        ax.scatter(keep_x, keep_v, color="#2CA02C", s=35, zorder=4)

        # running best staircase
        stair_x, stair_y = [xs[0]], [best[0]]
        for i in range(1, n):
            if best[i] > best[i - 1]:
                stair_x.extend([xs[i], xs[i]])
                stair_y.extend([best[i - 1], best[i]])
            else:
                stair_x.append(xs[i])
                stair_y.append(best[i])
        ax.plot(stair_x, stair_y, "-", color="#2CA02C", linewidth=1.8, alpha=0.8, zorder=3)

        # annotate kept points with short description
        for r in all_rows:
            if r["status"] == "keep":
                desc = r["description"].split("(")[0].strip()  # drop (STRUCTURAL) etc
                if len(desc) > 35:
                    desc = desc[:32] + "..."
                ax.annotate(
                    desc,
                    (r["_x"], float(r[key])),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=6.5, color="#2CA02C", rotation=30,
                )

        # campaign boundaries
        for bx in boundaries:
            ax.axvline(bx, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_ylabel(label, fontsize=11, color="#333333")

        vmin, vmax = min(vals), max(vals)
        pad = (vmax - vmin) * 0.15 if vmax > vmin else 0.01
        ax.set_ylim(vmin - pad, vmax + pad)

    axes[0].legend(
        handles=[
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="#B0C0B0", markersize=5, label="Discarded"),
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="#2CA02C", markersize=6, label="Kept"),
            plt.Line2D([], [], color="#2CA02C", linewidth=1.8, label="Running best"),
        ],
        loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#CCCCCC",
    )

    axes[-1].set_xlabel("Experiment #", fontsize=11, color="#333333")
    axes[0].set_title(
        f"Autoresearch Progress ({names}): {n} Experiments, {n_kept} Kept Improvements",
        fontsize=13, color="#333333", pad=12,
    )
    fig.tight_layout()

    out = CAMPAIGNS_DIR / "all_campaigns_metrics.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"saved → {out}")
    plt.close(fig)


def main():
    args = sys.argv[1:]

    if args:
        campaign_dirs = [pathlib.Path(a) for a in args]
    else:
        campaign_dirs = sorted(
            d for d in CAMPAIGNS_DIR.iterdir()
            if d.is_dir() and (d / "results.tsv").exists()
        )

    if not campaign_dirs:
        print("No campaigns found.")
        return

    for d in campaign_dirs:
        plot_single(d)

    plot_all(campaign_dirs)


if __name__ == "__main__":
    main()
