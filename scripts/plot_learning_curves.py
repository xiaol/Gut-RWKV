import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


GROUPS = (
    ("Cross-entropy", ("ce_seed42", "ce_seed43", "ce_seed44")),
    ("Pathwise RL", ("pathwise_seed42", "pathwise_seed43", "pathwise_seed44")),
    ("Policy and proper-score objectives", ("reinforce_seed42", "proper_score_seed42", "noisy_proper_seed42")),
)


def moving_average(values, window=15):
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(sum(values[start:index + 1]) / (index - start + 1))
    return result


def plot_curves(source, output):
    curves = json.loads(Path(source).read_text(encoding="utf-8"))["curves"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    colors = ("#2563eb", "#16a34a", "#dc2626")

    for axis, (title, labels) in zip(axes, GROUPS):
        for color, label in zip(colors, labels):
            points = curves[label]["points"]
            steps = [point["step"] for point in points]
            losses = [point["loss"] for point in points]
            display_name = label.replace("_seed", " seed ").replace("_", " ").title()
            axis.plot(steps, losses, color=color, alpha=0.16, linewidth=0.8)
            axis.plot(steps, moving_average(losses), color=color, linewidth=1.8, label=display_name)
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, frameon=False)

    axes[0].set_ylabel("Logged objective loss")
    figure.suptitle("Gut-RWKV training objectives", fontsize=14, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "Raw points are faint; solid lines are 15-point trailing means. Objective scales are not comparable.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", metadata={"Title": "Gut-RWKV training objectives"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Plot extracted Gut-RWKV learning curves")
    parser.add_argument("--input", default="reports/learning-curves.json")
    parser.add_argument("--output", default="reports/learning-curves.svg")
    args = parser.parse_args()
    plot_curves(args.input, args.output)


if __name__ == "__main__":
    main()
