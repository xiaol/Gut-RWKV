import argparse
import json
import math
from pathlib import Path


def read_progress(path):
    points = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("{\"step\""):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid progress JSON at {path}:{line_number}") from error
        if not isinstance(value.get("step"), int) or not isinstance(value.get("loss"), (int, float)):
            raise ValueError(f"Invalid progress fields at {path}:{line_number}")
        if not math.isfinite(value["loss"]):
            raise ValueError(f"Nonfinite progress loss at {path}:{line_number}")
        points.append(value)
    if not points:
        raise ValueError(f"No training progress found in {path}")
    if any(left["step"] >= right["step"] for left, right in zip(points, points[1:])):
        raise ValueError(f"Training progress steps are not strictly increasing in {path}")
    return points


def summarize(points, summary_path=None):
    losses = [point["loss"] for point in points]
    summary = {
        "logged_points": len(points),
        "first_logged_step": points[0]["step"],
        "last_logged_step": points[-1]["step"],
        "first_loss": losses[0],
        "last_logged_loss": losses[-1],
        "minimum_logged_loss": min(losses),
        "minimum_loss_step": points[losses.index(min(losses))]["step"],
        "maximum_logged_loss": max(losses),
        "maximum_loss_step": points[losses.index(max(losses))]["step"],
    }
    if summary_path:
        training_summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        summary["final_step"] = training_summary["steps"]
        summary["final_training_loss"] = training_summary["final_training_loss"]
        if training_summary["steps"] < points[-1]["step"]:
            raise ValueError("Training summary ends before logged progress")
        summary["final_initial_state_norm"] = training_summary.get("initial_state_norm")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract JSON training-loss curves from Gut-RWKV logs")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=LOG",
                        help="Curve label and train log path; repeat for multiple arms")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    curves = {}
    for specification in args.run:
        if "=" not in specification:
            parser.error("Each --run must be LABEL=LOG")
        label, filename = specification.split("=", 1)
        if not label or not filename:
            parser.error("Each --run must have a nonempty label and path")
        points = read_progress(filename)
        summary_path = Path(filename).parent / "train-summary.json"
        curves[label] = {
            "log": filename,
            "summary": str(summary_path) if summary_path.exists() else None,
            "summary_metrics": summarize(points, summary_path if summary_path.exists() else None),
            "points": points,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"curves": curves}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({label: curve["summary_metrics"] for label, curve in curves.items()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
