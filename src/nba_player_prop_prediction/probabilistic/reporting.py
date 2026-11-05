"""Human-readable model comparisons and reliability diagrams from saved results."""

import json
from pathlib import Path


def render(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((root / "summary.json").read_text())
    lines = [
        "# Milestone 3 experiment",
        "",
        "Appearance-conditional historical forecasts.",
        "Training/validation may be sampled; calibration and test use all eligible rows.",
        "",
        "| Target | Model | Test MAE | NLL | Raw Brier | Sigmoid Brier | Isotonic Brier |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for (target, models), ax in zip(summary.items(), axes.flat, strict=False):
        selection = json.loads((root / target / "validation_selection.json").read_text())[
            "selected"
        ]
        for name, r in models.items():
            if "probabilities" not in r:
                continue
            p = r["probabilities"]
            lines.append(
                f"| {target} | {name} | {r['test']['mae']:.3f} | "
                f"{r['test']['nll']:.3f} | {p['identity']['brier']:.4f} | "
                f"{p['sigmoid']['brier']:.4f} | {p['isotonic']['brier']:.4f} |"
            )
        if selection:
            for method, results in models[selection]["probabilities"].items():
                reliability = results["reliability"]
                ax.plot(
                    [r["predicted"] for r in reliability],
                    [r["observed"] for r in reliability],
                    marker="o",
                    label=method,
                )
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(
            title=f"{target}: validation-selected {selection}",
            xlabel="Predicted probability",
            ylabel="Observed hit frequency",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "reliability.png")
    plt.close(fig)
    lines.extend(
        [
            "",
            "Thresholds are statistical events, not verified sportsbook markets.",
            "Pooled calibrators preserve ordering. Intervals/NLL use raw distributions.",
            "Confidence reports include losing selections and game-cluster bootstrap intervals.",
            "No appearance, injury or lineup probability model; no PRA independence assumption.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines))


def base_rate_reference(data_path, root):
    """Training-only empirical threshold frequencies, without test-driven selection."""
    import numpy as np
    import pandas as pd

    from nba_player_prop_prediction.probabilistic.data import partitions
    from nba_player_prop_prediction.probabilistic.distributions import TARGETS
    from nba_player_prop_prediction.probabilistic.evaluation import probability_metrics
    from nba_player_prop_prediction.probabilistic.experiment import sample
    from nba_player_prop_prediction.utils.io import write_json

    config = json.loads((root / "experiment.json").read_text())
    data = pd.read_parquet(data_path)
    output = {}
    for target in config["targets"]:
        parts = partitions(data.loc[data.appearance & data[f"label_{target}"].notna()])
        training = sample(parts["train"], config["train_limit"])
        test = parts["test"]
        lines = np.array(TARGETS[target])
        rates = (training[f"label_{target}"].to_numpy()[:, None] >= lines).mean(axis=0)
        observed = test[f"label_{target}"].to_numpy()[:, None] >= lines
        output[target] = {
            "lines": lines.tolist(),
            "training_base_rates": rates.tolist(),
            "test": probability_metrics(np.broadcast_to(rates, observed.shape), observed),
        }
    write_json(root / "empirical_base_rates.json", output)
    return output
