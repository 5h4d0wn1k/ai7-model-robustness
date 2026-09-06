"""
AI7 — Model Robustness Auditor
Pure-Python adversarial robustness testing for binary classifiers.
"""

import math
import random
import copy

# ─── Embedded Synthetic Dataset ───────────────────────────────────────
# 2D points: class 0 clustered near (2,2), class 1 near (6,6)
TRAINING_DATA = [
    # Class 0
    ([1.0, 1.5], 0), ([1.8, 2.2], 0), ([2.5, 1.0], 0), ([2.0, 2.8], 0),
    ([1.2, 2.5], 0), ([2.8, 1.8], 0), ([1.5, 1.0], 0), ([2.2, 2.5], 0),
    ([3.0, 2.0], 0), ([1.0, 2.0], 0),
    # Class 1
    ([5.5, 6.0], 1), ([6.2, 5.5], 1), ([6.8, 6.5], 1), ([5.0, 6.8], 1),
    ([6.0, 7.0], 1), ([5.8, 5.2], 1), ([7.0, 6.0], 1), ([6.5, 7.2], 1),
    ([5.5, 5.5], 1), ([7.2, 5.8], 1),
]


class LogisticRegression:
    """Minimal logistic regression from scratch."""

    def __init__(self, n_features=2, lr=0.1):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = lr

    def _sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self._sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def train(self, data, epochs=100):
        for _ in range(epochs):
            for x, y in data:
                p = self.predict_proba(x)
                error = p - y
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * error * x[i]
                self.bias -= self.lr * error


def accuracy(model, data):
    correct = sum(1 for x, y in data if model.predict(x) == y)
    return correct / len(data) if data else 0.0


def l2_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def fgsm_attack(model, x, y_true, epsilon=0.3):
    """Fast Gradient Sign Method — using finite-difference gradient approx."""
    grads = []
    h = 1e-4
    for i in range(len(x)):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad_i = (model.predict_proba(x_plus) - model.predict_proba(x_minus)) / (2 * h)
        grads.append(grad_i)
    # Move in direction that increases loss (away from true label)
    sign = [-1.0 if y_true == 1 else 1.0 for g in grads]
    perturbed = [xi + epsilon * s for xi, s in zip(x, sign)]
    return perturbed


def gaussian_perturbation(x, sigma=0.5):
    noisy = [xi + random.gauss(0, sigma) for xi in x]
    return noisy


def label_flip_attack(data, flip_ratio=0.2):
    """Return a poisoned copy with flip_ratio fraction of labels flipped."""
    poisoned = list(data)
    n_flip = max(1, int(len(poisoned) * flip_ratio))
    indices = random.sample(range(len(poisoned)), n_flip)
    for i in indices:
        x, y = poisoned[i]
        poisoned[i] = (x, 1 - y)
    return poisoned


def min_perturbation_to_flip(model, x, y_true, step=0.05, max_eps=3.0):
    """Binary search for smallest epsilon that flips the prediction."""
    lo, hi = 0.0, max_eps
    if model.predict(x) != y_true:
        return None  # already misclassified — nothing to measure
    for _ in range(30):
        mid = (lo + hi) / 2
        perturbed = gaussian_perturbation(x, sigma=mid)
        if model.predict(perturbed) != y_true:
            hi = mid
        else:
            lo = mid
    return hi


def run_experiment(seed: int = 42, epsilon: float = 0.3,
                   epochs: int = 200) -> dict:
    """Run the robustness audit and return structured results (pure Python)."""
    random.seed(seed)

    model = LogisticRegression(lr=0.5)
    model.train(TRAINING_DATA, epochs=epochs)
    clean_acc = accuracy(model, TRAINING_DATA)

    flipped = 0
    total_dist = 0.0
    for x, y in TRAINING_DATA:
        adv = fgsm_attack(model, x, y, epsilon=epsilon)
        orig_pred = model.predict(x)
        adv_pred = model.predict(adv)
        dist = l2_distance(x, adv)
        total_dist += dist
        if adv_pred != orig_pred:
            flipped += 1
    fgsm_rate = flipped / len(TRAINING_DATA)
    avg_dist = total_dist / len(TRAINING_DATA)

    noise_levels = [0.1, 0.3, 0.5, 1.0]
    noise_curve = {}
    for sigma in noise_levels:
        noise_flipped = 0
        for x, y in TRAINING_DATA:
            for _ in range(10):
                noisy = gaussian_perturbation(x, sigma=sigma)
                if model.predict(noisy) != y:
                    noise_flipped += 1
        total_trials = len(TRAINING_DATA) * 10
        noise_curve[str(sigma)] = {
            "flip_rate": noise_flipped / total_trials,
            "flipped": noise_flipped,
            "trials": total_trials,
        }

    min_dists = []
    for x, y in TRAINING_DATA:
        d = min_perturbation_to_flip(model, x, y)
        if d is not None:
            min_dists.append(d)

    poison_curve = {}
    for ratio in (0.1, 0.2, 0.3, 0.5):
        model_p = LogisticRegression(lr=0.5)
        poisoned = label_flip_attack(TRAINING_DATA, flip_ratio=ratio)
        model_p.train(poisoned, epochs=epochs)
        poisoned_acc = accuracy(model_p, TRAINING_DATA)
        poison_curve[str(ratio)] = {
            "accuracy": poisoned_acc,
            "drop": clean_acc - poisoned_acc,
        }

    robustness_score = max(0.0, 1.0 - fgsm_rate) * 100.0
    if robustness_score >= 80:
        verdict = "ROBUST — Low evasion risk"
    elif robustness_score >= 50:
        verdict = "MODERATE — Some evasion vulnerability"
    else:
        verdict = "WEAK — Highly vulnerable to evasion"

    return {
        "dataset": {
            "samples": len(TRAINING_DATA),
            "classes": 2,
            "features": len(TRAINING_DATA[0][0]),
        },
        "model": {
            "type": "LogisticRegression",
            "lr": 0.5,
            "epochs": epochs,
            "weights": [round(w, 4) for w in model.weights],
            "bias": round(model.bias, 4),
            "clean_accuracy": clean_acc,
        },
        "fgsm_evasion": {
            "epsilon": epsilon,
            "samples_attacked": len(TRAINING_DATA),
            "predictions_flipped": flipped,
            "flip_rate": fgsm_rate,
            "avg_perturbation_l2": avg_dist,
        },
        "noise_robustness": noise_curve,
        "min_perturbation": {
            "samples_flippable": len(min_dists),
            "samples_total": len(TRAINING_DATA),
            "avg_epsilon": (sum(min_dists) / len(min_dists)
                            if min_dists else None),
            "min_epsilon": (min(min_dists) if min_dists else None),
            "max_epsilon": (max(min_dists) if min_dists else None),
        },
        "poisoning": poison_curve,
        "score": {
            "clean_accuracy": clean_acc,
            "fgsm_evasion_resilience": 1.0 - fgsm_rate,
            "robustness_score": round(robustness_score, 3),
            "verdict": verdict,
        },
    }


def format_report(results: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  AI7 — Model Robustness Audit Report")
    lines.append("=" * 60)
    d = results["dataset"]
    m = results["model"]
    lines.append(f"\nDataset     : {d['samples']} samples, {d['classes']} classes, "
                 f"{d['features']} features")
    lines.append(f"Model       : {m['type']} (lr={m['lr']}, {m['epochs']} epochs)")
    lines.append(f"Weights     : {m['weights']}")
    lines.append(f"Bias        : {m['bias']}")
    lines.append(f"Clean Acc   : {m['clean_accuracy']:.1%}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("  1. FGSM Evasion Attack")
    lines.append("-" * 60)
    f = results["fgsm_evasion"]
    lines.append(f"  Samples attacked     : {f['samples_attacked']}")
    lines.append(f"  Prediction flipped   : {f['predictions_flipped']}/"
                 f"{f['samples_attacked']} ({f['flip_rate']:.1%})")
    lines.append(f"  Avg perturbation L2  : {f['avg_perturbation_l2']:.4f}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("  2. Gaussian Noise Robustness")
    lines.append("-" * 60)
    for sigma, row in results["noise_robustness"].items():
        lines.append(f"  sigma={sigma}: flip rate={row['flip_rate']:.1%} "
                     f"({row['flipped']}/{row['trials']})")
    lines.append("")

    lines.append("-" * 60)
    lines.append("  3. Minimum Perturbation to Flip (binary search)")
    lines.append("-" * 60)
    mp = results["min_perturbation"]
    lines.append(f"  Samples where flip possible : {mp['samples_flippable']}/"
                 f"{mp['samples_total']}")
    if mp["avg_epsilon"] is not None:
        lines.append(f"  Avg min epsilon              : {mp['avg_epsilon']:.4f}")
        lines.append(f"  Min epsilon                  : {mp['min_epsilon']:.4f}")
        lines.append(f"  Max epsilon                  : {mp['max_epsilon']:.4f}")
    else:
        lines.append("  No samples could be flipped within max epsilon.")
    lines.append("")

    lines.append("-" * 60)
    lines.append("  4. Label-Flip Poisoning Assessment")
    lines.append("-" * 60)
    for ratio, row in results["poisoning"].items():
        lines.append(f"  Poison {float(ratio):.0%}: acc={row['accuracy']:.1%} "
                     f"(drop={row['drop']:+.1%})")
    lines.append("")

    lines.append("=" * 60)
    lines.append("  ROBUSTNESS SUMMARY")
    lines.append("=" * 60)
    s = results["score"]
    lines.append(f"  Clean Accuracy        : {s['clean_accuracy']:.1%}")
    lines.append(f"  FGSM Evasion Resilience: {s['fgsm_evasion_resilience']:.1%}")
    lines.append(f"  Robustness Score       : {s['robustness_score']:.0f}/100")
    lines.append(f"  Verdict                : {s['verdict']}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("  Audit complete.")
    lines.append("=" * 60)
    return "\n".join(lines)


def run_audit():
    print(format_report(run_experiment()))


def main(argv=None):
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        prog="ai7-model-robustness",
        description="Adversarial + corruption robustness scoring for a local "
                    "logistic classifier. Pure-Python, offline, self-contained.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="FGSM perturbation magnitude")
    parser.add_argument("--epochs", type=int, default=200,
                        help="training epochs")
    parser.add_argument("--output", metavar="FILE",
                        help="write JSON report to FILE (e.g. reports/ai7-report.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress human-readable output")
    args = parser.parse_args(argv)

    results = run_experiment(seed=args.seed, epsilon=args.epsilon,
                             epochs=args.epochs)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    if not args.quiet:
        print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
