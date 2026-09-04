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
    if model.predict(x) == y_true:
        return None  # already misclassified
    for _ in range(30):
        mid = (lo + hi) / 2
        perturbed = gaussian_perturbation(x, sigma=mid)
        if model.predict(perturbed) != y_true:
            hi = mid
        else:
            lo = mid
    return hi


def run_audit():
    random.seed(42)

    # Train model
    model = LogisticRegression(lr=0.5)
    model.train(TRAINING_DATA, epochs=200)
    clean_acc = accuracy(model, TRAINING_DATA)

    print("=" * 60)
    print("  AI7 — Model Robustness Audit Report")
    print("=" * 60)
    print()
    print(f"Dataset     : {len(TRAINING_DATA)} samples, 2 classes, 2 features")
    print(f"Model       : Logistic Regression (lr=0.5, 200 epochs)")
    print(f"Weights     : {[round(w, 4) for w in model.weights]}")
    print(f"Bias        : {model.bias:.4f}")
    print(f"Clean Acc   : {clean_acc:.1%}")
    print()

    # --- FGSM Evasion ---
    print("-" * 60)
    print("  1. FGSM Evasion Attack (epsilon=0.3)")
    print("-" * 60)
    flipped = 0
    total_dist = 0.0
    for x, y in TRAINING_DATA:
        adv = fgsm_attack(model, x, y, epsilon=0.3)
        orig_pred = model.predict(x)
        adv_pred = model.predict(adv)
        dist = l2_distance(x, adv)
        total_dist += dist
        if adv_pred != orig_pred:
            flipped += 1
    fgsm_rate = flipped / len(TRAINING_DATA)
    avg_dist = total_dist / len(TRAINING_DATA)
    print(f"  Samples attacked     : {len(TRAINING_DATA)}")
    print(f"  Prediction flipped   : {flipped}/{len(TRAINING_DATA)} ({fgsm_rate:.1%})")
    print(f"  Avg perturbation L2  : {avg_dist:.4f}")
    print()

    # --- Gaussian Noise ---
    print("-" * 60)
    print("  2. Gaussian Noise Robustness")
    print("-" * 60)
    noise_levels = [0.1, 0.3, 0.5, 1.0]
    for sigma in noise_levels:
        noise_flipped = 0
        for x, y in TRAINING_DATA:
            for _ in range(10):
                noisy = gaussian_perturbation(x, sigma=sigma)
                if model.predict(noisy) != y:
                    noise_flipped += 1
        total_trials = len(TRAINING_DATA) * 10
        rate = noise_flipped / total_trials
        print(f"  sigma={sigma:.1f}: flip rate={rate:.1%} ({noise_flipped}/{total_trials})")
    print()

    # --- Min perturbation to flip ---
    print("-" * 60)
    print("  3. Minimum Perturbation to Flip (binary search)")
    print("-" * 60)
    min_dists = []
    for x, y in TRAINING_DATA:
        d = min_perturbation_to_flip(model, x, y)
        if d is not None:
            min_dists.append(d)
    if min_dists:
        print(f"  Samples where flip possible : {len(min_dists)}/{len(TRAINING_DATA)}")
        print(f"  Avg min epsilon              : {sum(min_dists)/len(min_dists):.4f}")
        print(f"  Min epsilon                  : {min(min_dists):.4f}")
        print(f"  Max epsilon                  : {max(min_dists):.4f}")
    else:
        print("  No samples could be flipped within max epsilon.")
    print()

    # --- Poisoning ---
    print("-" * 60)
    print("  4. Label-Flip Poisoning Assessment")
    print("-" * 60)
    poison_ratios = [0.1, 0.2, 0.3, 0.5]
    for ratio in poison_ratios:
        model_p = LogisticRegression(lr=0.5)
        poisoned = label_flip_attack(TRAINING_DATA, flip_ratio=ratio)
        model_p.train(poisoned, epochs=200)
        poisoned_acc = accuracy(model_p, TRAINING_DATA)
        drop = clean_acc - poisoned_acc
        print(f"  Poison {ratio:.0%}: acc={poisoned_acc:.1%} (drop={drop:+.1%})")
    print()

    # --- Summary ---
    print("=" * 60)
    print("  ROBUSTNESS SUMMARY")
    print("=" * 60)
    robustness_score = max(0, 1.0 - fgsm_rate) * 100
    print(f"  Clean Accuracy        : {clean_acc:.1%}")
    print(f"  FGSM Evasion Resilience: {1-fgsm_rate:.1%}")
    print(f"  Robustness Score       : {robustness_score:.0f}/100")
    if robustness_score >= 80:
        verdict = "ROBUST — Low evasion risk"
    elif robustness_score >= 50:
        verdict = "MODERATE — Some evasion vulnerability"
    else:
        verdict = "WEAK — Highly vulnerable to evasion"
    print(f"  Verdict                : {verdict}")
    print()
    print("=" * 60)
    print("  Audit complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_audit()
