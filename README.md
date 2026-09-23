> **⚠️ EDUCATIONAL USE ONLY — AUTHORIZED TESTING ONLY.**
> This project exists for education, research, and **defense of systems you own
> or hold explicit written authorization to assess**. Unauthorized use is
> prohibited and may be illegal. Read [ETHICS.md](ETHICS.md) and
> [SCOPE.md](SCOPE.md) before use. Use at your own risk; **AS IS**, no warranty.

# AI7 — Model Robustness Auditor

Adversarial machine-learning robustness evaluation suite for binary classifiers — trains a
logistic-regression model from scratch, then runs FGSM evasion, Gaussian-noise perturbation,
minimum-perturbation search, and label-flip poisoning benchmarks to produce a composite robustness
score. Pure Python, offline, no orchestration of external model services.

![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![GitHub stars](https://img.shields.io/github/stars/5h4d0wn1k/ai7-model-robustness)
![GitHub last commit](https://img.shields.io/github/last-commit/5h4d0wn1k/ai7-model-robustness)
![GitHub issues](https://img.shields.io/github/issues/5h4d0wn1k/ai7-model-robustness)

## Why

ML classifiers are attacked before they are deployed: adversaries add imperceptible noise (FGSM),
or poison training data. Measuring that risk is the first defensive step, and it does not require a
GPU farm. AI7 implements the canonical adversarial robustness benchmarks from scratch — a
hand-rolled logistic-regression classifier, a finite-difference black-box FGSM gradient, binary
search for the minimum perturbation that flips a prediction, and a label-flip poisoning curve —
then distills everything into a 0-100 composite robustness score with a WEAK/MODERATE/STRONG
verdict. It is an ai-security and adversarial-ML educational instrument for auditing models you
train or own. Auditing production third-party models without authorization is prohibited.

## Features

- **From-scratch training** — `LogisticRegression.train/predict/predict_proba` on an embedded
  synthetic 2D dataset (no sklearn/numpy required).
- **FGSM evasion** — finite-difference gradient attack with flip rate + average L2 perturbation.
- **Noise robustness** — per-sigma flip-rate degradation curve.
- **Minimum perturbation** — binary search for the smallest epsilon that flips a prediction.
- **Label-flip poisoning** — accuracy drop across poisoning ratios.
- **Composite score** — `robustness_score` (0-100) and verdict in a structured JSON audit.

## Quickstart

Prerequisite: Python 3 (standard library only).

```bash
python3 firmware/model_robustness.py                    # offline audit, exit 0
python3 firmware/model_robustness.py --seed 42 --epsilon 0.3 --epochs 200
python3 firmware/model_robustness.py --output reports/ai7-report.json
python3 firmware/model_robustness.py --quiet --output reports/ai7-report.json
```

Library use:

```python
from firmware.model_robustness import LogisticRegression, fgsm_attack
model = LogisticRegression(lr=0.5); model.train(TRAINING_DATA, epochs=200)
adv = fgsm_attack(model, x=[1.5, 2.0], y_true=0, epsilon=0.3)
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Project structure

- `firmware/model_robustness.py` — model, attacks, and audit CLI.
- `tests/` — determinism, accuracy bounds, attack-shape, poisoning, and JSON tests.

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [ETHICS.md](ETHICS.md) · [SCOPE.md](SCOPE.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep the suite dependency-free and deterministic.

## License

MIT — see [LICENSE](LICENSE).