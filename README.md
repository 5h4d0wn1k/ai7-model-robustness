# AI7 — Model Robustness Auditor

Pure-Python adversarial robustness testing toolkit for binary classifiers.

## Overview

- Trains a logistic-regression classifier from scratch on an embedded synthetic 2D dataset
- Tests adversarial evasion via FGSM (Fast Gradient Sign Method)
- Measures robustness against Gaussian noise perturbations
- Evaluates minimum perturbation magnitude required to flip predictions
- Assesses impact of label-flip data poisoning at various ratios
- Produces a structured audit report with a composite robustness score

## Features

- **FGSM Evasion Attack**: Finite-difference gradient approximation for black-box models
- **Gaussian Noise Testing**: Multi-sigma robustness degradation curve
- **Minimum Perturbation Search**: Binary search for smallest perturbation radius that flips a prediction
- **Label-Flip Poisoning**: Accuracy degradation under controlled data poisoning
- **Robustness Score**: Composite metric combining evasion resilience and clean accuracy
- **Zero Dependencies**: Pure Python standard library implementation

## Installation

No external dependencies required — uses Python standard library only.

```bash
python3 firmware/model_robustness.py
```

## Usage

```python
from firmware.model_robustness import LogisticRegression, fgsm_attack, accuracy

# Train model
model = LogisticRegression(lr=0.5)
model.train(TRAINING_DATA, epochs=200)

# Test a single adversarial example
adv = fgsm_attack(model, x=[1.5, 2.0], y_true=0, epsilon=0.3)
print(model.predict(adv))
```

## Example Output

```
============================================================
  AI7 — Model Robustness Audit Report
============================================================

Dataset     : 20 samples, 2 classes, 2 features
Model       : Logistic Regression (lr=0.5, 200 epochs)
Clean Acc   : 100.0%

------------------------------------------------------------
  1. FGSM Evasion Attack (epsilon=0.3)
------------------------------------------------------------
  Samples attacked     : 20
  Prediction flipped   : 20/20 (100.0%)

------------------------------------------------------------
  ROBUSTNESS SUMMARY
------------------------------------------------------------
  Clean Accuracy         : 100.0%
  FGSM Evasion Resilience: 0.0%
  Robustness Score       : 0/100
  Verdict                : WEAK — Highly vulnerable to evasion
```

## IMPORTANT: Read before use.

This tool is provided **exclusively** for authorized security research, academic study, and defensive hardening. Use without explicit written authorization is illegal and unethical.

### Authorization Requirements

You must obtain explicit written permission from the system owner before running any robustness audit against models or systems you do not own. Auditing models in production without authorization may violate service agreements and applicable law.

### Legal Framework

Unauthorized access to or manipulation of computer systems is governed by the **Computer Fraud and Abuse Act (CFAA)** (18 U.S.C. § 1030), the **EU Directive on Attacks Against Information Systems** (2013/40/EU), and equivalent legislation in other jurisdictions. Penalties include imprisonment and significant fines.

### Acceptable Use

- Authorized red-team and penetration testing engagements
- Academic research on adversarial machine learning
- Defensive security audits of your own models and infrastructure
- CTF competitions and educational lab environments

### Prohibited Use

- Attacking models or systems without written authorization
- Using adversarial examples to bypass production security controls
- Poisoning training data of systems you do not own
- Any use that violates applicable law or terms of service

### No Warranty

This software is provided "as is" without warranty of any kind. The authors assume no liability for damages arising from use or misuse of this tool.

### Responsible Disclosure

If you discover vulnerabilities in third-party models using this tool, follow coordinated disclosure practices. Report to the vendor directly and allow reasonable time for remediation before public disclosure.

## License

MIT License
