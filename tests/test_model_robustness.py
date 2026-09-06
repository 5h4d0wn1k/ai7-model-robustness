"""AI7 model-robustness engine tests — real code paths, offline, pure stdlib."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "firmware"))

from model_robustness import (  # noqa: E402
    LogisticRegression,
    TRAINING_DATA,
    accuracy,
    fgsm_attack,
    label_flip_attack,
    min_perturbation_to_flip,
    run_experiment,
)


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegression(lr=0.5)
        self.model.train(TRAINING_DATA, epochs=50)

    def test_predict_binary(self):
        for x, _ in TRAINING_DATA[:5]:
            self.assertIn(self.model.predict(x), (0, 1))

    def test_accuracy_in_bounds(self):
        acc = accuracy(self.model, TRAINING_DATA)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_fgsm_changes_or_keeps_shape(self):
        x, y = TRAINING_DATA[0]
        adv = fgsm_attack(self.model, x, y, epsilon=0.3)
        self.assertEqual(len(adv), len(x))

    def test_label_flip_attacks_data(self):
        poisoned = label_flip_attack(TRAINING_DATA, flip_ratio=0.5)
        flipped = sum(1 for (_, y1), (_, y2) in zip(TRAINING_DATA, poisoned)
                      if y1 != y2)
        self.assertGreater(flipped, 0)
        self.assertEqual(len(poisoned), len(TRAINING_DATA))

    def test_min_perturbation_returns_value_for_correct_sample(self):
        model = LogisticRegression(lr=0.5)
        model.train(TRAINING_DATA, epochs=200)
        for x, y in TRAINING_DATA[:3]:
            if model.predict(x) == y:
                d = min_perturbation_to_flip(model, x, y)
                self.assertIsNotNone(d)
                self.assertGreater(d, 0.0)


class TestRunExperiment(unittest.TestCase):
    def test_returns_structured_results(self):
        r = run_experiment(seed=1, epsilon=0.3, epochs=50)
        self.assertIn("dataset", r)
        self.assertIn("model", r)
        self.assertIn("fgsm_evasion", r)
        self.assertIn("noise_robustness", r)
        self.assertIn("min_perturbation", r)
        self.assertIn("poisoning", r)
        self.assertIn("score", r)
        self.assertIn("robustness_score", r["score"])

    def test_flip_rate_in_bounds(self):
        r = run_experiment(seed=1, epsilon=0.3, epochs=50)
        self.assertGreaterEqual(r["fgsm_evasion"]["flip_rate"], 0.0)
        self.assertLessEqual(r["fgsm_evasion"]["flip_rate"], 1.0)

    def test_deterministic_with_seed(self):
        a = run_experiment(seed=5, epochs=50)
        b = run_experiment(seed=5, epochs=50)
        self.assertEqual(a["model"]["clean_accuracy"],
                         b["model"]["clean_accuracy"])


class TestCLI(unittest.TestCase):
    def test_cli_writes_json_report_and_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.json")
            from model_robustness import main
            code = main(["--seed", "1", "--epochs", "20",
                         "--output", out, "--quiet"])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("score", data)


if __name__ == "__main__":
    unittest.main()