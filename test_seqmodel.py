"""seqmodel 的标准库 unittest 测试（python -m unittest discover 可发现）。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import seqmodel

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEQMODEL = os.path.join(_HERE, "seqmodel.py")


def _tiny_lstm_model():
    """构造一个合法的 version 2 LSTM 模型对象（vocab 2 项、H=2）。"""
    V, H = 2, 2
    return {
        "version": 2,
        "vocab": ["a", "b"],
        "W": [[0.01 * (k + j + 1) for j in range(V + H)]
              for k in range(4 * H)],
        "b": [0.0] * (4 * H),
        "Why": [[0.05, -0.02], [-0.03, 0.04]],
        "by": [0.0, 0.1],
        "h0": [0.0, 0.0],
        "c0": [0.0, 0.0],
    }


class LSTMCellTest(unittest.TestCase):
    def test_forward_deterministic_and_shapes(self):
        cell = seqmodel.LSTMCell(3, 2, seed=7)
        x = [1.0, 0.0, 0.0]
        h0 = [0.0, 0.0]
        c0 = [0.0, 0.0]
        h1, c1, _ = cell.forward(x, h0, c0)
        h2, c2, _ = cell.forward(x, h0, c0)
        self.assertEqual(h1, h2)
        self.assertEqual(c1, c2)
        self.assertEqual(len(h1), 2)
        self.assertEqual(len(c1), 2)
        self.assertTrue(all(type(v) is float for v in h1 + c1))

    def test_forward_rejects_bad_shape(self):
        cell = seqmodel.LSTMCell(3, 2, seed=0)
        with self.assertRaises(ValueError):
            cell.forward([1.0, 0.0], [0.0, 0.0], [0.0, 0.0])


class SampleLstmCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_path = os.path.join(self._tmp.name, "model.json")
        text = json.dumps(_tiny_lstm_model(), ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False) + "\n"
        with open(self.model_path, "wb") as f:
            f.write(text.encode("utf-8"))

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, _SEQMODEL, "sample-lstm", self.model_path,
             *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_sample_lstm_success_and_determinism(self):
        first = self._run("a", "0", "1.0", "5")
        second = self._run("a", "0", "1.0", "5")
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stderr, b"")
        self.assertEqual(first.stdout, second.stdout)
        self.assertTrue(first.stdout.endswith(b"\n"))
        self.assertEqual(len(first.stdout.decode("utf-8")), 6)

    def test_sample_lstm_zero_length_outputs_lf_only(self):
        result = self._run("a", "0", "1.0", "0")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"\n")
        self.assertEqual(result.stderr, b"")

    def test_sample_lstm_bad_seed_fails_with_error_line(self):
        result = self._run("a", "01", "1.0", "5")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_sample_lstm_bad_start_fails(self):
        result = self._run("z", "0", "1.0", "5")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_sample_lstm_missing_model_fails(self):
        result = subprocess.run(
            [sys.executable, _SEQMODEL, "sample-lstm",
             os.path.join(self._tmp.name, "nope.json"),
             "a", "0", "1.0", "5"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")


if __name__ == "__main__":
    unittest.main()
