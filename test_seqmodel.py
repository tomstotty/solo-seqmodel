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


def _tiny_lstm_mha_relative_model():
    """构造一个合法的 version 5 相对 MHA 模型对象（vocab 2、H=2、2 头、R=1）。"""
    V, H = 2, 2
    eye = [[1.0 if a == j else 0.0 for j in range(H)] for a in range(H)]
    return {
        "version": 5,
        "vocab": ["a", "b"],
        "W": [[0.01 * (k + j + 1) for j in range(V + H)]
              for k in range(4 * H)],
        "b": [0.0] * (4 * H),
        "Wq": [list(row) for row in eye],
        "Wk": [list(row) for row in eye],
        "Wv": [list(row) for row in eye],
        "Wo": [list(row) for row in eye],
        "heads": 2,
        "bias": [[0.1, 0.0, -0.1], [0.0, 0.2, 0.0]],
        "Why": [[0.05, -0.02], [-0.03, 0.04]],
        "by": [0.0, 0.1],
        "h0": [0.0, 0.0],
        "c0": [0.0, 0.0],
    }


class SampleLstmMhaRelativeTopKCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_path = os.path.join(self._tmp.name, "model.json")
        text = json.dumps(_tiny_lstm_mha_relative_model(), ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False) + "\n"
        with open(self.model_path, "wb") as f:
            f.write(text.encode("utf-8"))

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, _SEQMODEL, "sample-lstm-mha-relative-top-k",
             self.model_path, *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_success_determinism_and_shape(self):
        first = self._run("a", "0", "1.0", "0.5", "1", "5", "4")
        second = self._run("a", "0", "1.0", "0.5", "1", "5", "4")
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stderr, b"")
        self.assertEqual(first.stdout, second.stdout)
        self.assertTrue(first.stdout.endswith(b"\n"))
        self.assertEqual(len(first.stdout.decode("utf-8")), 6)

    def test_zero_length_outputs_lf_only(self):
        result = self._run("a", "0", "1.0", "0.5", "1", "0", "4")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"\n")
        self.assertEqual(result.stderr, b"")

    def test_top_k_equal_vocab_size_succeeds(self):
        result = self._run("a", "0", "1.0", "0.5", "2", "5", "4")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(len(result.stdout.decode("utf-8")), 6)

    def test_top_k_above_vocab_size_fails(self):
        result = self._run("a", "0", "1.0", "0.5", "3", "5", "4")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_top_k_huge_digit_count_fails(self):
        result = self._run("a", "0", "1.0", "0.5", "9" * 400, "5", "4")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_bad_top_k_lexicon_fails(self):
        for top_k in ("0", "01", "1.0", "-1", "1_0", "+1"):
            result = self._run("a", "0", "1.0", "0.5", top_k, "5", "4")
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, b"")
            self.assertEqual(result.stderr, b"error\n")

    def test_bad_window_fails(self):
        result = self._run("a", "0", "1.0", "0.5", "1", "5", "0")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_missing_model_fails(self):
        result = subprocess.run(
            [sys.executable, _SEQMODEL, "sample-lstm-mha-relative-top-k",
             os.path.join(self._tmp.name, "nope.json"),
             "a", "0", "1.0", "0.5", "1", "5", "4"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")


class MhaRelativeBucketStatsCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_path = os.path.join(self._tmp.name, "model.json")
        text = json.dumps(_tiny_lstm_mha_relative_model(), ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False) + "\n"
        with open(self.model_path, "wb") as f:
            f.write(text.encode("utf-8"))
        self.corpus_path = os.path.join(self._tmp.name, "corpus.txt")
        with open(self.corpus_path, "wb") as f:
            f.write("abbaab".encode("utf-8"))

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, _SEQMODEL, "mha-relative-bucket-stats",
             self.model_path, self.corpus_path, *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_success_structure_and_determinism(self):
        first = self._run("4")
        second = self._run("4")
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stderr, b"")
        self.assertEqual(first.stdout, second.stdout)
        self.assertTrue(first.stdout.endswith(b"\n"))
        obj = json.loads(first.stdout.decode("utf-8"))
        self.assertEqual(list(obj.keys()),
                         ["version", "radius", "steps", "items", "mean"])
        self.assertEqual(obj["version"], 1)
        self.assertEqual(obj["radius"], 1)
        self.assertEqual(obj["steps"], 5)
        self.assertEqual(len(obj["items"]), 5)
        for t, rows in enumerate(obj["items"]):
            self.assertEqual(rows[0], t)
            self.assertEqual([row[0] for row in rows[1]], [0, 1])
            for _r, masses in rows[1]:
                self.assertEqual(len(masses), 3)
                vals = [float(v) for v in masses]
                self.assertAlmostEqual(sum(vals), 1.0, places=12)
        self.assertEqual([row[0] for row in obj["mean"]], [0, 1])
        for r in range(2):
            totals = [0.0] * 3
            for _t, rows in obj["items"]:
                for bj, v in enumerate(rows[r][1]):
                    totals[bj] += float(v)
            for bj in range(3):
                self.assertAlmostEqual(float(obj["mean"][r][1][bj]),
                                       totals[bj] / 5, places=12)

    def test_bad_window_fails_with_error_line(self):
        result = self._run("0")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_out_of_vocab_corpus_fails(self):
        with open(self.corpus_path, "wb") as f:
            f.write("abzab".encode("utf-8"))
        result = self._run("4")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_missing_model_fails(self):
        result = subprocess.run(
            [sys.executable, _SEQMODEL, "mha-relative-bucket-stats",
             os.path.join(self._tmp.name, "nope.json"), self.corpus_path,
             "4"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")


def _tiny_transformer_model():
    """构造一个合法的 version 6 单层 Transformer 模型对象（V=2、2 头、P=3）。"""
    V, P = 2, 3
    eye = [[1.0 if a == j else 0.0 for j in range(V)] for a in range(V)]
    return {
        "version": 6,
        "vocab": ["a", "b"],
        "heads": 2,
        "P": P,
        "Wq": [[0.1 * (2 * a + j + 1) - 0.03 for j in range(V)]
              for a in range(V)],
        "Wk": [[-0.07 + 0.05 * (a + j) for j in range(V)]
              for a in range(V)],
        "Wv": eye,
        "Wo": [[0.04 * (j - a) + 0.1 for j in range(V)]
              for a in range(V)],
        "W1": [[0.02 * (p + 2 * j) - 0.05 for j in range(V)]
              for p in range(P)],
        "b1": [0.01 * p - 0.02 for p in range(P)],
        "W2": [[0.03 * (j + p) - 0.04 for p in range(P)]
              for j in range(V)],
        "b2": [0.0, 0.1],
        "Why": [[0.05, -0.02], [-0.03, 0.04]],
        "by": [0.0, 0.1],
    }


class PerplexityTransformerWindowCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_path = os.path.join(self._tmp.name, "model.json")
        text = json.dumps(_tiny_transformer_model(), ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False) + "\n"
        with open(self.model_path, "wb") as f:
            f.write(text.encode("utf-8"))
        self.corpus_path = os.path.join(self._tmp.name, "corpus.txt")
        with open(self.corpus_path, "wb") as f:
            f.write("abbaabab".encode("utf-8"))

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, _SEQMODEL, "perplexity-transformer-window",
             self.model_path, self.corpus_path, *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _run_full(self):
        return subprocess.run(
            [sys.executable, _SEQMODEL, "perplexity-transformer",
             self.model_path, self.corpus_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_large_window_byte_identical_to_full_context(self):
        # T = 语料码点数 - 1 = 7；WINDOW=7（=T）与更大窗口都须与
        # perplexity-transformer 逐字节相同。
        full = self._run_full()
        self.assertEqual(full.returncode, 0)
        self.assertEqual(full.stderr, b"")
        for window in ("7", "8", "100", "9" * 80):
            result = self._run(window)
            self.assertEqual(result.returncode, 0, window)
            self.assertEqual(result.stderr, b"", window)
            self.assertEqual(result.stdout, full.stdout, window)
            self.assertTrue(result.stdout.endswith(b"\n"))

    def test_smaller_windows_are_deterministic_and_truncated(self):
        values = {}
        for window in ("1", "2", "3", "6"):
            first = self._run(window)
            second = self._run(window)
            self.assertEqual(first.returncode, 0, window)
            self.assertEqual(first.stderr, b"", window)
            self.assertEqual(first.stdout, second.stdout)
            self.assertTrue(first.stdout.endswith(b"\n"))
            # 恰为一个 .17g 数加 LF。
            self.assertEqual(first.stdout.count(b"\n"), 1)
            values[window] = first.stdout
        # 窗口 1 与窗口 7（满上下文）结果应不同：截断确实改变了预测。
        full = self._run_full()
        self.assertNotEqual(values["1"], full.stdout)
        self.assertNotEqual(values["2"], full.stdout)
        self.assertNotEqual(values["3"], full.stdout)
        self.assertNotEqual(values["6"], full.stdout)

    def test_bad_window_lexicon_fails(self):
        for window in ("0", "01", "1.0", "-1", "+1", "1_0", "", "a"):
            result = self._run(window)
            self.assertEqual(result.returncode, 2, window)
            self.assertEqual(result.stdout, b"", window)
            self.assertEqual(result.stderr, b"error\n", window)

    def test_out_of_vocab_corpus_fails(self):
        with open(self.corpus_path, "wb") as f:
            f.write("abzab".encode("utf-8"))
        result = self._run("3")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_too_short_corpus_fails(self):
        with open(self.corpus_path, "wb") as f:
            f.write("a".encode("utf-8"))
        result = self._run("3")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_missing_model_fails(self):
        result = subprocess.run(
            [sys.executable, _SEQMODEL, "perplexity-transformer-window",
             os.path.join(self._tmp.name, "nope.json"), self.corpus_path,
             "3"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")

    def test_wrong_argc_fails(self):
        result = subprocess.run(
            [sys.executable, _SEQMODEL, "perplexity-transformer-window",
             self.model_path, self.corpus_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"error\n")


if __name__ == "__main__":
    unittest.main()