"""seqmodel 的标准库 unittest 测试。

由 `python -m unittest discover` 默认发现（文件名匹配 test*.py）。
覆盖 sample-lstm 子命令的成功、确定性、零长度与失败契约。
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEQMODEL = os.path.join(_HERE, "seqmodel.py")


def _tiny_lstm_model():
    """构造一个合法的 version 2 LSTM 模型（八键、键序与形状合规）。"""
    vocab = ["a", "b", "c"]
    V = len(vocab)
    H = 2
    return {
        "version": 2,
        "vocab": vocab,
        "W": [[0.01 * (i + j + 1) for j in range(V + H)]
              for i in range(4 * H)],
        "b": [0.0] * (4 * H),
        "Why": [[0.05 * (k + j + 1) for j in range(H)] for k in range(V)],
        "by": [0.0] * V,
        "h0": [0.1, -0.2],
        "c0": [0.0, 0.0],
    }


def _run_cli(args):
    return subprocess.run(
        [sys.executable, _SEQMODEL] + args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class SampleLstmCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_path = os.path.join(self._tmp.name, "model.json")
        with open(self.model_path, "w", encoding="utf-8") as f:
            json.dump(_tiny_lstm_model(), f)

    def test_success_output_contract(self):
        result = _run_cli(["sample-lstm", self.model_path, "a", "1",
                           "1.0", "5"])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        text = result.stdout.decode("utf-8")
        self.assertTrue(text.endswith("\n"))
        body = text[:-1]
        self.assertEqual(len(body), 5)
        for ch in body:
            self.assertIn(ch, ["a", "b", "c"])

    def test_deterministic_byte_identical(self):
        args = ["sample-lstm", self.model_path, "b", "42", "0.7", "16"]
        first = _run_cli(args)
        second = _run_cli(args)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stderr, second.stderr)

    def test_zero_length_outputs_only_lf(self):
        result = _run_cli(["sample-lstm", self.model_path, "a", "0",
                           "2.5", "0"])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"\n")
        self.assertEqual(result.stderr, b"")

    def test_failure_contract(self):
        bad_argvs = [
            # 参数数量错误。
            ["sample-lstm", self.model_path, "a", "1", "1.0"],
            # START 不在词表内。
            ["sample-lstm", self.model_path, "z", "1", "1.0", "3"],
            # SEED 词法非法。
            ["sample-lstm", self.model_path, "a", "01", "1.0", "3"],
            # TEMPERATURE 非正。
            ["sample-lstm", self.model_path, "a", "1", "0", "3"],
            # LENGTH 词法非法。
            ["sample-lstm", self.model_path, "a", "1", "1.0", "-1"],
            # 模型文件不存在。
            ["sample-lstm", os.path.join(self._tmp.name, "nope.json"),
             "a", "1", "1.0", "3"],
        ]
        for argv in bad_argvs:
            with self.subTest(argv=argv):
                result = _run_cli(argv)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr, b"error\n")


if __name__ == "__main__":
    unittest.main()
