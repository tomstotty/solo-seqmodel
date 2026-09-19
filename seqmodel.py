"""从零实现的序列建模工具（仅使用 Python 标准库）。

本模块提供 :class:`VanillaRNN`，一个纯 Python、无第三方依赖的
单层 Vanilla RNN，前向与反向传播结果均可复现。
"""

import math


def _is_f(x):
    """数值 F：类型恰为 int 或 float（排除 bool）且有限。"""
    return type(x) in (int, float) and math.isfinite(x)


def _is_f_vector(v, n):
    """v 是否为长度 n 的 F 列表。"""
    if type(v) is not list or len(v) != n:
        return False
    return all(_is_f(x) for x in v)


def _is_f_matrix(m, r, c):
    """m 是否为 r×c 的 F 列表（每行也必须是 list）。"""
    if type(m) is not list or len(m) != r:
        return False
    return all(type(row) is list and len(row) == c and all(_is_f(x) for x in row)
               for row in m)


class VanillaRNN:
    """单层 Vanilla RNN：h_t = tanh(Wxh·x_t + Whh·h_{t-1} + bh)。

    参数初始化全部为 ``0.0``：

    - ``Wxh``：H×I
    - ``Whh``：H×H
    - ``bh`` ：H
    """

    def __init__(self, I, H):
        if type(I) is not int or type(H) is not int or I <= 0 or H <= 0:
            raise ValueError("I and H must be positive ints")
        self.I = I
        self.H = H
        self.Wxh = [[0.0 for _ in range(I)] for _ in range(H)]
        self.Whh = [[0.0 for _ in range(H)] for _ in range(H)]
        self.bh = [0.0 for _ in range(H)]
        self._cache = None

    def forward(self, xs, h0=None):
        """前向传播。

        :param xs: 非空 T×I 的 F 列表。
        :param h0: None 或长度 H 的 F 列表；None 等价于零向量。
        :return: T×H 的隐藏状态列表（list[list[float]]），并保存反向所需缓存。
        """
        I, H = self.I, self.H
        if not _is_f_matrix(xs, len(xs) if type(xs) is list else -1, I) or not xs:
            raise ValueError("xs must be a non-empty T x I list of finite numbers")
        if h0 is not None and not _is_f_vector(h0, H):
            raise ValueError("h0 must be None or an H-length list of finite numbers")

        T = len(xs)
        h_prev = [0.0 for _ in range(H)] if h0 is None else [float(v) for v in h0]
        hs = [h_prev]
        for t in range(T):
            x_t = xs[t]
            h_t = []
            for h in range(H):
                z = self.bh[h]
                for j in range(H):
                    z += self.Whh[h][j] * h_prev[j]
                for j in range(I):
                    z += self.Wxh[h][j] * x_t[j]
                h_t.append(math.tanh(z))
            hs.append(h_t)
            h_prev = h_t

        self._cache = (xs, hs, T)
        return [row[:] for row in hs[1:]]

    def backward(self, dhs):
        """反向传播（BPTT）。

        :param dhs: 与缓存形状一致的 T×H F 列表。
        :return: ``(dxs, dWxh, dWhh, dbh, dh0)``，形状分别为
                 T×I、H×I、H×H、H、H。
        :raises ValueError: 此前未成功执行过 forward，或 dhs 形状/数值非法。
        """
        if self._cache is None:
            raise ValueError("backward called before a successful forward")
        xs, hs, T = self._cache
        I, H = self.I, self.H
        if not _is_f_matrix(dhs, T, H):
            raise ValueError("dhs must be a T x H list of finite numbers matching the cache")

        dxs = [[0.0 for _ in range(I)] for _ in range(T)]
        dWxh = [[0.0 for _ in range(I)] for _ in range(H)]
        dWhh = [[0.0 for _ in range(H)] for _ in range(H)]
        dbh = [0.0 for _ in range(H)]
        dh_next = [0.0 for _ in range(H)]

        for t in range(T - 1, -1, -1):
            h_t = hs[t + 1]
            h_prev = hs[t]
            x_t = xs[t]
            dt = [
                (1.0 - h_t[h] * h_t[h]) * (dhs[t][h] + dh_next[h])
                for h in range(H)
            ]
            # dx_t = Wxh^T dt
            dxs[t] = [
                sum(self.Wxh[h][i] * dt[h] for h in range(H))
                for i in range(I)
            ]
            for h in range(H):
                dth = dt[h]
                for i in range(I):
                    dWxh[h][i] += dth * x_t[i]
                for j in range(H):
                    dWhh[h][j] += dth * h_prev[j]
                dbh[h] += dth
            # 流向 h_{t-1} 的梯度：dh_next = Whh^T dt
            dh_next = [
                sum(self.Whh[h][j] * dt[h] for h in range(H))
                for j in range(H)
            ]

        dh0 = dh_next
        return dxs, dWxh, dWhh, dbh, dh0
