"""seqmodel: 从零实现的序列建模库（仅 Python 标准库，离线）。

本模块提供确定性的 VanillaRNN：

    h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)

数值约定：标量类型 F 指 type(x) 为 int 或 float、非 bool 且 math.isfinite。
任何非 F 元素或形状错误都抛出 ValueError；参数个数错误沿用 Python 自带的
TypeError。
"""

import math


def _is_f(value):
    """F 判定：int/f.float（不含 bool）且有限。"""
    return (type(value) is int or type(value) is float) and math.isfinite(value)


def _check_matrix(values, rows, cols, name):
    """校验 values 为 rows×cols 的 F 列表的列表，返回逐行浅拷贝。"""
    if type(values) is not list or len(values) != rows:
        raise ValueError("%s must be a list of shape %d×%d" % (name, rows, cols))
    checked = []
    for row in values:
        if type(row) is not list or len(row) != cols:
            raise ValueError("%s must be a list of shape %d×%d"
                             % (name, rows, cols))
        for v in row:
            if not _is_f(v):
                raise ValueError("%s entries must be finite numbers, got %r"
                                 % (name, v))
        checked.append(list(row))
    return checked


def _check_vector(values, size, name):
    """校验 values 为长度 size 的 F 列表，返回浅拷贝。"""
    if type(values) is not list or len(values) != size:
        raise ValueError("%s must be a list of length %d" % (name, size))
    for v in values:
        if not _is_f(v):
            raise ValueError("%s entries must be finite numbers, got %r"
                             % (name, v))
    return list(values)


class VanillaRNN(object):
    """单隐藏层 Vanilla RNN，参数 Wxh/Whh/bh 初始化为全 0.0。

    Wxh 形状 H×I，Whh 形状 H×H，bh 形状 H。
    """

    def __init__(self, I, H):
        if type(I) is not int or type(H) is not int or I <= 0 or H <= 0:
            raise ValueError("I and H must be positive integers")
        self.I = I
        self.H = H
        self.Wxh = [[0.0] * I for _ in range(H)]
        self.Whh = [[0.0] * H for _ in range(H)]
        self.bh = [0.0] * H
        self._cache = None

    def forward(self, xs, h0=None):
        """前向传播，返回 T×H 的隐状态列表并在内部保存缓存。

        xs 须为非空的 T×I 的 F 列表；h0 为 None 或长度 H 的 F 列表，
        None 等价于全零初始隐状态。
        """
        if type(xs) is not list or len(xs) == 0:
            raise ValueError("xs must be a non-empty list")
        xs = _check_matrix(xs, len(xs), self.I, "xs")
        T = len(xs)
        if h0 is None:
            h_prev = [0.0] * self.H
        else:
            h_prev = _check_vector(h0, self.H, "h0")
        h0_cache = list(h_prev)

        I, H = self.I, self.H
        Wxh, Whh, bh = self.Wxh, self.Whh, self.bh
        hs = []
        for t in range(T):
            x = xs[t]
            h = [0.0] * H
            for i in range(H):
                acc = bh[i]
                wx_row = Wxh[i]
                for j in range(I):
                    acc += wx_row[j] * x[j]
                wh_row = Whh[i]
                for j in range(H):
                    acc += wh_row[j] * h_prev[j]
                h[i] = math.tanh(acc)
            hs.append(h)
            h_prev = h

        self._cache = {"xs": xs, "h0": h0_cache, "hs": hs}
        return [list(row) for row in hs]

    def backward(self, dhs):
        """沿时间反向传播，返回 (dxs, dWxh, dWhh, dbh, dh0)。

        须先成功调用过 forward；dhs 形状须与缓存的 T×H 一致且元素均为 F。
        五个返回值形状依次为 T×I、H×I、H×H、H、H。
        """
        cache = self._cache
        if cache is None:
            raise ValueError("backward requires a successful forward call first")
        xs, hs, h0 = cache["xs"], cache["hs"], cache["h0"]
        T, I, H = len(xs), self.I, self.H
        dhs = _check_matrix(dhs, T, H, "dhs")

        Wxh, Whh = self.Wxh, self.Whh
        dxs = [[0.0] * I for _ in range(T)]
        dWxh = [[0.0] * I for _ in range(H)]
        dWhh = [[0.0] * H for _ in range(H)]
        dbh = [0.0] * H
        dh_next = [0.0] * H

        for t in range(T - 1, -1, -1):
            h = hs[t]
            h_prev = hs[t - 1] if t > 0 else h0
            x = xs[t]

            # dt = (1 - h_t^2) ⊙ (dhs[t] + dh_next)
            dt = [0.0] * H
            for i in range(H):
                dt[i] = (1.0 - h[i] * h[i]) * (dhs[t][i] + dh_next[i])

            # dx_t = Wxhᵀ dt
            dx_row = dxs[t]
            for j in range(I):
                acc = 0.0
                for i in range(H):
                    acc += Wxh[i][j] * dt[i]
                dx_row[j] = acc

            # dbh += dt；dWxh += dt ⊗ x_t
            for i in range(H):
                di = dt[i]
                dbh[i] += di
                gw_row = dWxh[i]
                for j in range(I):
                    gw_row[j] += di * x[j]

            # dWhh += dt ⊗ h_{t-1}
            for i in range(H):
                di = dt[i]
                gw_row = dWhh[i]
                for j in range(H):
                    gw_row[j] += di * h_prev[j]

            # 传给 h_{t-1} 的梯度：dh_next = Whhᵀ dt
            new_dh = [0.0] * H
            for j in range(H):
                acc = 0.0
                for i in range(H):
                    acc += Whh[i][j] * dt[i]
                new_dh[j] = acc
            dh_next = new_dh

        dh0 = dh_next
        return dxs, dWxh, dWhh, dbh, dh0
