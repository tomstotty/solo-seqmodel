"""seqmodel: 从零实现的序列建模库（仅 Python 标准库，离线）。

本模块提供确定性的 VanillaRNN 与 LSTMCell：

    h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)

数值约定：标量类型 F 指 type(x) 为 int 或 float（不含 bool）、float(x)
可成功转换且 math.isfinite 为真；超大整数因无法转换成有限 float 而非 F。
任何非 F 元素或形状错误都抛出 ValueError；参数个数错误沿用 Python 自带的
TypeError。
"""

import math
import random


def _is_f(value):
    """F 判定：type 为 int/float（不含 bool）、float(v) 成功且有限。

    超大整数（如 10**400）无法转换为有限 float，视为非 F。
    """
    if type(value) is not int and type(value) is not float:
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


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


def clip_gradients(dWxh, dWhh, dbh, max_norm):
    """对 RNN 梯度做全局范数裁剪，返回四元组而不修改任何输入。

    dbh 须为非空 F 列表，其长度确定 H；dWxh 须为恰有 H 行的列表，
    每行均为长度相同且大于零的 F 列表，首行长度确定 I；dWhh 须为
    H×H 的 F 列表；max_norm 须为正数 F。任一校验失败抛 ValueError。

    按 dWxh 逐行、dWhh 逐行、dbh 的顺序以 float 累加平方和：
        global_norm = sqrt(sum_sq)
    若 global_norm > max_norm，三组元素统一乘 max_norm / global_norm；
    否则缩放因子为 1.0（零范数也不除零）。返回
        (clipped_dWxh, clipped_dWhh, clipped_dbh, global_norm)
    global_norm 为裁剪前的 float 范数；前三项保持原形状、元素均为 float，
    且逐层新建列表，不修改或复用输入列表。
    """
    def _is_grad_f(value):
        # 与 F 一致：int/float（不含 bool）且可转换为有限 float；
        # 超大整数（float 转换溢出）一律拒绝。
        if type(value) is not int and type(value) is not float:
            return False
        try:
            return math.isfinite(float(value))
        except OverflowError:
            return False

    def _require_f(value, name):
        if not _is_grad_f(value):
            raise ValueError("%s entries must be finite numbers, got %r"
                             % (name, value))

    if type(dbh) is not list or len(dbh) == 0:
        raise ValueError("dbh must be a non-empty list")
    H = len(dbh)
    for v in dbh:
        _require_f(v, "dbh")

    if type(dWxh) is not list or len(dWxh) != H:
        raise ValueError("dWxh must be a list with exactly %d rows" % H)
    first_row = dWxh[0]
    if type(first_row) is not list or len(first_row) == 0:
        raise ValueError("dWxh rows must be non-empty lists")
    I = len(first_row)
    for v in first_row:
        _require_f(v, "dWxh")
    for row in dWxh[1:]:
        if type(row) is not list or len(row) != I:
            raise ValueError("dWxh must be a rectangular list of shape %d×%d"
                             % (H, I))
        for v in row:
            _require_f(v, "dWxh")

    if type(dWhh) is not list or len(dWhh) != H:
        raise ValueError("dWhh must be a list of shape %d×%d" % (H, H))
    for row in dWhh:
        if type(row) is not list or len(row) != H:
            raise ValueError("dWhh must be a list of shape %d×%d" % (H, H))
        for v in row:
            _require_f(v, "dWhh")

    if not _is_grad_f(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be a finite positive number, got %r"
                         % (max_norm,))

    sum_sq = 0.0
    try:
        for matrix in (dWxh, dWhh):
            for row in matrix:
                for v in row:
                    fv = float(v)
                    sum_sq += fv * fv
                    if not math.isfinite(sum_sq):
                        raise ValueError(
                            "global norm accumulated to a non-finite value")
        for v in dbh:
            fv = float(v)
            sum_sq += fv * fv
            if not math.isfinite(sum_sq):
                raise ValueError(
                    "global norm accumulated to a non-finite value")
    except OverflowError:
        raise ValueError("global norm accumulated to a non-finite value")

    global_norm = math.sqrt(sum_sq)
    scale = max_norm / global_norm if global_norm > max_norm else 1.0

    clipped_dWxh = [[float(v) * scale for v in row] for row in dWxh]
    clipped_dWhh = [[float(v) * scale for v in row] for row in dWhh]
    clipped_dbh = [float(v) * scale for v in dbh]
    return clipped_dWxh, clipped_dWhh, clipped_dbh, global_norm


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


class LSTMCell(object):
    """单步 LSTM 单元，参数 W/b 由 seed 确定性初始化。

    W 形状 4H×(I+H)，b 形状 4H。以 r = random.Random(seed)、
    q = 1/sqrt(I+H)，按 W 行列序令每项为 float((2*r.random()-1)*q)，
    b 初始化为全 0.0。
    """

    def __init__(self, I, H, seed=0):
        if type(I) is not int or type(H) is not int or I <= 0 or H <= 0:
            raise ValueError("I and H must be positive integers")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.I = I
        self.H = H
        r = random.Random(seed)
        q = 1.0 / math.sqrt(I + H)
        self.W = [[float((2 * r.random() - 1) * q)
                   for _ in range(I + H)] for _ in range(4 * H)]
        self.b = [0.0] * (4 * H)

    @staticmethod
    def _sigmoid(v):
        """数值稳定的 sigmoid。"""
        if v >= 0:
            return 1.0 / (1.0 + math.exp(-v))
        ev = math.exp(v)
        return ev / (1.0 + ev)

    def forward(self, x, h_prev, c_prev):
        """单步前向传播，返回 (h, c, cache)。

        x、h_prev、c_prev 须分别为长度 I、H、H 的 F 列表；self.W、self.b
        须仍满足 4H×(I+H)、4H 的形状且元素均为 F，否则抛 ValueError。
        中间量出现非有限值同样抛 ValueError。返回的 h、c 为新的 float
        列表；cache 为 dict，键按 x、h_prev、c_prev、z、i、f、o、g、c、
        h、W 顺序插入，向量值均为新的 float 列表，W 为二维 float 深拷贝；
        返回对象不复用或修改输入及属性。
        """
        I, H = self.I, self.H
        x = _check_vector(x, I, "x")
        h_prev = _check_vector(h_prev, H, "h_prev")
        c_prev = _check_vector(c_prev, H, "c_prev")
        W = _check_matrix(self.W, 4 * H, I + H, "W")
        b = _check_vector(self.b, 4 * H, "b")

        z = x + h_prev

        # a[k] = b[k] + Σ_j W[k][j]*z[j]，按列序累加
        a = [0.0] * (4 * H)
        for k in range(4 * H):
            acc = float(b[k])
            row = W[k]
            for j in range(I + H):
                acc += float(row[j]) * float(z[j])
            if not math.isfinite(acc):
                raise ValueError("pre-activation accumulated to a "
                                 "non-finite value")
            a[k] = acc

        # 连续 H 段依次解释为 i、f、o、g
        i_gate = [self._sigmoid(a[k]) for k in range(0, H)]
        f_gate = [self._sigmoid(a[k]) for k in range(H, 2 * H)]
        o_gate = [self._sigmoid(a[k]) for k in range(2 * H, 3 * H)]
        g_gate = [math.tanh(a[k]) for k in range(3 * H, 4 * H)]

        c = [0.0] * H
        h = [0.0] * H
        for k in range(H):
            cv = f_gate[k] * float(c_prev[k]) + i_gate[k] * g_gate[k]
            if not math.isfinite(cv):
                raise ValueError("cell state became non-finite")
            c[k] = cv
            hv = o_gate[k] * math.tanh(cv)
            if not math.isfinite(hv):
                raise ValueError("hidden state became non-finite")
            h[k] = hv

        cache = {
            "x": [float(v) for v in x],
            "h_prev": [float(v) for v in h_prev],
            "c_prev": [float(v) for v in c_prev],
            "z": [float(v) for v in z],
            "i": [float(v) for v in i_gate],
            "f": [float(v) for v in f_gate],
            "o": [float(v) for v in o_gate],
            "g": [float(v) for v in g_gate],
            "c": [float(v) for v in c],
            "h": [float(v) for v in h],
            "W": [[float(v) for v in row] for row in W],
        }
        return [float(v) for v in h], [float(v) for v in c], cache

    def _check_forward_cache(self, cache):
        """校验单个 cache 严格符合 forward 的公开缓存契约。

        须为 dict，恰含 x、h_prev、c_prev、z、i、f、o、g、c、h、W 共 11 个
        有序键（多、少、乱序均非法），其中 x 长 I，h_prev、c_prev、i、f、o、
        g、c、h 长 H，z 长 I+H，W 为 4H×(I+H)，元素均为 F。任一容器、键序、
        形状或元素非法抛 ValueError；返回校验得到的（拷贝后的）
        (c_prev, z, i, f, o, g, c, W)。
        """
        I, H = self.I, self.H
        if type(cache) is not dict:
            raise ValueError("cache must be a dict returned by forward")
        expected_keys = ["x", "h_prev", "c_prev", "z", "i", "f",
                         "o", "g", "c", "h", "W"]
        if list(cache.keys()) != expected_keys:
            raise ValueError(
                "cache must contain exactly the 11 forward keys in order: %r"
                % expected_keys)
        _check_vector(cache["x"], I, "cache['x']")
        _check_vector(cache["h_prev"], H, "cache['h_prev']")
        c_prev = _check_vector(cache["c_prev"], H, "cache['c_prev']")
        z = _check_vector(cache["z"], I + H, "cache['z']")
        i_gate = _check_vector(cache["i"], H, "cache['i']")
        f_gate = _check_vector(cache["f"], H, "cache['f']")
        o_gate = _check_vector(cache["o"], H, "cache['o']")
        g_gate = _check_vector(cache["g"], H, "cache['g']")
        c = _check_vector(cache["c"], H, "cache['c']")
        _check_vector(cache["h"], H, "cache['h']")
        W = _check_matrix(cache["W"], 4 * H, I + H, "cache['W']")
        return c_prev, z, i_gate, f_gate, o_gate, g_gate, c, W

    def backward(self, dh, dc, cache):
        """单步反向传播，固定返回 (dx, dh_prev, dc_prev, dW, db)。

        dh、dc 须为长度 H 的 F 列表；cache 须为 dict 且严格符合 forward
        的公开缓存契约：恰含 x、h_prev、c_prev、z、i、f、o、g、c、h、W
        共 11 个有序键（多、少、乱序均非法），其中 x 长 I，h_prev、
        c_prev、i、f、o、g、c、h 长 H，z 长 I+H，W 为 4H×(I+H)，元素
        均为 F。任一容器、键序、形状或元素非法抛 ValueError；实参数量
        错误沿用 Python 自带的 TypeError。

        所有运算逐元素（变量取 cache 同名值）：
            D       = dc + dh*o*(1 - tanh(c)^2)
            dc_prev = D*f
            da 依次拼接 D*g*i*(1-i)、D*c_prev*f*(1-f)、
                       dh*tanh(c)*o*(1-o)、D*i*(1-g^2) 各 H 项
            dz[j] 对每个 j 以 0.0 起按 k=0..4H-1 升序累加
                   W[k][j]*da[k]
            dx = dz[:I]，dh_prev = dz[I:]
            dW[k][j] = da[k]*z[j]，db[k] = da[k]
        五个返回值形状依次为 I、H、H、4H×(I+H)、4H，均为全新 float
        列表（dW 逐层新建），不修改或复用 dh、dc、cache 及其内容。任一
        中间结果非有限同样抛 ValueError。
        """
        I, H = self.I, self.H
        dh = _check_vector(dh, H, "dh")
        dc = _check_vector(dc, H, "dc")

        c_prev, z, i_gate, f_gate, o_gate, g_gate, c, W = \
            self._check_forward_cache(cache)

        M = I + H

        # D、dc_prev 与四段 da（顺序 i、f、o、g）。
        D = [0.0] * H
        dc_prev = [0.0] * H
        da = [0.0] * (4 * H)
        for k in range(H):
            ik = float(i_gate[k])
            fk = float(f_gate[k])
            ok = float(o_gate[k])
            gk = float(g_gate[k])
            tc = math.tanh(float(c[k]))

            dk = (float(dc[k])
                  + float(dh[k]) * ok * (1.0 - tc * tc))
            if not math.isfinite(dk):
                raise ValueError("cell gradient became non-finite")
            D[k] = dk

            dcp = dk * fk
            if not math.isfinite(dcp):
                raise ValueError("dc_prev became non-finite")
            dc_prev[k] = dcp

            dai = dk * gk * ik * (1.0 - ik)
            if not math.isfinite(dai):
                raise ValueError("input gate gradient became non-finite")
            da[k] = dai

            daf = dk * float(c_prev[k]) * fk * (1.0 - fk)
            if not math.isfinite(daf):
                raise ValueError("forget gate gradient became non-finite")
            da[H + k] = daf

            dao = float(dh[k]) * tc * ok * (1.0 - ok)
            if not math.isfinite(dao):
                raise ValueError("output gate gradient became non-finite")
            da[2 * H + k] = dao

            dag = dk * ik * (1.0 - gk * gk)
            if not math.isfinite(dag):
                raise ValueError("block input gradient became non-finite")
            da[3 * H + k] = dag

        # dz = Wᵀ da：每个 j 独立以 0.0 起按 k 升序累加。
        dz = [0.0] * M
        for j in range(M):
            acc = 0.0
            for k in range(4 * H):
                acc += W[k][j] * da[k]
                if not math.isfinite(acc):
                    raise ValueError(
                        "dz accumulated to a non-finite value")
            dz[j] = acc

        # dW = da ⊗ z，db = da；逐层新建 float 列表。
        dW = [[0.0] * M for _ in range(4 * H)]
        for k in range(4 * H):
            row = dW[k]
            dak = da[k]
            for j in range(M):
                v = dak * z[j]
                if not math.isfinite(v):
                    raise ValueError("dW became non-finite")
                row[j] = v
        db = [float(v) for v in da]

        dx = dz[:I]
        dh_prev = dz[I:]
        return dx, dh_prev, dc_prev, dW, db

    def backward_sequence(self, dhs, caches, dh_last=None, dc_last=None,
                          tbptt_steps=None):
        """沿整条序列反向传播（可选截断 BPTT），固定返回
        (dxs, dh0, dc0, dW, db)。

        caches 须为非空的时序 list，长度确定 T，每项均为 dict 且严格符合
        forward 的公开缓存契约（同 backward 对单个 cache 的要求）；dhs 须为
        与 caches 同长的 T×H 的 F 列表；dh_last、dc_last 为 None（等价全零
        末端梯度）或长度 H 的 F 列表；tbptt_steps 为 None 或非 bool 的正
        整数 K。任一校验失败抛 ValueError；实参数量错误沿用 Python 自带的
        TypeError。

        置 ph = dh_last 或零、pc = dc_last 或零，自 t=T-1 至 0 依次调用
        现有 backward(dhs[t]+ph, pc, caches[t])：dx 放回 dxs 的 t 位，返回
        的 dh_prev、dc_prev 成为下一步的 ph、pc；dW、db 各元素自 0.0 起按
        t 降序累加。给定 K 时窗口从末端对齐：每处理完 K 步且尚有更早的步，
        便将 ph、pc 清零，故跨窗状态梯度为零。

        五个返回值形状依次为 T×I、H、H、4H×(I+H)、4H；dh0、dc0 即处理
        t=0 一步后所得的状态梯度。所有结果均为全新 float 列表（矩阵逐层
        新建），不修改输入、缓存或参数；中间量非有限同样抛 ValueError。
        """
        I, H = self.I, self.H

        # caches：非空时序 list，每项严格符合 forward 缓存契约。
        if type(caches) is not list or len(caches) == 0:
            raise ValueError("caches must be a non-empty list")
        T = len(caches)
        for cache in caches:
            self._check_forward_cache(cache)

        # dhs：与 caches 同长的 T×H 的 F 列表。
        dhs = _check_matrix(dhs, T, H, "dhs")

        # 末端状态梯度：None 等价全零。
        if dh_last is None:
            ph = [0.0] * H
        else:
            ph = _check_vector(dh_last, H, "dh_last")
        if dc_last is None:
            pc = [0.0] * H
        else:
            pc = _check_vector(dc_last, H, "dc_last")

        # tbptt_steps：None 或非 bool 正整数。
        if tbptt_steps is not None:
            if type(tbptt_steps) is bool or type(tbptt_steps) is not int \
                    or tbptt_steps <= 0:
                raise ValueError(
                    "tbptt_steps must be None or a positive integer, got %r"
                    % (tbptt_steps,))

        dxs = [None] * T
        dW = [[0.0] * (I + H) for _ in range(4 * H)]
        db = [0.0] * (4 * H)

        steps_done = 0
        for t in range(T - 1, -1, -1):
            # dhs[t] + ph：交由 backward 再做 F 校验（溢出即 ValueError）。
            dh_in = [dhs[t][j] + ph[j] for j in range(H)]
            dx, dh_prev, dc_prev, step_dW, step_db = self.backward(
                dh_in, pc, caches[t])
            dxs[t] = [float(v) for v in dx]

            # 各参数梯度元素自 0.0 起按 t 降序累加，累加后须仍有限。
            for k in range(4 * H):
                row = dW[k]
                srow = step_dW[k]
                for j in range(I + H):
                    row[j] += srow[j]
                    if not math.isfinite(row[j]):
                        raise ValueError(
                            "dW accumulated to a non-finite value")
                db[k] += step_db[k]
                if not math.isfinite(db[k]):
                    raise ValueError(
                        "db accumulated to a non-finite value")

            ph = dh_prev
            pc = dc_prev
            steps_done += 1

            # 窗口末端对齐：每满 K 步且尚有更早步，截断跨窗状态梯度。
            if (tbptt_steps is not None
                    and steps_done % tbptt_steps == 0 and t > 0):
                ph = [0.0] * H
                pc = [0.0] * H

        dh0 = [float(v) for v in ph]
        dc0 = [float(v) for v in pc]
        return dxs, dh0, dc0, dW, db
