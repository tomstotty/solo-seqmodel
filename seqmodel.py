"""seqmodel: 从零实现的序列建模库（仅 Python 标准库，离线）。

本模块提供确定性的 VanillaRNN、LSTMCell 与缩放点积 attention：

    h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)

数值约定：标量类型 F 指 type(x) 为 int 或 float（不含 bool）、float(x)
可成功转换且 math.isfinite 为真；超大整数因无法转换成有限 float 而非 F。
任何非 F 元素或形状错误都抛出 ValueError；参数个数错误沿用 Python 自带的
TypeError。
"""

import json
import math
import random
import re
import sys


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


def _attention_check(q, k, v, mask):
    """attention 与 attention_backward 共用的契约校验。

    仅做形状、元素与 mask 校验（保持 attention 原有校验与错误信息不变），
    不复制 q、k、v（调用方仍按原值参与“先乘后加”，以保留大整数精确乘积）。
    返回 (Tq, Tk, D, Dv, active)；active 为逐行新建的 bool 矩阵。
    """
    # q：非空 Tq×D 的 F 列表矩阵，D>0，首行长度确定 D。
    if type(q) is not list or len(q) == 0:
        raise ValueError("q must be a non-empty list")
    Tq = len(q)
    first_row = q[0]
    if type(first_row) is not list or len(first_row) == 0:
        raise ValueError("q rows must be non-empty lists")
    D = len(first_row)
    for x in first_row:
        if not _is_f(x):
            raise ValueError("q entries must be finite numbers, got %r"
                             % (x,))
    for row in q[1:]:
        if type(row) is not list or len(row) != D:
            raise ValueError("q must be a rectangular list of shape Tq×%d"
                             % D)
        for x in row:
            if not _is_f(x):
                raise ValueError("q entries must be finite numbers, got %r"
                                 % (x,))

    # k：非空 Tk×D 的 F 列表矩阵，与 q 共享 D。
    if type(k) is not list or len(k) == 0:
        raise ValueError("k must be a non-empty list")
    Tk = len(k)
    for row in k:
        if type(row) is not list or len(row) != D:
            raise ValueError("k must be a list of shape Tk×%d" % D)
        for x in row:
            if not _is_f(x):
                raise ValueError("k entries must be finite numbers, got %r"
                                 % (x,))

    # v：Tk×Dv 的 F 列表矩阵，Dv>0，与 k 共享 Tk。
    if type(v) is not list or len(v) != Tk or Tk == 0:
        raise ValueError("v must be a list with exactly %d rows" % Tk)
    first_row = v[0]
    if type(first_row) is not list or len(first_row) == 0:
        raise ValueError("v rows must be non-empty lists")
    Dv = len(first_row)
    for x in first_row:
        if not _is_f(x):
            raise ValueError("v entries must be finite numbers, got %r"
                             % (x,))
    for row in v[1:]:
        if type(row) is not list or len(row) != Dv:
            raise ValueError("v must be a rectangular list of shape %d×%d"
                             % (Tk, Dv))
        for x in row:
            if not _is_f(x):
                raise ValueError("v entries must be finite numbers, got %r"
                                 % (x,))

    # mask：None 或 Tq×Tk 的列表矩阵，元素 type 恰为 bool。
    if mask is None:
        active = [[True] * Tk for _ in range(Tq)]
    else:
        if type(mask) is not list or len(mask) != Tq:
            raise ValueError("mask must be a list of shape %d×%d"
                             % (Tq, Tk))
        active = []
        for row in mask:
            if type(row) is not list or len(row) != Tk:
                raise ValueError("mask must be a list of shape %d×%d"
                                 % (Tq, Tk))
            for m in row:
                if type(m) is not bool:
                    raise ValueError("mask entries must be exactly bool, "
                                     "got %r" % (m,))
            active.append(list(row))
        # 全屏蔽行在此提前判定（s 计算前）。
        for i in range(Tq):
            if not any(active[i]):
                raise ValueError("mask row %d is entirely False" % i)

    return Tq, Tk, D, Dv, active


def _attention_weights(q, k, active, Tq, Tk, D):
    """按前向语义由 q、k 重算 softmax 权重 w，返回 (w, sqrt(D))。

    对每个 (i, j)，s[i][j] 从 0.0 按 d 升序累加 q[i][d]*k[j][d]（按原值
    先乘后加），再除以 sqrt(D)；每行只在 True 位以 exp(s - max(s)) 计算
    softmax，False 位为 0.0，分母按 j 升序从 0.0 累加。w 逐层新建、元素
    均为 float。任一算术溢出或非有限结果抛 ValueError。
    """
    scale = math.sqrt(float(D))
    w = []
    for i in range(Tq):
        qi = q[i]

        # s[i][j] 从 0.0 按 d 升序累加点积，再除 sqrt(D)。
        s_row = [0.0] * Tk
        for j in range(Tk):
            kj = k[j]
            acc = 0.0
            for d in range(D):
                # 先按原值相乘（int*int 保持精确整数乘积）再加到浮点累加器；
                # 合法 F 大整数在此转换溢出时改抛 ValueError，而非泄漏
                # OverflowError。
                try:
                    acc += qi[d] * kj[d]
                except OverflowError:
                    raise ValueError(
                        "score dot product accumulated to a non-finite value")
                if not math.isfinite(acc):
                    raise ValueError(
                        "score dot product accumulated to a non-finite value")
            sval = acc / scale
            if not math.isfinite(sval):
                raise ValueError("score became non-finite after scaling")
            s_row[j] = sval

        # 行最大值只在 True 位取，保证至少一个 True 参与 softmax。
        m = None
        for j in range(Tk):
            if active[i][j]:
                sj = s_row[j]
                if m is None or sj > m:
                    m = sj

        # e = exp(s - m) 仅在 True 位计算；分母按 j 升序从 0.0 累加。
        e_row = [0.0] * Tk
        denom = 0.0
        for j in range(Tk):
            if active[i][j]:
                try:
                    ev = math.exp(s_row[j] - m)
                except OverflowError:
                    raise ValueError("softmax exp overflowed")
                if not math.isfinite(ev):
                    raise ValueError("softmax exp became non-finite")
                e_row[j] = ev
                denom += ev
                if not math.isfinite(denom):
                    raise ValueError(
                        "softmax denominator accumulated to a non-finite value")

        # False 位权重保持 0.0；归一化后须有限。
        w_row = [0.0] * Tk
        for j in range(Tk):
            if active[i][j]:
                wv = e_row[j] / denom
                if not math.isfinite(wv):
                    raise ValueError("attention weight became non-finite")
                w_row[j] = wv
        w.append(w_row)

    return w, scale


def attention(q, k, v, mask=None):
    """缩放点积注意力，返回 (c, w)，不修改或复用任何输入。

    q、k、v 须分别为非空 Tq×D、Tk×D、Tk×Dv 的 F 列表矩阵，D、Dv>0；
    mask 须为 None 或 Tq×Tk 的列表矩阵且元素 type 恰为 bool
    （True 参与、False 屏蔽），None 等价全 True。某行全 False 抛
    ValueError。

    对每个 (i, j)，s[i][j] 从 0.0 按 d 升序累加 q[i][d]*k[j][d]，
    再除以 sqrt(D)。每行只在 True 位以 exp(s - max(s)) 计算 softmax，
    False 位权重为 0.0，分母按 j 升序从 0.0 累加；c[i][a] 从 0.0 按
    j 升序累加 w[i][j]*v[j][a]。返回 Tq×Dv 的 c 与 Tq×Tk 的 w，元素
    均为 float，所有行逐层新建。任一容器、形状、元素或 mask 校验失败，
    或任一乘加、除法、指数、归一化、输出产生非有限值，均抛 ValueError；
    实参数量错误沿用 Python 自带的 TypeError。相同输入结果确定。
    """
    Tq, Tk, D, Dv, active = _attention_check(q, k, v, mask)
    w, _scale = _attention_weights(q, k, active, Tq, Tk, D)

    c = []
    for i in range(Tq):
        w_row = w[i]

        # c[i][a] 从 0.0 按 j 升序累加 w[i][j]*v[j][a]。
        c_row = [0.0] * Dv
        for a in range(Dv):
            acc = 0.0
            for j in range(Tk):
                wv = w_row[j]
                if wv != 0.0:
                    # 同样按原值先乘后加；大整数转换溢出改抛 ValueError。
                    try:
                        acc += wv * v[j][a]
                    except OverflowError:
                        raise ValueError(
                            "context accumulated to a non-finite value")
                    if not math.isfinite(acc):
                        raise ValueError(
                            "context accumulated to a non-finite value")
            c_row[a] = acc
        c.append(c_row)

    return c, w


def attention_backward(q, k, v, dc, mask=None):
    """缩放点积注意力的反向传播，返回 (dq, dk, dv)，不修改或复用任何输入。

    q、k、v、mask 完全沿用 attention 的契约；dc 须为 Tq×Dv 的 F 列表
    矩阵（Tq、Dv 由 q、v 确定），否则抛 ValueError。实参数量错误沿用
    Python 自带的 TypeError。

    按同一前向语义重算 w（False 位为 0.0），随后：
        dw[i][j] = Σ_a dc[i][a]*v[j][a]（mask False 位取 0.0）
        r[i]     = Σ_j w[i][j]*dw[i][j]
        ds[i][j] = w[i][j]*(dw[i][j] - r[i])/sqrt(D)（False 位取 0.0）
        dq[i][d] = Σ_j ds[i][j]*k[j][d]
        dk[j][d] = Σ_i ds[i][j]*q[i][d]
        dv[j][a] = Σ_i w[i][j]*dc[i][a]
    每个求和均自 0.0 起按其下标升序累加。返回形状依次为 Tq×D、Tk×D、
    Tk×Dv，元素均为 float，所有列表逐层新建。任一中间量或输出溢出、非
    有限均抛 ValueError；相同输入结果确定。
    """
    Tq, Tk, D, Dv, active = _attention_check(q, k, v, mask)

    # dc：Tq×Dv 的 F 列表矩阵（逐行浅拷贝，读取用，不修改原输入）。
    dcc = _check_matrix(dc, Tq, Dv, "dc")

    # 按同一前向语义重算 w，并取得 sqrt(D)。
    w, scale = _attention_weights(q, k, active, Tq, Tk, D)

    # dw、r、ds 逐行计算；False 位 dw、ds 保持 0.0。
    dw = [[0.0] * Tk for _ in range(Tq)]
    ds = [[0.0] * Tk for _ in range(Tq)]
    r = [0.0] * Tq
    for i in range(Tq):
        dci = dcc[i]
        wi = w[i]
        dwi = dw[i]
        dsi = ds[i]

        # dw[i][j] = Σ_a dc[i][a]*v[j][a]，a 升序，仅 True 位。
        for j in range(Tk):
            if active[i][j]:
                vj = v[j]
                acc = 0.0
                for a in range(Dv):
                    try:
                        acc += dci[a] * vj[a]
                    except OverflowError:
                        raise ValueError(
                            "dw accumulated to a non-finite value")
                    if not math.isfinite(acc):
                        raise ValueError(
                            "dw accumulated to a non-finite value")
                dwi[j] = acc

        # r[i] = Σ_j w[i][j]*dw[i][j]，j 升序（False 位 w 为 0.0）。
        racc = 0.0
        for j in range(Tk):
            try:
                racc += wi[j] * dwi[j]
            except OverflowError:
                raise ValueError("r accumulated to a non-finite value")
            if not math.isfinite(racc):
                raise ValueError("r accumulated to a non-finite value")
        r[i] = racc

        # ds[i][j] = w[i][j]*(dw[i][j] - r[i])/sqrt(D)，仅 True 位。
        for j in range(Tk):
            if active[i][j]:
                try:
                    val = wi[j] * (dwi[j] - racc) / scale
                except OverflowError:
                    raise ValueError("ds became non-finite")
                if not math.isfinite(val):
                    raise ValueError("ds became non-finite")
                dsi[j] = val

    # dq[i][d] = Σ_j ds[i][j]*k[j][d]，每个 (i,d) 独立按 j 升序累加。
    dq = [[0.0] * D for _ in range(Tq)]
    for i in range(Tq):
        dsi = ds[i]
        dqi = dq[i]
        for d in range(D):
            acc = 0.0
            for j in range(Tk):
                try:
                    acc += dsi[j] * k[j][d]
                except OverflowError:
                    raise ValueError("dq accumulated to a non-finite value")
                if not math.isfinite(acc):
                    raise ValueError("dq accumulated to a non-finite value")
            dqi[d] = acc

    # dk[j][d] = Σ_i ds[i][j]*q[i][d]，每个 (j,d) 独立按 i 升序累加。
    dk = [[0.0] * D for _ in range(Tk)]
    for j in range(Tk):
        dkj = dk[j]
        for d in range(D):
            acc = 0.0
            for i in range(Tq):
                try:
                    acc += ds[i][j] * q[i][d]
                except OverflowError:
                    raise ValueError("dk accumulated to a non-finite value")
                if not math.isfinite(acc):
                    raise ValueError("dk accumulated to a non-finite value")
            dkj[d] = acc

    # dv[j][a] = Σ_i w[i][j]*dc[i][a]，每个 (j,a) 独立按 i 升序累加。
    dv = [[0.0] * Dv for _ in range(Tk)]
    for j in range(Tk):
        dvj = dv[j]
        for a in range(Dv):
            acc = 0.0
            for i in range(Tq):
                try:
                    acc += w[i][j] * dcc[i][a]
                except OverflowError:
                    raise ValueError("dv accumulated to a non-finite value")
                if not math.isfinite(acc):
                    raise ValueError("dv accumulated to a non-finite value")
            dvj[a] = acc

    return dq, dk, dv


def attention_context_backward(n, memory, du):
    """注意力上下文加残差 u = n + attention([n], memory, memory)[0] 的反向。

    n 须为非空 H 长 F 列表；memory 须为非空 T×H 的 F 列表矩阵（T、H 由
    n 确定，每行恰长 H）；du 须为 H 长 F 列表，否则抛 ValueError。实参
    数量错误沿用 Python 自带的 TypeError。

    令
        (dq, dk, dv) = attention_backward([n], memory, memory, [du], None)
    按 i 升序计算 dn[i] = float(du[i]) + dq[0][i]；按 t、i 升序计算
    dmemory[t][i] = dk[t][i] + dv[t][i]。任一结果非有限抛 ValueError。
    返回 (dn, dmemory)：H 长与 T×H 的全新 float 列表（矩阵逐层新建），
    不修改或复用 n、memory、du 及其行。相同输入结果确定。
    """
    # n：非空 H 长 F 列表，H 由其长度确定。
    if type(n) is not list or len(n) == 0:
        raise ValueError("n must be a non-empty list")
    H = len(n)
    for x in n:
        if not _is_f(x):
            raise ValueError("n entries must be finite numbers, got %r"
                             % (x,))

    # memory：非空 T×H 的 F 列表矩阵。
    if type(memory) is not list or len(memory) == 0:
        raise ValueError("memory must be a non-empty list")
    T = len(memory)
    for t in range(T):
        row = memory[t]
        if type(row) is not list or len(row) != H:
            raise ValueError("memory must be a list of shape %d×%d"
                             % (T, H))
        for x in row:
            if not _is_f(x):
                raise ValueError(
                    "memory entries must be finite numbers, got %r" % (x,))

    # du：H 长 F 列表。
    if type(du) is not list or len(du) != H:
        raise ValueError("du must be a list of length %d" % H)
    for x in du:
        if not _is_f(x):
            raise ValueError("du entries must be finite numbers, got %r"
                             % (x,))

    dq, dk, dv = attention_backward([n], memory, memory, [du], None)

    # dn[i] = float(du[i]) + dq[0][i]，i 升序；结果须有限。
    dq0 = dq[0]
    dn = [0.0] * H
    for i in range(H):
        val = float(du[i]) + dq0[i]
        if not math.isfinite(val):
            raise ValueError("dn became non-finite")
        dn[i] = val

    # dmemory[t][i] = dk[t][i] + dv[t][i]，t、i 升序；结果须有限。
    dmemory = [[0.0] * H for _ in range(T)]
    for t in range(T):
        dkt = dk[t]
        dvt = dv[t]
        row = dmemory[t]
        for i in range(H):
            val = dkt[i] + dvt[i]
            if not math.isfinite(val):
                raise ValueError("dmemory became non-finite")
            row[i] = val

    return dn, dmemory


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

        每个隐层仿射累加器自 float(bh[i]) 起，先依 j 升序加 Wxh_i,j*x_j，
        再依 j 升序加 Whh_i,j*h_prev_j；任一乘积、每次累加或 tanh 结果非
        有限（含大整数乘积转浮点溢出）均抛 ValueError。
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
                acc = float(bh[i])
                wx_row = Wxh[i]
                for j in range(I):
                    # 按原值先乘后加；合法 F 大整数乘积转浮点溢出改抛
                    # ValueError，而非泄漏 OverflowError。
                    try:
                        acc += wx_row[j] * x[j]
                    except OverflowError:
                        raise ValueError(
                            "hidden affine accumulated non-finitely")
                    if not math.isfinite(acc):
                        raise ValueError(
                            "hidden affine accumulated non-finitely")
                wh_row = Whh[i]
                for j in range(H):
                    try:
                        acc += wh_row[j] * h_prev[j]
                    except OverflowError:
                        raise ValueError(
                            "hidden affine accumulated non-finitely")
                    if not math.isfinite(acc):
                        raise ValueError(
                            "hidden affine accumulated non-finitely")
                ni = math.tanh(acc)
                if not math.isfinite(ni):
                    raise ValueError("tanh produced a non-finite value")
                h[i] = ni
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


# 模型 JSON 顶层唯一允许的键及其出现顺序。
_MODEL_KEYS = ["version", "vocab", "Wxh", "Whh", "bh", "Why", "by", "h0"]


def _load_perplexity_model(path):
    """读取并校验 perplexity 模型文件，返回解包后的八元组。

    文件须为 UTF-8 编码的 JSON 对象，顶层键恰为
    version、vocab、Wxh、Whh、bh、Why、by、h0 且按此顺序出现（重复或多余
    均非法）：version 的 type 恰为 int 且值为 1；vocab 为非空列表，每项
    是恰含一个码点的 str，元素唯一且按码点严格升序；其余六项为 F 列表，
    形状依次为 H×V、H×H、H、V×H、V、H，其中 H>0、V=len(vocab)。
    任何读取、UTF-8、JSON 或校验失败均抛 ValueError（或 OSError）。
    """
    with open(path, "rb") as f:
        raw = f.read()
    # 先按严格 UTF-8 解码，再交由 json 解析（object_pairs_hook 保留键序与
    # 重复键，root 非对象时不会得到 (key, value) 二元组列表）。
    text = raw.decode("utf-8")
    pairs = json.loads(text, object_pairs_hook=list)
    if type(pairs) is not list or len(pairs) != len(_MODEL_KEYS):
        raise ValueError("model must be a JSON object with exactly 8 keys")
    for pair, key in zip(pairs, _MODEL_KEYS):
        if type(pair) is not tuple or len(pair) != 2 or pair[0] != key:
            raise ValueError("model keys must be exactly %r in order"
                             % _MODEL_KEYS)
    model = dict(pairs)

    version = model["version"]
    if type(version) is not int or version != 1:
        raise ValueError("version must be exactly int 1, got %r" % (version,))

    vocab = model["vocab"]
    if type(vocab) is not list or len(vocab) == 0:
        raise ValueError("vocab must be a non-empty list")
    for ch in vocab:
        # len(str) 按码点计数，组合字符序列等多码点串在此被拒。
        if type(ch) is not str or len(ch) != 1:
            raise ValueError("vocab entries must be single-codepoint strings, "
                             "got %r" % (ch,))
    if len(set(vocab)) != len(vocab) or vocab != sorted(vocab):
        raise ValueError("vocab entries must be unique and sorted by codepoint")
    V = len(vocab)

    Wxh = model["Wxh"]
    if type(Wxh) is not list or len(Wxh) == 0:
        raise ValueError("Wxh must be a non-empty list of rows")
    H = len(Wxh)
    for row in Wxh:
        if type(row) is not list or len(row) != V:
            raise ValueError("Wxh must be a list of shape H×%d" % V)
        for v in row:
            if not _is_f(v):
                raise ValueError("Wxh entries must be finite numbers, got %r"
                                 % (v,))

    Whh = model["Whh"]
    if type(Whh) is not list or len(Whh) != H:
        raise ValueError("Whh must be a list of shape %d×%d" % (H, H))
    for row in Whh:
        if type(row) is not list or len(row) != H:
            raise ValueError("Whh must be a list of shape %d×%d" % (H, H))
        for v in row:
            if not _is_f(v):
                raise ValueError("Whh entries must be finite numbers, got %r"
                                 % (v,))

    bh = model["bh"]
    if type(bh) is not list or len(bh) != H:
        raise ValueError("bh must be a list of length %d" % H)
    for v in bh:
        if not _is_f(v):
            raise ValueError("bh entries must be finite numbers, got %r" % (v,))

    Why = model["Why"]
    if type(Why) is not list or len(Why) != V:
        raise ValueError("Why must be a list of shape %d×H" % V)
    for row in Why:
        if type(row) is not list or len(row) != H:
            raise ValueError("Why must be a list of shape %d×%d" % (V, H))
        for v in row:
            if not _is_f(v):
                raise ValueError("Why entries must be finite numbers, got %r"
                                 % (v,))

    by = model["by"]
    if type(by) is not list or len(by) != V:
        raise ValueError("by must be a list of length %d" % V)
    for v in by:
        if not _is_f(v):
            raise ValueError("by entries must be finite numbers, got %r" % (v,))

    h0 = model["h0"]
    if type(h0) is not list or len(h0) != H:
        raise ValueError("h0 must be a list of length %d" % H)
    for v in h0:
        if not _is_f(v):
            raise ValueError("h0 entries must be finite numbers, got %r" % (v,))

    return vocab, Wxh, Whh, bh, Why, by, h0


def _perplexity(model_path, corpus_path):
    """计算 RNN 语言模型在给定语料上的困惑度，返回待写出的字符串。

    语料按二进制读取后以严格 UTF-8 解码为全文码点序列（含换行，不做任何
    换行符转换）；不足 2 个码点或出现 vocab 表外字符均失败。置 h=h0、
    L=0.0、T=len(CORPUS)-1，t 升序（x、y 为当前、下一字符索引）：
        n_i = tanh(bh_i + Wxh_i,x + Σ_j Whh_i,j*h_j)
        z_k = by_k + Σ_j Why_k,j*n_j
        m=max(z)，d 从 0.0 依 k 累加 exp(z_k-m)
        L += m + log(d) - z_y，随后 h=n
    仿射累加自 float 偏置起，n 先加 Wxh_i,x 再依 j 升序加 Whh，z 依 j
    升序加 Why。任一中间量非有限（含最终 exp(L/T) 溢出）均抛 ValueError。
    成功返回 format(exp(L/T), '.17g') + '\\n'。
    """
    vocab, Wxh, Whh, bh, Why, by, h0 = _load_perplexity_model(model_path)
    V = len(vocab)
    H = len(bh)

    with open(corpus_path, "rb") as f:
        corpus = f.read().decode("utf-8")
    if len(corpus) < 2:
        raise ValueError("corpus must contain at least 2 codepoints")

    table = {ch: i for i, ch in enumerate(vocab)}
    ids = [0] * len(corpus)
    for t, ch in enumerate(corpus):
        ix = table.get(ch)
        if ix is None:
            raise ValueError("corpus contains an out-of-vocab character")
        ids[t] = ix

    h = h0
    L = 0.0
    T = len(ids) - 1
    for t in range(T):
        x = ids[t]
        y = ids[t + 1]

        # n_i = tanh(bh_i + Wxh_i,x + Σ_j Whh_i,j*h_j)：累加器自 float
        # 偏置起，先加 Wxh，再依 j 升序加 Whh*h。
        n = [0.0] * H
        for i in range(H):
            acc = float(bh[i])
            acc += Wxh[i][x]
            if not math.isfinite(acc):
                raise ValueError("hidden affine accumulated non-finitely")
            wh_row = Whh[i]
            for j in range(H):
                acc += wh_row[j] * h[j]
                if not math.isfinite(acc):
                    raise ValueError("hidden affine accumulated non-finitely")
            ni = math.tanh(acc)
            if not math.isfinite(ni):
                raise ValueError("tanh produced a non-finite value")
            n[i] = ni

        # z_k = by_k + Σ_j Why_k,j*n_j，依 j 升序自 float 偏置累加。
        z = [0.0] * V
        for k in range(V):
            acc = float(by[k])
            why_row = Why[k]
            for j in range(H):
                acc += why_row[j] * n[j]
                if not math.isfinite(acc):
                    raise ValueError("output affine accumulated non-finitely")
            z[k] = acc

        # log-sum-exp：m=max(z)，d 从 0.0 依 k 升序累加 exp(z_k-m)。
        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError("softmax denominator accumulated non-finitely")

        step = m + math.log(d) - z[y]
        if not math.isfinite(step):
            raise ValueError("cross-entropy step is non-finite")
        L += step
        if not math.isfinite(L):
            raise ValueError("total cross-entropy accumulated non-finitely")
        h = n

    perplexity = math.exp(L / T)
    if not math.isfinite(perplexity):
        raise ValueError("perplexity is non-finite")
    return format(perplexity, ".17g") + "\n"


# LSTM 模型 JSON 顶层唯一允许的键及其出现顺序。
_LSTM_MODEL_KEYS = ["version", "vocab", "W", "b", "Why", "by", "h0", "c0"]


def _load_perplexity_lstm_model(path):
    """读取并校验 perplexity-lstm 模型文件，返回解包后的七元组。

    文件须为 UTF-8 编码的 JSON 对象，顶层键恰为
    version、vocab、W、b、Why、by、h0、c0 且按此顺序出现（重复或多余
    均非法）：version 的 type 恰为 int 且值为 2；vocab 的契约与
    perplexity 相同（非空列表，每项是恰含一个码点的 str，元素唯一且按
    码点严格升序）；其余六项为 F 列表，形状依次为 4H×(V+H)、4H、V×H、
    V、H、H，其中 V=len(vocab)、H=len(h0)>0。任何读取、UTF-8、JSON 或
    校验失败均抛 ValueError（或 OSError）。
    """
    with open(path, "rb") as f:
        raw = f.read()
    # 先按严格 UTF-8 解码，再交由 json 解析（object_pairs_hook 保留键序与
    # 重复键，root 非对象时不会得到 (key, value) 二元组列表）。
    text = raw.decode("utf-8")
    pairs = json.loads(text, object_pairs_hook=list)
    if type(pairs) is not list or len(pairs) != len(_LSTM_MODEL_KEYS):
        raise ValueError("model must be a JSON object with exactly 8 keys")
    for pair, key in zip(pairs, _LSTM_MODEL_KEYS):
        if type(pair) is not tuple or len(pair) != 2 or pair[0] != key:
            raise ValueError("model keys must be exactly %r in order"
                             % _LSTM_MODEL_KEYS)
    model = dict(pairs)

    version = model["version"]
    if type(version) is not int or version != 2:
        raise ValueError("version must be exactly int 2, got %r" % (version,))

    vocab = model["vocab"]
    if type(vocab) is not list or len(vocab) == 0:
        raise ValueError("vocab must be a non-empty list")
    for ch in vocab:
        # len(str) 按码点计数，组合字符序列等多码点串在此被拒。
        if type(ch) is not str or len(ch) != 1:
            raise ValueError("vocab entries must be single-codepoint strings, "
                             "got %r" % (ch,))
    if len(set(vocab)) != len(vocab) or vocab != sorted(vocab):
        raise ValueError("vocab entries must be unique and sorted by codepoint")
    V = len(vocab)

    h0 = model["h0"]
    if type(h0) is not list or len(h0) == 0:
        raise ValueError("h0 must be a non-empty list")
    H = len(h0)
    for v in h0:
        if not _is_f(v):
            raise ValueError("h0 entries must be finite numbers, got %r" % (v,))

    W = model["W"]
    if type(W) is not list or len(W) != 4 * H:
        raise ValueError("W must be a list of shape %d×%d" % (4 * H, V + H))
    for row in W:
        if type(row) is not list or len(row) != V + H:
            raise ValueError("W must be a list of shape %d×%d" % (4 * H, V + H))
        for v in row:
            if not _is_f(v):
                raise ValueError("W entries must be finite numbers, got %r"
                                 % (v,))

    b = model["b"]
    if type(b) is not list or len(b) != 4 * H:
        raise ValueError("b must be a list of length %d" % (4 * H))
    for v in b:
        if not _is_f(v):
            raise ValueError("b entries must be finite numbers, got %r" % (v,))

    Why = model["Why"]
    if type(Why) is not list or len(Why) != V:
        raise ValueError("Why must be a list of shape %d×H" % V)
    for row in Why:
        if type(row) is not list or len(row) != H:
            raise ValueError("Why must be a list of shape %d×%d" % (V, H))
        for v in row:
            if not _is_f(v):
                raise ValueError("Why entries must be finite numbers, got %r"
                                 % (v,))

    by = model["by"]
    if type(by) is not list or len(by) != V:
        raise ValueError("by must be a list of length %d" % V)
    for v in by:
        if not _is_f(v):
            raise ValueError("by entries must be finite numbers, got %r" % (v,))

    c0 = model["c0"]
    if type(c0) is not list or len(c0) != H:
        raise ValueError("c0 must be a list of length %d" % H)
    for v in c0:
        if not _is_f(v):
            raise ValueError("c0 entries must be finite numbers, got %r" % (v,))

    return vocab, W, b, Why, by, h0, c0


def _perplexity_lstm(model_path, corpus_path):
    """计算 LSTM 语言模型在给定语料上的困惑度，返回待写出的字符串。

    MODEL 沿用 perplexity 的 UTF-8 JSON、vocab 与 F 规则，顶层键恰为
    version、vocab、W、b、Why、by、h0、c0（version 恰为 int 2）；CORPUS
    沿用 perplexity 的严格 UTF-8 全文码点、词表及至少 2 码点契约。

    置 h=h0、c=c0、L=0.0、T=len(CORPUS)-1，t 升序：以当前字符的 V 长
    one-hot 为 x，调用装入 W、b 的 LSTMCell.forward(x, h, c)，取前两项
    更新 h、c；随后按 perplexity 相同的下标、float 偏置与升序累加规则
    计算 z_k = by_k + Σ_j Why_k,j*h_j，并以减最大值的 log-sum-exp 累加
    下一字符的负对数似然。任一中间量非有限（含最终 exp(L/T) 溢出）均抛
    ValueError。成功返回 format(exp(L/T), '.17g') + '\\n'。
    """
    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    with open(corpus_path, "rb") as f:
        corpus = f.read().decode("utf-8")
    if len(corpus) < 2:
        raise ValueError("corpus must contain at least 2 codepoints")

    table = {ch: i for i, ch in enumerate(vocab)}
    ids = [0] * len(corpus)
    for t, ch in enumerate(corpus):
        ix = table.get(ch)
        if ix is None:
            raise ValueError("corpus contains an out-of-vocab character")
        ids[t] = ix

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    h = list(h0)
    c = list(c0)
    L = 0.0
    T = len(ids) - 1
    for t in range(T):
        y = ids[t + 1]

        # 当前字符的 V 长 one-hot 输入。
        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        # z_k = by_k + Σ_j Why_k,j*h_j，依 j 升序自 float 偏置累加。
        z = [0.0] * V
        for k in range(V):
            acc = float(by[k])
            why_row = Why[k]
            for j in range(H):
                acc += why_row[j] * h[j]
                if not math.isfinite(acc):
                    raise ValueError("output affine accumulated non-finitely")
            z[k] = acc

        # log-sum-exp：m=max(z)，d 从 0.0 依 k 升序累加 exp(z_k-m)。
        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError("softmax denominator accumulated non-finitely")

        step = m + math.log(d) - z[y]
        if not math.isfinite(step):
            raise ValueError("cross-entropy step is non-finite")
        L += step
        if not math.isfinite(L):
            raise ValueError("total cross-entropy accumulated non-finitely")

    perplexity = math.exp(L / T)
    if not math.isfinite(perplexity):
        raise ValueError("perplexity is non-finite")
    return format(perplexity, ".17g") + "\n"


def _train(model_path, corpus_path, out_path):
    """对模型做一次全语料 SGD 更新并把新模型写入 OUT。

    MODEL、CORPUS 完全沿用 perplexity 的读取、形状、F、UTF-8、词表及语料
    契约。以当前字符的 one-hot 向量和 h0 调用装入参数的 VanillaRNN.forward
    得到 N-1 步隐状态；标签为下一字符，按 _perplexity 的顺序计算各步
    logit 与 softmax。令 g = p - onehot(y)，按 t 升序累加
    dWhy += g⊗h、dby += g，并按 k 升序计算 dhs = Whyᵀg，随后调用
    backward(dhs) 得到 dWxh、dWhh、dbh。依 dWxh、dWhh、dbh、dWhy、dby
    行序求全局范数，超过 5.0 即统一缩放至 5.0；五组参数减去 0.1 倍梯度，
    h0 不变。任一中间量或结果非有限均抛 ValueError。

    OUT 复用 perplexity 模型的八个键、键序与形状，数组元素均转为 float；
    文件内容恰为 json.dumps(obj, ensure_ascii=True,
    separators=(',', ':'), allow_nan=False) 的 UTF-8 编码再加一个 LF，
    负零保留为 -0.0。
    """
    vocab, Wxh, Whh, bh, Why, by, h0 = _load_perplexity_model(model_path)
    V = len(vocab)
    H = len(bh)

    with open(corpus_path, "rb") as f:
        corpus = f.read().decode("utf-8")
    if len(corpus) < 2:
        raise ValueError("corpus must contain at least 2 codepoints")

    table = {ch: i for i, ch in enumerate(vocab)}
    ids = [0] * len(corpus)
    for t, ch in enumerate(corpus):
        ix = table.get(ch)
        if ix is None:
            raise ValueError("corpus contains an out-of-vocab character")
        ids[t] = ix

    T = len(ids) - 1

    # 当前字符的 one-hot 输入序列。
    xs = []
    for t in range(T):
        row = [0.0] * V
        row[ids[t]] = 1.0
        xs.append(row)

    rnn = VanillaRNN(V, H)
    rnn.Wxh = [list(row) for row in Wxh]
    rnn.Whh = [list(row) for row in Whh]
    rnn.bh = list(bh)
    hs = rnn.forward(xs, h0)

    dWhy = [[0.0] * H for _ in range(V)]
    dby = [0.0] * V
    dhs = [[0.0] * H for _ in range(T)]

    for t in range(T):
        n = hs[t]
        y = ids[t + 1]

        # logit：与 _perplexity 相同，自 float 偏置起依 j 升序累加。
        z = [0.0] * V
        for k in range(V):
            acc = float(by[k])
            why_row = Why[k]
            for j in range(H):
                acc += why_row[j] * n[j]
                if not math.isfinite(acc):
                    raise ValueError("output affine accumulated non-finitely")
            z[k] = acc

        # softmax：m=max(z)，d 从 0.0 依 k 升序累加 exp(z_k-m)。
        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            ev = math.exp(z[k] - m)
            if not math.isfinite(ev):
                raise ValueError("softmax exp became non-finite")
            e[k] = ev
            d += ev
            if not math.isfinite(d):
                raise ValueError("softmax denominator accumulated non-finitely")

        # g = p - onehot(y)。
        g = [0.0] * V
        for k in range(V):
            gk = e[k] / d
            if k == y:
                gk -= 1.0
            if not math.isfinite(gk):
                raise ValueError("output gradient became non-finite")
            g[k] = gk

        # 按 t 升序累加 dWhy += g⊗h、dby += g。
        for k in range(V):
            gk = g[k]
            dby[k] += gk
            if not math.isfinite(dby[k]):
                raise ValueError("dby accumulated to a non-finite value")
            dw_row = dWhy[k]
            for j in range(H):
                dw_row[j] += gk * n[j]
                if not math.isfinite(dw_row[j]):
                    raise ValueError("dWhy accumulated to a non-finite value")

        # dhs = Whyᵀg：每个 j 独立以 0.0 起按 k 升序累加。
        dh_row = dhs[t]
        for j in range(H):
            acc = 0.0
            for k in range(V):
                acc += Why[k][j] * g[k]
                if not math.isfinite(acc):
                    raise ValueError("dhs accumulated to a non-finite value")
            dh_row[j] = acc

    _dxs, dWxh, dWhh, dbh, _dh0 = rnn.backward(dhs)

    # 依 dWxh、dWhh、dbh、dWhy、dby 行序累加平方和求全局范数。
    sum_sq = 0.0
    for group in (dWxh, dWhh, dbh, dWhy, dby):
        if type(group[0]) is list:
            for row in group:
                for v in row:
                    if not math.isfinite(v):
                        raise ValueError("gradient is non-finite")
                    sum_sq += v * v
                    if not math.isfinite(sum_sq):
                        raise ValueError(
                            "global norm accumulated to a non-finite value")
        else:
            for v in group:
                if not math.isfinite(v):
                    raise ValueError("gradient is non-finite")
                sum_sq += v * v
                if not math.isfinite(sum_sq):
                    raise ValueError(
                        "global norm accumulated to a non-finite value")

    global_norm = math.sqrt(sum_sq)
    scale = 5.0 / global_norm if global_norm > 5.0 else 1.0

    # 五组参数减 0.1 倍（裁剪后的）梯度；h0 不变。结果非有限即失败。
    def _updated(old, grad):
        value = float(old) - 0.1 * (grad * scale)
        if not math.isfinite(value):
            raise ValueError("updated parameter became non-finite")
        return value

    new_Wxh = [[_updated(Wxh[i][j], dWxh[i][j]) for j in range(V)]
               for i in range(H)]
    new_Whh = [[_updated(Whh[i][j], dWhh[i][j]) for j in range(H)]
               for i in range(H)]
    new_bh = [_updated(bh[i], dbh[i]) for i in range(H)]
    new_Why = [[_updated(Why[k][j], dWhy[k][j]) for j in range(H)]
               for k in range(V)]
    new_by = [_updated(by[k], dby[k]) for k in range(V)]

    obj = {
        "version": 1,
        "vocab": vocab,
        "Wxh": new_Wxh,
        "Whh": new_Whh,
        "bh": new_bh,
        "Why": new_Why,
        "by": new_by,
        "h0": [float(v) for v in h0],
    }
    text = json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"
    with open(out_path, "wb") as f:
        f.write(text.encode("utf-8"))


def _train_lstm(model_path, corpus_path, out_path):
    """对 LSTM 模型做一次全语料 SGD 更新并把新模型写入 OUT。

    MODEL、CORPUS 完全沿用 perplexity-lstm version 2 的八键、F、严格
    UTF-8、形状、词表及语料至少 2 码点契约。置 h=h0、c=c0，t 升序：以
    当前字符的 V 长 one-hot 为 x，调用装入 W、b 的 LSTMCell.forward
    (x, h, c)，以返回的前两项更新 h、c 并缓存 cache；输出层 logit、
    softmax、g=p-onehot(y)、dWhy += g⊗h、dby += g 与 dhs = Whyᵀg 的
    公式与累加顺序完全沿用 train，仅隐状态换为 LSTM 的 h。随后调用
    backward_sequence(dhs, caches) 取得整条序列的 dW、db。

    依 dW、db、dWhy、dby 行序求全局范数，超过 5.0 即统一缩放至 5.0；四组
    参数减去 0.1 倍梯度，h0、c0 不变。任一中间量或结果非有限均抛
    ValueError。

    OUT 复用 perplexity-lstm 模型的八个键、键序与形状（version 恰为
    int 2），数组元素均转为 float；文件内容恰为 json.dumps(obj,
    ensure_ascii=True, separators=(',', ':'), allow_nan=False) 的 UTF-8
    编码再加一个 LF，负零保留为 -0.0。
    """
    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    with open(corpus_path, "rb") as f:
        corpus = f.read().decode("utf-8")
    if len(corpus) < 2:
        raise ValueError("corpus must contain at least 2 codepoints")

    table = {ch: i for i, ch in enumerate(vocab)}
    ids = [0] * len(corpus)
    for t, ch in enumerate(corpus):
        ix = table.get(ch)
        if ix is None:
            raise ValueError("corpus contains an out-of-vocab character")
        ids[t] = ix

    T = len(ids) - 1

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    # t 升序以 (h0, c0) 为初态逐步前向，缓存每一步的隐状态与 cache。
    hs = []
    caches = []
    h = list(h0)
    c = list(c0)
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0
        h, c, cache = cell.forward(x, h, c)
        hs.append(h)
        caches.append(cache)

    dWhy = [[0.0] * H for _ in range(V)]
    dby = [0.0] * V
    dhs = [[0.0] * H for _ in range(T)]

    for t in range(T):
        n = hs[t]
        y = ids[t + 1]

        # logit：与 train 相同，自 float 偏置起依 j 升序累加。
        z = [0.0] * V
        for k in range(V):
            acc = float(by[k])
            why_row = Why[k]
            for j in range(H):
                acc += why_row[j] * n[j]
                if not math.isfinite(acc):
                    raise ValueError("output affine accumulated non-finitely")
            z[k] = acc

        # softmax：m=max(z)，d 从 0.0 依 k 升序累加 exp(z_k-m)。
        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            ev = math.exp(z[k] - m)
            if not math.isfinite(ev):
                raise ValueError("softmax exp became non-finite")
            e[k] = ev
            d += ev
            if not math.isfinite(d):
                raise ValueError("softmax denominator accumulated non-finitely")

        # g = p - onehot(y)。
        g = [0.0] * V
        for k in range(V):
            gk = e[k] / d
            if k == y:
                gk -= 1.0
            if not math.isfinite(gk):
                raise ValueError("output gradient became non-finite")
            g[k] = gk

        # 按 t 升序累加 dWhy += g⊗h、dby += g。
        for k in range(V):
            gk = g[k]
            dby[k] += gk
            if not math.isfinite(dby[k]):
                raise ValueError("dby accumulated to a non-finite value")
            dw_row = dWhy[k]
            for j in range(H):
                dw_row[j] += gk * n[j]
                if not math.isfinite(dw_row[j]):
                    raise ValueError("dWhy accumulated to a non-finite value")

        # dhs = Whyᵀg：每个 j 独立以 0.0 起按 k 升序累加。
        dh_row = dhs[t]
        for j in range(H):
            acc = 0.0
            for k in range(V):
                acc += Why[k][j] * g[k]
                if not math.isfinite(acc):
                    raise ValueError("dhs accumulated to a non-finite value")
            dh_row[j] = acc

    _dxs, _dh0, _dc0, dW, db = cell.backward_sequence(dhs, caches)

    # 依 dW、db、dWhy、dby 行序累加平方和求全局范数。
    sum_sq = 0.0
    for group in (dW, db, dWhy, dby):
        if type(group[0]) is list:
            for row in group:
                for v in row:
                    if not math.isfinite(v):
                        raise ValueError("gradient is non-finite")
                    sum_sq += v * v
                    if not math.isfinite(sum_sq):
                        raise ValueError(
                            "global norm accumulated to a non-finite value")
        else:
            for v in group:
                if not math.isfinite(v):
                    raise ValueError("gradient is non-finite")
                sum_sq += v * v
                if not math.isfinite(sum_sq):
                    raise ValueError(
                        "global norm accumulated to a non-finite value")

    global_norm = math.sqrt(sum_sq)
    scale = 5.0 / global_norm if global_norm > 5.0 else 1.0

    # 四组参数减 0.1 倍（裁剪后的）梯度；h0、c0 不变。结果非有限即失败。
    def _updated(old, grad):
        value = float(old) - 0.1 * (grad * scale)
        if not math.isfinite(value):
            raise ValueError("updated parameter became non-finite")
        return value

    new_W = [[_updated(W[k][j], dW[k][j]) for j in range(V + H)]
             for k in range(4 * H)]
    new_b = [_updated(b[k], db[k]) for k in range(4 * H)]
    new_Why = [[_updated(Why[k][j], dWhy[k][j]) for j in range(H)]
               for k in range(V)]
    new_by = [_updated(by[k], dby[k]) for k in range(V)]

    obj = {
        "version": 2,
        "vocab": vocab,
        "W": new_W,
        "b": new_b,
        "Why": new_Why,
        "by": new_by,
        "h0": [float(v) for v in h0],
        "c0": [float(v) for v in c0],
    }
    text = json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"
    with open(out_path, "wb") as f:
        f.write(text.encode("utf-8"))


_INT_RE = re.compile(r"\A(?:0|-?[1-9][0-9]*)\Z")
_NONNEG_INT_RE = re.compile(r"\A(?:0|[1-9][0-9]*)\Z")
_WINDOW_RE = re.compile(r"\A[1-9][0-9]*\Z")


def _sample(model_path, start, seed_text, temperature_text, length_text):
    """从 RNN 语言模型采样 LENGTH 个码点，返回待写出的字符串。

    MODEL 沿用 perplexity 的八键读取契约。START 须恰为词表内的一个码点；
    SEED 整串匹配 0|-?[1-9][0-9]*；TEMPERATURE 经 float() 解析后须有限且
    严格大于 0；LENGTH 整串匹配 0|[1-9][0-9]*，否则抛 ValueError。

    置 r=random.Random(int(SEED))、h=h0、x=START 索引。循环 LENGTH 次：
    按 _perplexity 的公式、下标及累加顺序由 x、h 计算 n 与 logit z；依 k
    升序令 a_k=z_k/TEMPERATURE、m=max(a)、e_k=exp(a_k-m)，d 自 0.0 累加
    e_k。令 u=r.random()*d，自 0.0 依 k 升序累加 e_k，选择首个累计值严格
    大于 u 的字符 k（无则取词表末项）；追加 vocab[k]，再令 h=n、x=k。任一
    运算非有限均抛 ValueError。成功返回 LENGTH 个码点再加一个 LF。
    """
    vocab, Wxh, Whh, bh, Why, by, h0 = _load_perplexity_model(model_path)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # SEED：整串匹配整数词法（禁止空白、+ 前缀、前导零、下划线）。
    if not _INT_RE.match(seed_text):
        raise ValueError("SEED must match 0|-?[1-9][0-9]*")
    seed = int(seed_text)

    # TEMPERATURE：float() 可解析且有限、严格大于 0；inf/nan/0/负数均失败。
    temperature = float(temperature_text)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("TEMPERATURE must be a finite positive float")

    # LENGTH：整串匹配非负整数词法。
    if not _NONNEG_INT_RE.match(length_text):
        raise ValueError("LENGTH must match 0|[1-9][0-9]*")
    length = int(length_text)

    return _sample_loop(vocab, Wxh, Whh, bh, Why, by, h0, seed,
                        vocab.index(start), length, lambda t: temperature)


def _sample_loop(vocab, Wxh, Whh, bh, Why, by, h0, seed, start_index,
                 length, temperature_at):
    """sample 与 sample-anneal 共用的采样循环，返回待写出的字符串。

    整次调用仅初始化一次 r=random.Random(seed)；置 h=h0、x=start_index。
    t 自 0 升序循环 length 次：先令 temperature=temperature_at(t)，再按
    _perplexity 的公式、下标及累加顺序由 x、h 计算 n 与 logit z；依 k
    升序令 a_k=z_k/temperature、m=max(a)、e_k=exp(a_k-m)，d 自 0.0 累加
    e_k。令 u=r.random()*d，自 0.0 依 k 升序累加 e_k，选择首个累计值严格
    大于 u 的字符 k（无则取词表末项）；追加 vocab[k]，再令 h=n、x=k。任一
    运算非有限均抛 ValueError。返回 length 个码点再加一个 LF。
    """
    V = len(vocab)
    H = len(bh)
    rng = random.Random(seed)
    h = h0
    x = start_index
    out = []

    for t in range(length):
        temperature = temperature_at(t)
        # n_i = tanh(bh_i + Wxh_i,x + Σ_j Whh_i,j*h_j)，与 _perplexity
        # 相同的起点、下标与累加顺序。
        n = [0.0] * H
        for i in range(H):
            acc = float(bh[i])
            acc += Wxh[i][x]
            if not math.isfinite(acc):
                raise ValueError("hidden affine accumulated non-finitely")
            wh_row = Whh[i]
            for j in range(H):
                acc += wh_row[j] * h[j]
                if not math.isfinite(acc):
                    raise ValueError("hidden affine accumulated non-finitely")
            ni = math.tanh(acc)
            if not math.isfinite(ni):
                raise ValueError("tanh produced a non-finite value")
            n[i] = ni

        # z_k = by_k + Σ_j Why_k,j*n_j，依 j 升序自 float 偏置累加。
        z = [0.0] * V
        for k in range(V):
            acc = float(by[k])
            why_row = Why[k]
            for j in range(H):
                acc += why_row[j] * n[j]
                if not math.isfinite(acc):
                    raise ValueError("output affine accumulated non-finitely")
            z[k] = acc

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        m = None
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
            if m is None or ak > m:
                m = ak
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            try:
                ek = math.exp(a[k] - m)
            except OverflowError:
                raise ValueError("softmax exp overflowed")
            if not math.isfinite(ek):
                raise ValueError("softmax exp became non-finite")
            e[k] = ek
            d += ek
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        # u=r.random()*d；自 0.0 依 k 升序累加 e_k，选首个累计值严格大于
        # u 者；无则取词表末项。
        u = rng.random() * d
        if not math.isfinite(u):
            raise ValueError("sample threshold became non-finite")
        chosen = V - 1
        cum = 0.0
        for k in range(V):
            cum += e[k]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > u:
                chosen = k
                break

        out.append(vocab[chosen])
        h = n
        x = chosen

    return "".join(out) + "\n"


def _sample_anneal(model_path, start, seed_text, start_t_text, end_t_text,
                   length_text):
    """以线性退火温度从 RNN 语言模型采样 LENGTH 个码点，返回待写出的字符串。

    MODEL、START、SEED、LENGTH 的校验与 _sample 完全一致。START_T、END_T
    各经 float() 解析，结果须有限且严格大于 0，否则抛 ValueError。整次
    调用仅初始化一次 random.Random(int(SEED))。

    LENGTH 为 0 时不计算温度；为 1 时仅用 START_T；否则 t 自 0 升序，第
    t 步温度严格按 Python 表达式 START_T+(END_T-START_T)*t/(LENGTH-1)
    求值，结果非有限即抛 ValueError。每步以该温度替换 _sample 的固定温度，
    隐藏态、logit、稳定 softmax 与按词表升序累计抽样的公式及运算顺序均与
    _sample 相同，生成后继续更新同一 h 和 x。成功返回 LENGTH 个码点再加
    一个 LF。
    """
    vocab, Wxh, Whh, bh, Why, by, h0 = _load_perplexity_model(model_path)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # SEED：整串匹配整数词法（禁止空白、+ 前缀、前导零、下划线）。
    if not _INT_RE.match(seed_text):
        raise ValueError("SEED must match 0|-?[1-9][0-9]*")
    seed = int(seed_text)

    # START_T、END_T：float() 可解析且有限、严格大于 0。
    start_t = float(start_t_text)
    if not math.isfinite(start_t) or start_t <= 0.0:
        raise ValueError("START_T must be a finite positive float")
    end_t = float(end_t_text)
    if not math.isfinite(end_t) or end_t <= 0.0:
        raise ValueError("END_T must be a finite positive float")

    # LENGTH：整串匹配非负整数词法。
    if not _NONNEG_INT_RE.match(length_text):
        raise ValueError("LENGTH must match 0|[1-9][0-9]*")
    length = int(length_text)

    if length == 1:
        # 仅用 START_T，不求值退火表达式。
        def temperature_at(t):
            return start_t
    elif length >= 2:
        def temperature_at(t):
            # 严格按 START_T+(END_T-START_T)*t/(LENGTH-1) 的运算顺序求值。
            temp = start_t + (end_t - start_t) * t / (length - 1)
            if not math.isfinite(temp):
                raise ValueError("annealed temperature became non-finite")
            return temp
    else:
        temperature_at = None  # LENGTH 为 0：循环不执行，不计算温度。

    return _sample_loop(vocab, Wxh, Whh, bh, Why, by, h0, seed,
                        vocab.index(start), length, temperature_at)


def _sample_lstm(model_path, start, seed_text, temperature_text, length_text):
    """从 LSTM 语言模型采样 LENGTH 个码点，返回待写出的字符串。

    MODEL 严格沿用 perplexity-lstm 的 version 2 八键顺序、形状、F 及
    UTF-8 契约；START、SEED、TEMPERATURE、LENGTH 的词法与校验完全沿用
    _sample。整次调用仅初始化一次 r=random.Random(int(SEED))，不写任何
    文件。

    置 h=h0、c=c0、x=START 索引。循环 LENGTH 次：以 x 的 V 长 one-hot
    调用装入 W、b 的 LSTMCell.forward(x, h, c)，取前两项更新 h、c；按
    perplexity-lstm 的 float 偏置与 j 升序累加计算 logit
    z_k = by_k + Σ_j Why_k,j*h_j；温度缩放、稳定 softmax、随机阈值、
    k 升序累计与选中规则逐项沿用 _sample；追加 vocab[k] 并令 x=k。任一
    中间量非有限均抛 ValueError。成功返回 LENGTH 个码点再加一个 LF。
    """
    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # SEED：整串匹配整数词法（禁止空白、+ 前缀、前导零、下划线）。
    if not _INT_RE.match(seed_text):
        raise ValueError("SEED must match 0|-?[1-9][0-9]*")
    seed = int(seed_text)

    # TEMPERATURE：float() 可解析且有限、严格大于 0；inf/nan/0/负数均失败。
    temperature = float(temperature_text)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("TEMPERATURE must be a finite positive float")

    # LENGTH：整串匹配非负整数词法。
    if not _NONNEG_INT_RE.match(length_text):
        raise ValueError("LENGTH must match 0|[1-9][0-9]*")
    length = int(length_text)

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    rng = random.Random(seed)
    h = list(h0)
    c = list(c0)
    x = vocab.index(start)
    out = []

    for _t in range(length):
        # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, c = cell.forward(xvec, h, c)[:2]

        # z_k = by_k + Σ_j Why_k,j*h_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, h)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        m = None
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
            if m is None or ak > m:
                m = ak
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            try:
                ek = math.exp(a[k] - m)
            except OverflowError:
                raise ValueError("softmax exp overflowed")
            if not math.isfinite(ek):
                raise ValueError("softmax exp became non-finite")
            e[k] = ek
            d += ek
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        # u=r.random()*d；自 0.0 依 k 升序累加 e_k，选首个累计值严格大于
        # u 者；无则取词表末项。
        u = rng.random() * d
        if not math.isfinite(u):
            raise ValueError("sample threshold became non-finite")
        chosen = V - 1
        cum = 0.0
        for k in range(V):
            cum += e[k]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > u:
                chosen = k
                break

        out.append(vocab[chosen])
        x = chosen

    return "".join(out) + "\n"


def _sample_lstm_anneal(model_path, start, seed_text, start_t_text,
                        end_t_text, length_text):
    """以线性退火温度从 LSTM 语言模型采样 LENGTH 个码点，返回待写出的字符串。

    MODEL、START、SEED、LENGTH 的校验与 _sample_lstm 完全一致。START_T、
    END_T 各经 float() 解析，结果须有限且严格大于 0，否则抛 ValueError。
    整次调用仅初始化一次 r=random.Random(int(SEED))，不写任何文件。

    LENGTH 为 0 时不计算温度；为 1 时仅用 START_T；否则 t 自 0 升序，第
    t 步温度严格按 Python 表达式 START_T+(END_T-START_T)*t/(LENGTH-1)
    求值，结果非有限或不大于 0 即抛 ValueError。置 h=h0、c=c0、x=START
    索引，每步以 x 的 V 长 one-hot 调用装入 W、b 的 LSTMCell.forward(x, h,
    c)，取前两项更新 h、c；logit、温度缩放、稳定 softmax、随机阈值、k 升序
    累计与选中规则逐项沿用 _sample_lstm；追加 vocab[k] 并令 x=k。任一中间
    量非有限均抛 ValueError。成功返回 LENGTH 个码点再加一个 LF。
    """
    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # SEED：整串匹配整数词法（禁止空白、+ 前缀、前导零、下划线）。
    if not _INT_RE.match(seed_text):
        raise ValueError("SEED must match 0|-?[1-9][0-9]*")
    seed = int(seed_text)

    # START_T、END_T：float() 可解析且有限、严格大于 0。
    start_t = float(start_t_text)
    if not math.isfinite(start_t) or start_t <= 0.0:
        raise ValueError("START_T must be a finite positive float")
    end_t = float(end_t_text)
    if not math.isfinite(end_t) or end_t <= 0.0:
        raise ValueError("END_T must be a finite positive float")

    # LENGTH：整串匹配非负整数词法。
    if not _NONNEG_INT_RE.match(length_text):
        raise ValueError("LENGTH must match 0|[1-9][0-9]*")
    length = int(length_text)

    if length == 1:
        # 仅用 START_T，不求值退火表达式。
        def temperature_at(t):
            return start_t
    elif length >= 2:
        def temperature_at(t):
            # 严格按 START_T+(END_T-START_T)*t/(LENGTH-1) 的运算顺序求值。
            temp = start_t + (end_t - start_t) * t / (length - 1)
            if not math.isfinite(temp) or temp <= 0.0:
                raise ValueError(
                    "annealed temperature became non-finite or non-positive")
            return temp
    else:
        temperature_at = None  # LENGTH 为 0：循环不执行，不计算温度。

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    rng = random.Random(seed)
    h = list(h0)
    c = list(c0)
    x = vocab.index(start)
    out = []

    for t in range(length):
        temperature = temperature_at(t)
        # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, c = cell.forward(xvec, h, c)[:2]

        # z_k = by_k + Σ_j Why_k,j*h_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, h)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        m = None
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
            if m is None or ak > m:
                m = ak
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            try:
                ek = math.exp(a[k] - m)
            except OverflowError:
                raise ValueError("softmax exp overflowed")
            if not math.isfinite(ek):
                raise ValueError("softmax exp became non-finite")
            e[k] = ek
            d += ek
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        # u=r.random()*d；自 0.0 依 k 升序累加 e_k，选首个累计值严格大于
        # u 者；无则取词表末项。
        u = rng.random() * d
        if not math.isfinite(u):
            raise ValueError("sample threshold became non-finite")
        chosen = V - 1
        cum = 0.0
        for k in range(V):
            cum += e[k]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > u:
                chosen = k
                break

        out.append(vocab[chosen])
        x = chosen

    return "".join(out) + "\n"


def _rnn_n(Wxh, Whh, bh, x, h):
    """按 _perplexity 的公式、下标与累加顺序由 x、h 计算新隐状态 n。

    n_i = tanh(bh_i + Wxh_i,x + Σ_j Whh_i,j*h_j)：累加器自 float 偏置起，
    先加 Wxh_i,x，再依 j 升序加 Whh_i,j*h_j。任一累加或 tanh 结果非有限
    均抛 ValueError。
    """
    H = len(bh)
    n = [0.0] * H
    for i in range(H):
        acc = float(bh[i])
        acc += Wxh[i][x]
        if not math.isfinite(acc):
            raise ValueError("hidden affine accumulated non-finitely")
        wh_row = Whh[i]
        for j in range(H):
            acc += wh_row[j] * h[j]
            if not math.isfinite(acc):
                raise ValueError("hidden affine accumulated non-finitely")
        ni = math.tanh(acc)
        if not math.isfinite(ni):
            raise ValueError("tanh produced a non-finite value")
        n[i] = ni
    return n


def _output_logits(Why, by, hidden):
    """按 _perplexity 的下标与累加顺序计算输出 logit z。

    z_k = by_k + Σ_j Why_k,j*hidden_j，依 j 升序自 float 偏置累加。
    任一累加非有限均抛 ValueError。
    """
    V = len(by)
    H = len(hidden)
    z = [0.0] * V
    for k in range(V):
        acc = float(by[k])
        why_row = Why[k]
        for j in range(H):
            acc += why_row[j] * hidden[j]
            if not math.isfinite(acc):
                raise ValueError("output affine accumulated non-finitely")
        z[k] = acc
    return z


def _attn_context(n, memory):
    """以 attention([n], M_t, M_t, None) 的首行输出逐项加进 n，返回 u。

    逐项令 u_i = n_i + c[0][i]，相加结果非有限即抛 ValueError。
    """
    c, _w = attention([n], memory, memory, None)
    c0 = c[0]
    u = [0.0] * len(n)
    for i in range(len(n)):
        ui = n[i] + c0[i]
        if not math.isfinite(ui):
            raise ValueError("attention-adjusted hidden state became "
                             "non-finite")
        u[i] = ui
    return u


def _window_tail(memory, window_text):
    """返回 memory 末尾恰 min(WINDOW, len(memory)) 项的全新列表（行共享）。

    WINDOW 为已通过 _WINDOW_RE 词法校验的 [1-9][0-9]* 文本（任意位数均
    合法），全程不把该无界文本转为 int（避开 Python 的整数文本位数上限）；
    截断长度通过十进制位数与同长度字典序比较，与 len(memory) 的十进制文本
    比对得到。attention 不修改其输入行，故行对象可与 memory 共享。
    """
    n_items = len(memory)
    limit_text = str(n_items)
    # 位数多者数值大；同位数则字典序与数值序一致（两者均无前导零）。
    if len(window_text) > len(limit_text) or (
            len(window_text) == len(limit_text)
            and window_text >= limit_text):
        w = n_items
    else:
        # WINDOW < len(memory)：仅在此处需要实际整数。str(len(memory)) 受
        # 内存约束而有界，其位数必不超过此值，故 window_text 位数同样不超过
        # 上限，int() 不会触发整数文本位数限制。
        w = int(window_text)
    return list(memory[n_items - w:])


def _perplexity_attn(model_path, corpus_path, window_text):
    """带注意力上下文的困惑度，返回待写出的字符串。

    MODEL、CORPUS 完全沿用 perplexity 的读取、形状、F、UTF-8、词表及语料
    契约；WINDOW 整串匹配 [1-9][0-9]*（任意位数均合法，不转 int），否则
    抛 ValueError。

    置 h=h0、L=0.0、T=len(CORPUS)-1，t 升序（x、y 为当前、下一字符索引）：
    先按 _perplexity 的公式、下标与累加顺序由 x、h 求 n_t；M_t 取
    [h0, n_0, ..., n_{t-1}] 末尾至多 WINDOW 项（顺序从旧到新），以
    attention([n_t], M_t, M_t, None) 返回首项 c，逐项令
    u_i = n_t[i] + c[0][i]。logit 仅以 u 替代原隐状态 n，其余下标、稳定
    softmax、交叉熵累加顺序均与 _perplexity 相同；随后 h=n_t。任一运算
    非有限（含最终 exp(L/T) 溢出）均抛 ValueError。成功返回
    format(exp(L/T), '.17g') + '\\n'。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, Wxh, Whh, bh, Why, by, h0 = _load_perplexity_model(model_path)
    V = len(vocab)

    with open(corpus_path, "rb") as f:
        corpus = f.read().decode("utf-8")
    if len(corpus) < 2:
        raise ValueError("corpus must contain at least 2 codepoints")

    table = {ch: i for i, ch in enumerate(vocab)}
    ids = [0] * len(corpus)
    for t, ch in enumerate(corpus):
        ix = table.get(ch)
        if ix is None:
            raise ValueError("corpus contains an out-of-vocab character")
        ids[t] = ix

    # 记忆序列 [h0, n_0, ..., n_{t-1}]；h0 保留模型原值（F 允许 int），
    # 使 t=0 的 Whh*h 与 _perplexity 同为“原值先乘后加”；n 各项本就是
    # float。attention 不修改其输入，故直接共享行即可。
    memory = [h0]
    h = h0
    L = 0.0
    T = len(ids) - 1
    for t in range(T):
        x = ids[t]
        y = ids[t + 1]

        n = _rnn_n(Wxh, Whh, bh, x, h)
        M_t = _window_tail(memory, window_text)
        u = _attn_context(n, M_t)

        # logit 仅以 u 替代原隐状态；下标与累加顺序同 _perplexity。
        z = _output_logits(Why, by, u)

        # log-sum-exp：m=max(z)，d 从 0.0 依 k 升序累加 exp(z_k-m)。
        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError("softmax denominator accumulated non-finitely")

        step = m + math.log(d) - z[y]
        if not math.isfinite(step):
            raise ValueError("cross-entropy step is non-finite")
        L += step
        if not math.isfinite(L):
            raise ValueError("total cross-entropy accumulated non-finitely")
        h = n
        memory.append([float(v) for v in n])

    perplexity = math.exp(L / T)
    if not math.isfinite(perplexity):
        raise ValueError("perplexity is non-finite")
    return format(perplexity, ".17g") + "\n"


def _perplexity_lstm_attn(model_path, corpus_path, window_text):
    """LSTM 加注意力上下文的困惑度，返回待写出的字符串。

    MODEL、CORPUS 完全沿用 perplexity-lstm version 2 的八键顺序、F、形状、
    严格 UTF-8、词表及语料至少 2 码点契约；WINDOW 的词法与任意位数安全截取
    完全沿用 perplexity-attn（整串匹配 [1-9][0-9]*，不转 int），否则抛
    ValueError。

    置 h=h0、c=c0、L=0.0、T=len(CORPUS)-1、memory=[h0]，t 升序：以当前
    字符的 V 长 one-hot 为 x，调用装入 W、b 的 LSTMCell.forward(x, h, c)，
    取前两项更新 h、c；M 取 memory 末尾至多 WINDOW 项（顺序从旧到新），以
    attention([h], M, M, None) 返回首项 ctx，按 i 升序令
    u[i] = h[i] + ctx[0][i]。logit 仅以 u 替代 perplexity-lstm 中的 h，
    其余 Why/by 仿射、稳定 log-sum-exp 及下一字符负对数似然的下标与累加
    顺序均与 perplexity-lstm 相同；随后向 memory 追加 h 的 float 副本。任一
    运算非有限（含最终 exp(L/T) 溢出）均抛 ValueError。成功返回
    format(exp(L/T), '.17g') + '\\n'。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    with open(corpus_path, "rb") as f:
        corpus = f.read().decode("utf-8")
    if len(corpus) < 2:
        raise ValueError("corpus must contain at least 2 codepoints")

    table = {ch: i for i, ch in enumerate(vocab)}
    ids = [0] * len(corpus)
    for t, ch in enumerate(corpus):
        ix = table.get(ch)
        if ix is None:
            raise ValueError("corpus contains an out-of-vocab character")
        ids[t] = ix

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    # 记忆序列 [h0, h_0, ..., h_{t-1}]；h0 保留模型原值（F 允许 int），
    # attention 按原值“先乘后加”；各步 h 本就是 float。attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    c = list(c0)
    L = 0.0
    T = len(ids) - 1
    for t in range(T):
        y = ids[t + 1]

        # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # z_k = by_k + Σ_j Why_k,j*u_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, u)

        # log-sum-exp：m=max(z)，d 从 0.0 依 k 升序累加 exp(z_k-m)。
        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError("softmax denominator accumulated non-finitely")

        step = m + math.log(d) - z[y]
        if not math.isfinite(step):
            raise ValueError("cross-entropy step is non-finite")
        L += step
        if not math.isfinite(L):
            raise ValueError("total cross-entropy accumulated non-finitely")

        memory.append([float(v) for v in h])

    perplexity = math.exp(L / T)
    if not math.isfinite(perplexity):
        raise ValueError("perplexity is non-finite")
    return format(perplexity, ".17g") + "\n"


def _sample_attn(model_path, start, seed_text, temperature_text, length_text,
                 window_text):
    """带注意力上下文从 RNN 语言模型采样 LENGTH 个码点，返回待写出的字符串。

    除 WINDOW 外，MODEL、START、SEED、TEMPERATURE、LENGTH 的契约与 _sample
    完全一致；WINDOW 整串匹配 [1-9][0-9]*（任意位数均合法，不转 int），否则
    抛 ValueError。整次调用仅初始化一次 r=random.Random(int(SEED))，不写
    任何文件。

    置 h=h0、x=START 索引，记忆序列为 [h0, n_0, ..., n_{t-1}]。循环
    LENGTH 次：先按 _sample 的公式、下标与累加顺序由 x、h 求 n_t；M_t 取
    记忆末尾至多 WINDOW 项（顺序从旧到新），以 attention([n_t], M_t,
    M_t, None) 返回首项 c，逐项令 u_i = n_t[i] + c[0][i]。logit 仅以 u
    替代原隐状态 n，稳定 softmax 与按词表升序累计抽样的公式及运算顺序均
    与 _sample 相同；追加 vocab[k]，再令 h=n_t 并把 n_t 压入记忆。任一
    运算非有限均抛 ValueError。成功返回 LENGTH 个码点再加一个 LF。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, Wxh, Whh, bh, Why, by, h0 = _load_perplexity_model(model_path)
    V = len(vocab)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # SEED：整串匹配整数词法（禁止空白、+ 前缀、前导零、下划线）。
    if not _INT_RE.match(seed_text):
        raise ValueError("SEED must match 0|-?[1-9][0-9]*")
    seed = int(seed_text)

    # TEMPERATURE：float() 可解析且有限、严格大于 0；inf/nan/0/负数均失败。
    temperature = float(temperature_text)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("TEMPERATURE must be a finite positive float")

    # LENGTH：整串匹配非负整数词法。
    if not _NONNEG_INT_RE.match(length_text):
        raise ValueError("LENGTH must match 0|[1-9][0-9]*")
    length = int(length_text)

    rng = random.Random(seed)
    # 与 _perplexity_attn 相同：h0 保留模型原值，保证首步“原值先乘后加”。
    memory = [h0]
    h = h0
    x = vocab.index(start)
    out = []

    for _t in range(length):
        n = _rnn_n(Wxh, Whh, bh, x, h)
        M_t = _window_tail(memory, window_text)
        u = _attn_context(n, M_t)

        # logit 仅以 u 替代原隐状态；下标与累加顺序同 _sample_loop。
        z = _output_logits(Why, by, u)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        m = None
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
            if m is None or ak > m:
                m = ak
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            try:
                ek = math.exp(a[k] - m)
            except OverflowError:
                raise ValueError("softmax exp overflowed")
            if not math.isfinite(ek):
                raise ValueError("softmax exp became non-finite")
            e[k] = ek
            d += ek
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        # u=r.random()*d；自 0.0 依 k 升序累加 e_k，选首个累计值严格大于
        # u 者；无则取词表末项。
        threshold = rng.random() * d
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = V - 1
        cum = 0.0
        for k in range(V):
            cum += e[k]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = k
                break

        out.append(vocab[chosen])
        h = n
        x = chosen
        memory.append([float(v) for v in n])

    return "".join(out) + "\n"


def _sample_lstm_attn(model_path, start, seed_text, temperature_text,
                      length_text, window_text):
    """带注意力上下文从 LSTM 语言模型采样 LENGTH 个码点，返回待写出的字符串。

    MODEL、START、SEED、TEMPERATURE、LENGTH 的读取、词法、形状、F、确定性
    抽样及失败契约完全沿用 _sample_lstm；WINDOW 整串匹配 [1-9][0-9]*（任意
    位数均合法，不转 int），安全截取沿用 perplexity-lstm-attn，否则抛
    ValueError。整次调用仅初始化一次 r=random.Random(int(SEED))，不写任何
    文件。

    置 h=h0、c=c0、x=START 索引、memory=[h0]。循环 LENGTH 次：以 x 的
    V 长 one-hot 调用装入 W、b 的 LSTMCell.forward(x, h, c)，取前两项更新
    h、c；M 取 memory 末尾 min(WINDOW, len(memory)) 项（顺序从旧到新），
    以 attention([h], M, M, None) 返回首项 ctx，按 i 升序令
    u[i] = h[i] + ctx[0][i]。logit 仅以 u 替代 h，温度缩放、稳定 softmax、
    随机阈值、k 升序累计与选中规则逐项沿用 _sample_lstm；追加 vocab[k]，
    令 x=k，再向 memory 追加 h 的 float 副本。任一中间量非有限均抛
    ValueError。成功返回 LENGTH 个码点再加一个 LF。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # SEED：整串匹配整数词法（禁止空白、+ 前缀、前导零、下划线）。
    if not _INT_RE.match(seed_text):
        raise ValueError("SEED must match 0|-?[1-9][0-9]*")
    seed = int(seed_text)

    # TEMPERATURE：float() 可解析且有限、严格大于 0；inf/nan/0/负数均失败。
    temperature = float(temperature_text)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("TEMPERATURE must be a finite positive float")

    # LENGTH：整串匹配非负整数词法。
    if not _NONNEG_INT_RE.match(length_text):
        raise ValueError("LENGTH must match 0|[1-9][0-9]*")
    length = int(length_text)

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    rng = random.Random(seed)
    # 与 _perplexity_lstm_attn 相同：h0 保留模型原值，attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    c = list(c0)
    x = vocab.index(start)
    out = []

    for _t in range(length):
        # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, c = cell.forward(xvec, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # logit 仅以 u 替代 h；下标与累加顺序同 _sample_lstm。
        z = _output_logits(Why, by, u)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        m = None
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
            if m is None or ak > m:
                m = ak
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            try:
                ek = math.exp(a[k] - m)
            except OverflowError:
                raise ValueError("softmax exp overflowed")
            if not math.isfinite(ek):
                raise ValueError("softmax exp became non-finite")
            e[k] = ek
            d += ek
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        # u=r.random()*d；自 0.0 依 k 升序累加 e_k，选首个累计值严格大于
        # u 者；无则取词表末项。
        threshold = rng.random() * d
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = V - 1
        cum = 0.0
        for k in range(V):
            cum += e[k]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = k
                break

        out.append(vocab[chosen])
        x = chosen
        memory.append([float(v) for v in h])

    return "".join(out) + "\n"


def _sample_lstm_attn_anneal(model_path, start, seed_text, start_t_text,
                             end_t_text, length_text, window_text):
    """以线性退火温度、带注意力上下文从 LSTM 语言模型采样 LENGTH 个码点。

    除温度外，MODEL、START、SEED、LENGTH、WINDOW、LSTM 状态、注意力记忆、
    Why/by logit、稳定 softmax 及词表升序阈值抽样均沿用 _sample_lstm_attn；
    WINDOW 整串匹配 [1-9][0-9]*（任意位数均合法，不转 int），安全截取沿用
    perplexity-lstm-attn，否则抛 ValueError。START_T、END_T 各经 float()
    解析，结果须有限且严格大于 0。整次调用仅初始化一次
    r=random.Random(int(SEED))，不写任何文件。

    LENGTH 为 0 时不计算温度；为 1 时仅用 START_T；否则 t 自 0 升序，第
    t 步温度严格按 Python 表达式 START_T+(END_T-START_T)*t/(LENGTH-1)
    求值，结果非有限或不大于 0 即抛 ValueError。置 h=h0、c=c0、x=START
    索引、memory=[h0]，每步以该温度替换 _sample_lstm_attn 的固定温度，
    沿用同一 h、c、x 和 memory，生成后更新 x 并向 memory 追加 h 的 float
    副本。任一中间量非有限均抛 ValueError。成功返回 LENGTH 个码点再加
    一个 LF。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # SEED：整串匹配整数词法（禁止空白、+ 前缀、前导零、下划线）。
    if not _INT_RE.match(seed_text):
        raise ValueError("SEED must match 0|-?[1-9][0-9]*")
    seed = int(seed_text)

    # START_T、END_T：float() 可解析且有限、严格大于 0。
    start_t = float(start_t_text)
    if not math.isfinite(start_t) or start_t <= 0.0:
        raise ValueError("START_T must be a finite positive float")
    end_t = float(end_t_text)
    if not math.isfinite(end_t) or end_t <= 0.0:
        raise ValueError("END_T must be a finite positive float")

    # LENGTH：整串匹配非负整数词法。
    if not _NONNEG_INT_RE.match(length_text):
        raise ValueError("LENGTH must match 0|[1-9][0-9]*")
    length = int(length_text)

    if length == 1:
        # 仅用 START_T，不求值退火表达式。
        def temperature_at(t):
            return start_t
    elif length >= 2:
        def temperature_at(t):
            # 严格按 START_T+(END_T-START_T)*t/(LENGTH-1) 的运算顺序求值。
            temp = start_t + (end_t - start_t) * t / (length - 1)
            if not math.isfinite(temp) or temp <= 0.0:
                raise ValueError(
                    "annealed temperature became non-finite or non-positive")
            return temp
    else:
        temperature_at = None  # LENGTH 为 0：循环不执行，不计算温度。

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    rng = random.Random(seed)
    # 与 _sample_lstm_attn 相同：h0 保留模型原值，attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    c = list(c0)
    x = vocab.index(start)
    out = []

    for t in range(length):
        temperature = temperature_at(t)
        # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, c = cell.forward(xvec, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # logit 仅以 u 替代 h；下标与累加顺序同 _sample_lstm。
        z = _output_logits(Why, by, u)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        m = None
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
            if m is None or ak > m:
                m = ak
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            try:
                ek = math.exp(a[k] - m)
            except OverflowError:
                raise ValueError("softmax exp overflowed")
            if not math.isfinite(ek):
                raise ValueError("softmax exp became non-finite")
            e[k] = ek
            d += ek
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        # u=r.random()*d；自 0.0 依 k 升序累加 e_k，选首个累计值严格大于
        # u 者；无则取词表末项。
        threshold = rng.random() * d
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = V - 1
        cum = 0.0
        for k in range(V):
            cum += e[k]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = k
                break

        out.append(vocab[chosen])
        x = chosen
        memory.append([float(v) for v in h])

    return "".join(out) + "\n"


def _sample_lstm_attn_top_p(model_path, start, seed_text, start_t_text,
                            end_t_text, top_p_text, length_text,
                            window_text):
    """以线性退火温度、带注意力上下文与 top-p 截断从 LSTM 模型采样。

    除 TOP_P 及下述核采样规则外，MODEL、START、SEED、LENGTH、WINDOW、
    线性退火温度、LSTM 状态、注意力记忆、Why/by logit、温度缩放、稳定
    softmax 的 e_k 与 d、有限性检查及失败契约完全沿用
    _sample_lstm_attn_anneal；整次调用仅初始化一次
    r=random.Random(int(SEED))，不写任何文件。

    TOP_P 经 float() 解析，结果须有限且 0<TOP_P<=1，否则抛 ValueError。
    每步沿用原顺序算出温度缩放后的 e_k=exp(a_k-max(a))，d 自 0.0 依 k
    升序累加。将索引按 (-e_k, k) 升序排列，依此序自 0.0 累加 e，截取首个
    使累计值 >= TOP_P*d 的最短前缀；s 为该前缀按该序自 0.0 累加之和。每步
    令 u=r.random()*s，再按前缀顺序自 0.0 累加 e，选首个累计值严格大于 u
    的索引，无则选前缀末项，其字符作为下一输入。任一新增运算非有限均抛
    ValueError。成功返回 LENGTH 个码点再加一个 LF。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # SEED：整串匹配整数词法（禁止空白、+ 前缀、前导零、下划线）。
    if not _INT_RE.match(seed_text):
        raise ValueError("SEED must match 0|-?[1-9][0-9]*")
    seed = int(seed_text)

    # START_T、END_T：float() 可解析且有限、严格大于 0。
    start_t = float(start_t_text)
    if not math.isfinite(start_t) or start_t <= 0.0:
        raise ValueError("START_T must be a finite positive float")
    end_t = float(end_t_text)
    if not math.isfinite(end_t) or end_t <= 0.0:
        raise ValueError("END_T must be a finite positive float")

    # TOP_P：float() 可解析且有限、0<TOP_P<=1。
    top_p = float(top_p_text)
    if not math.isfinite(top_p) or top_p <= 0.0 or top_p > 1.0:
        raise ValueError("TOP_P must be a finite float with 0<TOP_P<=1")

    # LENGTH：整串匹配非负整数词法。
    if not _NONNEG_INT_RE.match(length_text):
        raise ValueError("LENGTH must match 0|[1-9][0-9]*")
    length = int(length_text)

    if length == 1:
        # 仅用 START_T，不求值退火表达式。
        def temperature_at(t):
            return start_t
    elif length >= 2:
        def temperature_at(t):
            # 严格按 START_T+(END_T-START_T)*t/(LENGTH-1) 的运算顺序求值。
            temp = start_t + (end_t - start_t) * t / (length - 1)
            if not math.isfinite(temp) or temp <= 0.0:
                raise ValueError(
                    "annealed temperature became non-finite or non-positive")
            return temp
    else:
        temperature_at = None  # LENGTH 为 0：循环不执行，不计算温度。

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    rng = random.Random(seed)
    # 与 _sample_lstm_attn 相同：h0 保留模型原值，attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    c = list(c0)
    x = vocab.index(start)
    out = []

    for t in range(length):
        temperature = temperature_at(t)
        # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, c = cell.forward(xvec, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # logit 仅以 u 替代 h；下标与累加顺序同 _sample_lstm。
        z = _output_logits(Why, by, u)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        m = None
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
            if m is None or ak > m:
                m = ak
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            try:
                ek = math.exp(a[k] - m)
            except OverflowError:
                raise ValueError("softmax exp overflowed")
            if not math.isfinite(ek):
                raise ValueError("softmax exp became non-finite")
            e[k] = ek
            d += ek
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        # 索引按 (-e_k, k) 升序排列，依此序自 0.0 累加 e，截取首个使累计值
        # >= TOP_P*d 的最短前缀；s 为该前缀按该序自 0.0 累加之和。
        cutoff = top_p * d
        if not math.isfinite(cutoff):
            raise ValueError("top-p cutoff became non-finite")
        order = sorted(range(V), key=lambda k: (-e[k], k))
        prefix = []
        s = 0.0
        for k in order:
            prefix.append(k)
            s += e[k]
            if not math.isfinite(s):
                raise ValueError("top-p prefix sum accumulated non-finitely")
            if s >= cutoff:
                break

        # u=r.random()*s；按前缀顺序自 0.0 累加 e，选首个累计值严格大于 u
        # 者；无则取前缀末项。
        threshold = rng.random() * s
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = prefix[-1]
        cum = 0.0
        for k in prefix:
            cum += e[k]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = k
                break

        out.append(vocab[chosen])
        x = chosen
        memory.append([float(v) for v in h])

    return "".join(out) + "\n"


def _train_attn(model_path, corpus_path, out_path, window_text):
    """带注意力上下文的一次全语料 SGD 更新并把新模型写入 OUT。

    MODEL、CORPUS、OUT 的八键 JSON、F/UTF-8 校验、一次全语料 SGD、0.1
    学习率与 5.0 全局裁剪均沿用 _train；WINDOW 整串匹配 [1-9][0-9]*
    （任意位数均合法，不转 int），否则抛 ValueError。

    前向严格复用 _perplexity_attn 的 N-1 步：以当前字符 one-hot 与 h0
    调用装入参数的 VanillaRNN.forward 得到 n_0..n_{T-1}；M_t 取
    [h0, n_0, ..., n_{t-1}] 末尾至多 WINDOW 项（顺序从旧到新），u_t 逐项
    等于 n_t 加 attention([n_t], M_t, M_t, None) 的首行上下文；logit 仅以
    u_t 替代隐状态，其余下标、稳定 softmax 顺序均与 _train 相同。

    令 g_t = p_t - onehot(y_t)，按 t 升序累加 dWhy += g_t⊗u_t、dby += g_t
    （组内下标顺序同 _train），并对每个 j 自 0.0 按 k 升序求
    du_t = Whyᵀg_t。置 dhs 为全零，按 t 降序调用
    attention_context_backward(n_t, M_t, du_t)：dn 按 i 升序加至 dhs[t]；
    再按 M_t 从旧到新、i 升序把 dmemory 各行加至其对应梯度——首行若为
    h0 则梯度丢弃，完整记忆中的第 q（q>=1）行对应 n_{q-1} 即 dhs[q-1]。
    任一累加非有限抛 ValueError。随后以同一 RNN 前向缓存调用
    VanillaRNN.backward(dhs)，裁剪、五组参数更新与 OUT 写出均同 _train，
    h0 不变。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, Wxh, Whh, bh, Why, by, h0 = _load_perplexity_model(model_path)
    V = len(vocab)
    H = len(bh)

    with open(corpus_path, "rb") as f:
        corpus = f.read().decode("utf-8")
    if len(corpus) < 2:
        raise ValueError("corpus must contain at least 2 codepoints")

    table = {ch: i for i, ch in enumerate(vocab)}
    ids = [0] * len(corpus)
    for t, ch in enumerate(corpus):
        ix = table.get(ch)
        if ix is None:
            raise ValueError("corpus contains an out-of-vocab character")
        ids[t] = ix

    T = len(ids) - 1

    # 当前字符的 one-hot 输入序列；n_0..n_{T-1} 由同一 RNN 前向产生并缓存。
    xs = []
    for t in range(T):
        row = [0.0] * V
        row[ids[t]] = 1.0
        xs.append(row)

    rnn = VanillaRNN(V, H)
    rnn.Wxh = [list(row) for row in Wxh]
    rnn.Whh = [list(row) for row in Whh]
    rnn.bh = list(bh)
    hs = rnn.forward(xs, h0)

    # 严格复用 _perplexity_attn 的记忆、M_t 与 u_t：memory 初值为 h0
    # 原值，每步后压入 n_t 的 float 副本；M_t 为末尾至多 WINDOW 项。
    memory = [h0]
    m_list = []
    u_list = []
    for t in range(T):
        n = hs[t]
        M_t = _window_tail(memory, window_text)
        u = _attn_context(n, M_t)
        m_list.append(M_t)
        u_list.append(u)
        memory.append([float(v) for v in n])

    dWhy = [[0.0] * H for _ in range(V)]
    dby = [0.0] * V
    du_list = [None] * T

    for t in range(T):
        u = u_list[t]
        y = ids[t + 1]

        # logit：仅以 u 替代 n，其余与 _train 相同。
        z = [0.0] * V
        for k in range(V):
            acc = float(by[k])
            why_row = Why[k]
            for j in range(H):
                acc += why_row[j] * u[j]
                if not math.isfinite(acc):
                    raise ValueError("output affine accumulated non-finitely")
            z[k] = acc

        # softmax：m=max(z)，d 从 0.0 依 k 升序累加 exp(z_k-m)。
        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        e = [0.0] * V
        d = 0.0
        for k in range(V):
            ev = math.exp(z[k] - m)
            if not math.isfinite(ev):
                raise ValueError("softmax exp became non-finite")
            e[k] = ev
            d += ev
            if not math.isfinite(d):
                raise ValueError("softmax denominator accumulated non-finitely")

        # g = p - onehot(y)。
        g = [0.0] * V
        for k in range(V):
            gk = e[k] / d
            if k == y:
                gk -= 1.0
            if not math.isfinite(gk):
                raise ValueError("output gradient became non-finite")
            g[k] = gk

        # 按 t 升序累加 dWhy += g⊗u、dby += g（组内顺序同 _train）。
        for k in range(V):
            gk = g[k]
            dby[k] += gk
            if not math.isfinite(dby[k]):
                raise ValueError("dby accumulated to a non-finite value")
            dw_row = dWhy[k]
            for j in range(H):
                dw_row[j] += gk * u[j]
                if not math.isfinite(dw_row[j]):
                    raise ValueError("dWhy accumulated to a non-finite value")

        # du = Whyᵀg：每个 j 独立以 0.0 起按 k 升序累加。
        du = [0.0] * H
        for j in range(H):
            acc = 0.0
            for k in range(V):
                acc += Why[k][j] * g[k]
                if not math.isfinite(acc):
                    raise ValueError("du accumulated to a non-finite value")
            du[j] = acc
        du_list[t] = du

    # 注意力残差与记忆的反向：dhs 置零，按 t 降序累加。
    dhs = [[0.0] * H for _ in range(T)]
    for t in range(T - 1, -1, -1):
        dn, dmemory = attention_context_backward(hs[t], m_list[t], du_list[t])

        # 先按 i 升序把 dn 加至 dhs[t]。
        dh_row = dhs[t]
        for i in range(H):
            dh_row[i] += dn[i]
            if not math.isfinite(dh_row[i]):
                raise ValueError("dhs accumulated to a non-finite value")

        # 再按 M_t 从旧到新：base 为其首行在完整记忆
        # [h0, n_0, ..., n_{t-1}] 中的下标；0 即 h0（梯度丢弃），
        # q>=1 对应 n_{q-1}。
        base = t + 1 - len(m_list[t])
        for p, dm_row in enumerate(dmemory):
            q = base + p
            if q == 0:
                continue
            target = dhs[q - 1]
            for i in range(H):
                target[i] += dm_row[i]
                if not math.isfinite(target[i]):
                    raise ValueError("dhs accumulated to a non-finite value")

    _dxs, dWxh, dWhh, dbh, _dh0 = rnn.backward(dhs)

    # 依 dWxh、dWhh、dbh、dWhy、dby 行序累加平方和求全局范数。
    sum_sq = 0.0
    for group in (dWxh, dWhh, dbh, dWhy, dby):
        if type(group[0]) is list:
            for row in group:
                for v in row:
                    if not math.isfinite(v):
                        raise ValueError("gradient is non-finite")
                    sum_sq += v * v
                    if not math.isfinite(sum_sq):
                        raise ValueError(
                            "global norm accumulated to a non-finite value")
        else:
            for v in group:
                if not math.isfinite(v):
                    raise ValueError("gradient is non-finite")
                sum_sq += v * v
                if not math.isfinite(sum_sq):
                    raise ValueError(
                        "global norm accumulated to a non-finite value")

    global_norm = math.sqrt(sum_sq)
    scale = 5.0 / global_norm if global_norm > 5.0 else 1.0

    # 五组参数减 0.1 倍（裁剪后的）梯度；h0 不变。结果非有限即失败。
    def _updated(old, grad):
        value = float(old) - 0.1 * (grad * scale)
        if not math.isfinite(value):
            raise ValueError("updated parameter became non-finite")
        return value

    new_Wxh = [[_updated(Wxh[i][j], dWxh[i][j]) for j in range(V)]
               for i in range(H)]
    new_Whh = [[_updated(Whh[i][j], dWhh[i][j]) for j in range(H)]
               for i in range(H)]
    new_bh = [_updated(bh[i], dbh[i]) for i in range(H)]
    new_Why = [[_updated(Why[k][j], dWhy[k][j]) for j in range(H)]
               for k in range(V)]
    new_by = [_updated(by[k], dby[k]) for k in range(V)]

    obj = {
        "version": 1,
        "vocab": vocab,
        "Wxh": new_Wxh,
        "Whh": new_Whh,
        "bh": new_bh,
        "Why": new_Why,
        "by": new_by,
        "h0": [float(v) for v in h0],
    }
    text = json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"
    with open(out_path, "wb") as f:
        f.write(text.encode("utf-8"))


def main(argv):
    """命令行入口：perplexity、train 与 sample 三个子命令。

    python seqmodel.py perplexity MODEL CORPUS：成功时 stdout 恰为
    format(exp(L/T), '.17g') + '\\n' 并返回 0。

    python seqmodel.py train MODEL CORPUS OUT：对模型做一次全语料 SGD
    并写出新模型，成功时 stdout 为空并返回 0。

    python seqmodel.py perplexity-lstm MODEL CORPUS：MODEL 为 version 2 的
    LSTM 模型（键 version、vocab、W、b、Why、by、h0、c0），以
    LSTMCell.forward 逐步推进隐状态与细胞状态，logit 与 log-sum-exp 规则
    同 perplexity；输出契约与 perplexity 相同，不写任何文件。

    python seqmodel.py train-lstm MODEL CORPUS OUT：对 version 2 的 LSTM
    模型做一次全语料 SGD 并写出新模型。t 升序以 (h0, c0) 为初态调用装入
    W、b 的 LSTMCell.forward 并缓存；输出层 g、dWhy、dby、dhs 沿用 train
    的公式与累加顺序（隐状态换为 LSTM 的 h），再以 backward_sequence 取得
    dW、db；梯度组序为 dW、db、dWhy、dby，全局范数、5.0 裁剪与 0.1 更新
    规则同 train，仅更新 W、b、Why、by，h0、c0 不变。成功时 stdout 为空
    并返回 0。

    python seqmodel.py sample MODEL START SEED TEMPERATURE LENGTH：成功时
    stdout 恰为 LENGTH 个采样码点的 UTF-8 编码再加一个 LF（LENGTH 为 0 时
    仅 LF），不写任何文件，返回 0。

    python seqmodel.py sample-lstm MODEL START SEED TEMPERATURE LENGTH：
    MODEL 为 version 2 的 LSTM 模型（键 version、vocab、W、b、Why、by、
    h0、c0），置 h=h0、c=c0、x=START 索引，每步以 x 的 V 长 one-hot 调用
    装入 W、b 的 LSTMCell.forward 推进 h、c，logit、温度缩放、稳定
    softmax 与抽样规则同 sample；输出契约与 sample 相同，不写任何文件。

    python seqmodel.py sample-anneal MODEL START SEED START_T END_T LENGTH：
    以线性退火温度采样，第 t 步温度为
    START_T+(END_T-START_T)*t/(LENGTH-1)（LENGTH 为 1 时仅用 START_T，
    为 0 时不计算温度）；输出契约与 sample 相同。

    python seqmodel.py sample-lstm-anneal MODEL START SEED START_T END_T
    LENGTH：以线性退火温度从 version 2 的 LSTM 模型采样，每步以 h0、c0
    为初态、x 的 V 长 one-hot 调用装入 W、b 的 LSTMCell.forward 推进
    h、c，logit、稳定 softmax 与抽样规则同 sample-lstm，温度退火规则同
    sample-anneal；整次仅初始化一次随机源；输出契约与 sample 相同，不写
    任何文件。

    python seqmodel.py perplexity-attn MODEL CORPUS WINDOW：每步先按
    perplexity 的 RNN 公式求 n_t，再以记忆 [h0, n_0, ..., n_{t-1}] 末尾
    至多 WINDOW 项（从旧到新）为键/值调用 attention，用 n_t 与首行上下
    文之和作为 logit 隐状态，随后 h=n_t；输出契约与 perplexity 相同。

    python seqmodel.py perplexity-lstm-attn MODEL CORPUS WINDOW：MODEL 沿用
    perplexity-lstm 的 version 2 八键、F、形状与严格 UTF-8 契约；WINDOW
    沿用 perplexity-attn 的词法及任意位数安全截取。置 h=h0、c=c0、
    memory=[h0]，t 升序以当前字符 one-hot 调用装入 W、b 的
    LSTMCell.forward 更新 h、c，再以 memory 末尾至多 WINDOW 项（从旧到新）
    为键/值调用 attention，用 h 与首行上下文之和作为 logit 隐状态，随后向
    memory 追加 h 的 float 副本；输出契约与 perplexity 相同，不写文件。

    python seqmodel.py sample-attn MODEL START SEED TEMPERATURE LENGTH
    WINDOW：以同样的注意力上下文替换 logit 隐状态，softmax 与抽样契约
    与 sample 相同，整次仅初始化一次随机源、不写文件。WINDOW 整串匹配
    [1-9][0-9]*。

    python seqmodel.py sample-lstm-attn MODEL START SEED TEMPERATURE
    LENGTH WINDOW：MODEL、START、SEED、TEMPERATURE、LENGTH 沿用
    sample-lstm，WINDOW 沿用 perplexity-lstm-attn 的词法及任意位数安全
    截取。置 h=h0、c=c0、x=START 索引、memory=[h0]，每步以 x 的 V 长
    one-hot 调用装入 W、b 的 LSTMCell.forward 更新 h、c，以 memory 末尾
    min(WINDOW, len(memory)) 项（从旧到新）为键/值调用 attention，用
    h 与首行上下文之和 u 替代 h 计算 logit 并抽样，随后向 memory 追加
    h 的 float 副本；输出契约与 sample 相同，不写文件。

    python seqmodel.py sample-lstm-attn-anneal MODEL START SEED START_T
    END_T LENGTH WINDOW：以线性退火温度、带注意力上下文从 version 2 的
    LSTM 模型采样。除温度外，MODEL、START、SEED、LENGTH、WINDOW、LSTM
    状态、注意力记忆、logit、稳定 softmax 与抽样规则均沿用
    sample-lstm-attn；温度退火规则同 sample-anneal（LENGTH 为 1 时仅用
    START_T，为 0 时不计算温度，否则第 t 步为
    START_T+(END_T-START_T)*t/(LENGTH-1) 且每步须有限并大于 0）；整次
    仅初始化一次随机源；输出契约与 sample 相同，不写文件。

    python seqmodel.py sample-lstm-attn-top-p MODEL START SEED START_T
    END_T TOP_P LENGTH WINDOW：以线性退火温度、带注意力上下文与 top-p
    截断从 version 2 的 LSTM 模型采样。除 TOP_P 及核采样外，参数校验、
    LENGTH 为 0/1 语义、线性温度、LSTM 状态、注意力记忆、Why/by logit、
    有限性与错误协议完全沿用 sample-lstm-attn-anneal。TOP_P 经 float()
    解析，须有限且 0<TOP_P<=1。每步沿用原顺序算出温度缩放后的
    e_k=exp(a_k-max(a))，d 自 0.0 依 k 升序累加；将索引按 (-e_k, k)
    升序排列，依此序自 0.0 累加 e，截取首个使累计值 >= TOP_P*d 的最短
    前缀，s 为其按该序自 0.0 累加之和；整次仅初始化一次随机源，每步令
    u=random()*s，再按前缀顺序自 0.0 累加 e，选首个累计值严格大于 u 的
    索引，无则选前缀末项，其字符作为下一输入；任一新增运算非有限即按
    原错误协议失败。输出契约与 sample 相同，不写文件。

    python seqmodel.py train-attn MODEL CORPUS OUT WINDOW：前向严格复用
    perplexity-attn 的 n_t、M_t、u_t 与 logit 顺序；反向令
    g_t = p_t-onehot(y_t)，按 t 升序累加 dWhy、dby（以 u_t 为隐状态），
    按 t 降序经 attention_context_backward 把梯度映射回 dhs（h0 梯度丢弃），
    再以同一 RNN 前向缓存调用 VanillaRNN.backward，裁剪、更新与 train 相同，
    h0 不变。

    参数数量、词法、文件读取、UTF-8/JSON 解析、模型/语料校验或非有限计算等
    任何失败均返回 2，stdout 为空，且 stderr 恰为 "error\\n"，不输出回溯。
    """
    try:
        if len(argv) == 4 and argv[1] == "perplexity":
            output = _perplexity(argv[2], argv[3])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 5 and argv[1] == "train":
            _train(argv[2], argv[3], argv[4])
        elif len(argv) == 4 and argv[1] == "perplexity-lstm":
            output = _perplexity_lstm(argv[2], argv[3])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 5 and argv[1] == "train-lstm":
            _train_lstm(argv[2], argv[3], argv[4])
        elif len(argv) == 7 and argv[1] == "sample":
            output = _sample(argv[2], argv[3], argv[4], argv[5], argv[6])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 7 and argv[1] == "sample-lstm":
            output = _sample_lstm(argv[2], argv[3], argv[4], argv[5],
                                  argv[6])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 8 and argv[1] == "sample-anneal":
            output = _sample_anneal(argv[2], argv[3], argv[4], argv[5],
                                    argv[6], argv[7])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 8 and argv[1] == "sample-lstm-anneal":
            output = _sample_lstm_anneal(argv[2], argv[3], argv[4], argv[5],
                                         argv[6], argv[7])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 5 and argv[1] == "perplexity-attn":
            output = _perplexity_attn(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 5 and argv[1] == "perplexity-lstm-attn":
            output = _perplexity_lstm_attn(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 6 and argv[1] == "train-attn":
            _train_attn(argv[2], argv[3], argv[4], argv[5])
        elif len(argv) == 8 and argv[1] == "sample-attn":
            output = _sample_attn(argv[2], argv[3], argv[4], argv[5],
                                  argv[6], argv[7])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 8 and argv[1] == "sample-lstm-attn":
            output = _sample_lstm_attn(argv[2], argv[3], argv[4], argv[5],
                                       argv[6], argv[7])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 9 and argv[1] == "sample-lstm-attn-anneal":
            output = _sample_lstm_attn_anneal(argv[2], argv[3], argv[4],
                                              argv[5], argv[6], argv[7],
                                              argv[8])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 10 and argv[1] == "sample-lstm-attn-top-p":
            output = _sample_lstm_attn_top_p(argv[2], argv[3], argv[4],
                                             argv[5], argv[6], argv[7],
                                             argv[8], argv[9])
            sys.stdout.buffer.write(output.encode("utf-8"))
        else:
            raise ValueError(
                "usage: seqmodel.py perplexity MODEL CORPUS | "
                "seqmodel.py train MODEL CORPUS OUT | "
                "seqmodel.py sample MODEL START SEED TEMPERATURE LENGTH | "
                "seqmodel.py sample-anneal MODEL START SEED START_T END_T "
                "LENGTH")
    except Exception:
        sys.stderr.buffer.write(b"error\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
