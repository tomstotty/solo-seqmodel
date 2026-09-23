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
import os
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


def _multihead_check_heads(heads, D, Dv):
    """multihead_attention(_backward) 共用的 heads 契约校验。

    heads 的 type 须恰为 int（不含 bool）且为正，并同时整除查询维 D 与值维
    Dv，否则抛 ValueError。
    """
    if type(heads) is bool or type(heads) is not int or heads <= 0:
        raise ValueError(
            "heads must be a non-bool positive int, got %r" % (heads,))
    if D % heads != 0 or Dv % heads != 0:
        raise ValueError(
            "heads (%d) must divide both query dim D=%d and value dim Dv=%d"
            % (heads, D, Dv))


def _slice_columns(matrix, rows, width, offset):
    """取 matrix 各行 [offset, offset+width) 的逐行新建子矩阵。"""
    return [[matrix[i][offset + d] for d in range(width)] for i in range(rows)]


def multihead_attention(q, k, v, heads, mask=None):
    """多头缩放点积注意力，返回 (c, w)，不修改或复用任何输入。

    q、k、v、mask 完全沿用 attention 的契约（q 为 Tq×D、k 为 Tk×D、v 为
    Tk×Dv 的非空 F 列表矩阵，mask 为 None 或元素 type 恰为 bool 的
    Tq×Tk 矩阵，全屏蔽行抛 ValueError）；heads 须为非 bool 的正 int，并
    同时整除 D 与 Dv，否则抛 ValueError。实参数量错误沿用 Python 自带的
    TypeError。

    按连续列将 q、k 均分为 heads 个 D/heads 维子矩阵、v 均分为 heads 个
    Dv/heads 维子矩阵，各头按索引升序以同一 mask 调用 attention；各头
    上下文按头序、维序拼回 Tq×Dv 的 c，w 形状为 heads×Tq×Tk
    （w[h] 即第 h 头的注意力权重）。所有输出逐层新建、元素均为 float。
    任一校验或中间有限性失败均抛 ValueError；相同输入结果确定。
    """
    Tq, Tk, D, Dv, active = _attention_check(q, k, v, mask)
    _multihead_check_heads(heads, D, Dv)

    Hd = D // heads
    Hdv = Dv // heads
    c = [[0.0] * Dv for _ in range(Tq)]
    w = []
    for h in range(heads):
        qh = _slice_columns(q, Tq, Hd, h * Hd)
        kh = _slice_columns(k, Tk, Hd, h * Hd)
        vh = _slice_columns(v, Tk, Hdv, h * Hdv)
        ch, wh = attention(qh, kh, vh, active)

        # c 按头序拼回；attention 已保证有限，此处再经 float 转换与有限性
        # 校验后写入全新行。
        coff = h * Hdv
        for i in range(Tq):
            chi = ch[i]
            crow = c[i]
            for a in range(Hdv):
                cv = float(chi[a])
                if not math.isfinite(cv):
                    raise ValueError(
                        "multihead context became non-finite")
                crow[coff + a] = cv
        w.append(wh)

    return c, w


def multihead_attention_backward(q, k, v, dc, heads, mask=None):
    """多头缩放点积注意力的反向传播，返回 (dq, dk, dv)，不修改或复用输入。

    q、k、v、heads、mask 完全沿用 multihead_attention 的契约；dc 须为
    Tq×Dv 的 F 列表矩阵，否则抛 ValueError。实参数量错误沿用 Python 自带
    的 TypeError。

    按连续列切分 q、k、v（各头维度同前向），dc 同样按 Dv/heads 切分，各
    头按索引升序调用 attention_backward；dq、dk、dv 按各梯度原来的列位拼
    回，形状分别同 q、k、v。所有结果逐层新建、元素均为 float。任一校验或
    中间有限性失败均抛 ValueError；相同输入结果确定。
    """
    Tq, Tk, D, Dv, active = _attention_check(q, k, v, mask)
    _multihead_check_heads(heads, D, Dv)

    # dc：Tq×Dv 的 F 列表矩阵（逐行浅拷贝，读取用，不修改原输入）。
    dcc = _check_matrix(dc, Tq, Dv, "dc")

    Hd = D // heads
    Hdv = Dv // heads
    dq = [[0.0] * D for _ in range(Tq)]
    dk = [[0.0] * D for _ in range(Tk)]
    dv = [[0.0] * Dv for _ in range(Tk)]
    for h in range(heads):
        qh = _slice_columns(q, Tq, Hd, h * Hd)
        kh = _slice_columns(k, Tk, Hd, h * Hd)
        vh = _slice_columns(v, Tk, Hdv, h * Hdv)
        dch = _slice_columns(dcc, Tq, Hdv, h * Hdv)
        dqh, dkh, dvh = attention_backward(qh, kh, vh, dch, active)

        # 各头梯度按原列位拼回；attention_backward 已保证有限，此处再经
        # float 转换与有限性校验。
        qoff = h * Hd
        for i in range(Tq):
            dqhi = dqh[i]
            dqi = dq[i]
            for d in range(Hd):
                gv = float(dqhi[d])
                if not math.isfinite(gv):
                    raise ValueError("dq became non-finite")
                dqi[qoff + d] = gv
        for j in range(Tk):
            dkhj = dkh[j]
            dkj = dk[j]
            dvhj = dvh[j]
            dvj = dv[j]
            for d in range(Hd):
                gv = float(dkhj[d])
                if not math.isfinite(gv):
                    raise ValueError("dk became non-finite")
                dkj[qoff + d] = gv
            voff = h * Hdv
            for a in range(Hdv):
                gv = float(dvhj[a])
                if not math.isfinite(gv):
                    raise ValueError("dv became non-finite")
                dvj[voff + a] = gv

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


class MHA(object):
    """可训练的投影多头自注意力（四组 D 阶投影均初始化为单位矩阵）。

    Wq、Wk、Wv、Wo 各自独立，形状均为 D×D 的 float 单位矩阵（对角 1.0、
    其余 0.0），互不共享行或别名。D、heads 须为非 bool 的正 int 且 heads
    整除 D，否则抛 ValueError。
    """

    def __init__(self, D, heads):
        if type(D) is bool or type(D) is not int or D <= 0:
            raise ValueError("D must be a non-bool positive int, got %r"
                             % (D,))
        if type(heads) is bool or type(heads) is not int or heads <= 0:
            raise ValueError(
                "heads must be a non-bool positive int, got %r" % (heads,))
        if D % heads != 0:
            raise ValueError(
                "heads (%d) must divide D=%d" % (heads, D))
        self.D = D
        self.heads = heads
        # 四个投影各自逐层新建，互不共享。
        self.Wq = [[1.0 if a == j else 0.0 for j in range(D)]
                   for a in range(D)]
        self.Wk = [[1.0 if a == j else 0.0 for j in range(D)]
                   for a in range(D)]
        self.Wv = [[1.0 if a == j else 0.0 for j in range(D)]
                   for a in range(D)]
        self.Wo = [[1.0 if a == j else 0.0 for j in range(D)]
                   for a in range(D)]
        self._cache = None

    def _check_x(self, x):
        """校验 x 为非空 T×D 的 F 矩阵，返回逐行新建的 float 拷贝。"""
        D = self.D
        if type(x) is not list or len(x) == 0:
            raise ValueError("x must be a non-empty list")
        T = len(x)
        xc = []
        for row in x:
            if type(row) is not list or len(row) != D:
                raise ValueError("x must be a list of shape %d×%d" % (T, D))
            nrow = []
            for v in row:
                if not _is_f(v):
                    raise ValueError(
                        "x entries must be finite numbers, got %r" % (v,))
                nrow.append(float(v))
            xc.append(nrow)
        return xc

    @staticmethod
    def _project(W, x, T, D, what):
        """q[t][a]=Σ_j W[a][j]*x[t][j]（j 自 0.0 升序），返回全新 float 矩阵。"""
        out = [[0.0] * D for _ in range(T)]
        for t in range(T):
            xt = x[t]
            row_out = out[t]
            for a in range(D):
                Wa = W[a]
                acc = 0.0
                for j in range(D):
                    acc += Wa[j] * xt[j]
                    if not math.isfinite(acc):
                        raise ValueError(
                            "%s projection accumulated to a non-finite value"
                            % what)
                row_out[a] = acc
        return out

    def forward(self, x, mask=None):
        """投影多头自注意力前向，返回 (y, w) 并缓存反向所需快照。

        x 须为非空 T×D 的 F 矩阵；mask 须为 None 或 T×T、元素 type 恰为
        bool 且每行至少一个 True 的矩阵；self.Wq/Wk/Wv/Wo 须仍为 D×D 的 F
        矩阵，否则抛 ValueError。依次求 q=xWqᵀ、k=xWkᵀ、v=xWvᵀ，调用
        multihead_attention(q, k, v, heads, mask) 得 (c, w)，再求 y=cWoᵀ。
        y 为 T×D、w 为 heads×T×T 的逐层新建 float 列表，不修改或复用输入
        与投影属性。任一中间量非有限抛 ValueError；任何失败都清空缓存。
        实参数量错误沿用 Python 自带的 TypeError。
        """
        D, heads = self.D, self.heads
        # 任何失败的 forward 都使既有缓存失效。
        self._cache = None
        T = 0
        try:
            xc = self._check_x(x)
            T = len(xc)
            Wq = _check_matrix(self.Wq, D, D, "Wq")
            Wk = _check_matrix(self.Wk, D, D, "Wk")
            Wv = _check_matrix(self.Wv, D, D, "Wv")
            Wo = _check_matrix(self.Wo, D, D, "Wo")
            Wq = [[float(v) for v in row] for row in Wq]
            Wk = [[float(v) for v in row] for row in Wk]
            Wv = [[float(v) for v in row] for row in Wv]
            Wo = [[float(v) for v in row] for row in Wo]

            q = self._project(Wq, xc, T, D, "q")
            k = self._project(Wk, xc, T, D, "k")
            v = self._project(Wv, xc, T, D, "v")

            # mask 的形状、bool 类型与全屏蔽行契约由 multihead_attention 沿用
            # attention 的同一校验保证。
            c, w = multihead_attention(q, k, v, heads, mask)

            # y[t][a] = Σ_j Wo[a][j]*c[t][j]，j 自 0.0 升序。
            y = [[0.0] * D for _ in range(T)]
            for t in range(T):
                ct = c[t]
                yt = y[t]
                for a in range(D):
                    Woa = Wo[a]
                    acc = 0.0
                    for j in range(D):
                        acc += Woa[j] * ct[j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "output projection accumulated to a "
                                "non-finite value")
                    yt[a] = acc

            # mask 独立快照：None 保持 None，否则逐行新建 bool 拷贝。
            if mask is None:
                mask_c = None
            else:
                mask_c = [list(mask[t]) for t in range(T)]
        except ValueError:
            self._cache = None
            raise

        self._cache = ("mha", T, D, heads, xc, q, k, v, c, w, mask_c,
                       Wq, Wk, Wv, Wo)
        return y, w

    def backward(self, dy):
        """投影多头自注意力反向，返回 (dx, dWq, dWk, dWv, dWo)。

        须紧随一次成功的 forward（其后缓存未被失败清空），否则抛
        ValueError；dy 须为与前向 y 同形的 T×D F 矩阵，否则抛 ValueError。
        先按链式求
            dWo[a][j] = Σ_t dy[t][a]*c[t][j]
            dc[t][j]  = Σ_a dy[t][a]*Wo[a][j]
        （求和下标分别按 t、a 自 0.0 升序），调用
        multihead_attention_backward(q, k, v, dc, heads, mask) 得
        (dq, dk, dv)，再求
            dWq[a][j] = Σ_t dq[t][a]*x[t][j]（dWk、dWv 同理）
            dx[t][j]  = Σ_a (dq[t][a]Wq[a][j] + dk[t][a]Wk[a][j]
                             + dv[t][a]Wv[a][j])
        （求和下标按 a 自 0.0 升序，三条路径之和）。dx 为 T×D，其余四个为
        D×D 的逐层新建 float 列表，不修改 dy、缓存或投影属性；成功时缓存
        保留，重复调用结果相同。任一中间量非有限或反向失败均抛 ValueError
        并清空缓存。实参数量错误沿用 Python 自带的 TypeError。
        """
        cache = self._cache
        try:
            if type(cache) is not tuple or len(cache) != 15 \
                    or cache[0] != "mha":
                raise ValueError(
                    "backward requires a successful forward pass before it")
            (_, T, D, heads, xc, q, k, v, c, w, mask_c,
             Wq, Wk, Wv, Wo) = cache
            dy = _check_matrix(dy, T, D, "dy")
            dy = [[float(z) for z in row] for row in dy]

            # dWo[a][j] = Σ_t dy[t][a]*c[t][j]，t 自 0.0 升序。
            dWo = [[0.0] * D for _ in range(D)]
            for a in range(D):
                dWoa = dWo[a]
                for j in range(D):
                    acc = 0.0
                    for t in range(T):
                        acc += dy[t][a] * c[t][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dWo accumulated to a non-finite value")
                    dWoa[j] = acc

            # dc[t][j] = Σ_a dy[t][a]*Wo[a][j]，a 自 0.0 升序。
            dc = [[0.0] * D for _ in range(T)]
            for t in range(T):
                dyt = dy[t]
                dct = dc[t]
                for j in range(D):
                    acc = 0.0
                    for a in range(D):
                        acc += dyt[a] * Wo[a][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dc accumulated to a non-finite value")
                    dct[j] = acc

            dq, dk, dv = multihead_attention_backward(
                q, k, v, dc, heads, mask_c)

            # dWq/dWk/dWv[a][j] = Σ_t d{q,k,v}[t][a]*x[t][j]，t 升序。
            dWq = [[0.0] * D for _ in range(D)]
            dWk = [[0.0] * D for _ in range(D)]
            dWv = [[0.0] * D for _ in range(D)]
            for a in range(D):
                dWqa, dWka, dWva = dWq[a], dWk[a], dWv[a]
                for j in range(D):
                    aq = ak = av = 0.0
                    for t in range(T):
                        aq += dq[t][a] * xc[t][j]
                        if not math.isfinite(aq):
                            raise ValueError(
                                "dWq accumulated to a non-finite value")
                        ak += dk[t][a] * xc[t][j]
                        if not math.isfinite(ak):
                            raise ValueError(
                                "dWk accumulated to a non-finite value")
                        av += dv[t][a] * xc[t][j]
                        if not math.isfinite(av):
                            raise ValueError(
                                "dWv accumulated to a non-finite value")
                    dWqa[j], dWka[j], dWva[j] = aq, ak, av

            # dx[t][j]：q、k、v 三路径之和，按 a 自 0.0 升序累加。
            dx = [[0.0] * D for _ in range(T)]
            for t in range(T):
                dqt, dkt, dvt = dq[t], dk[t], dv[t]
                dxt = dx[t]
                for j in range(D):
                    acc = 0.0
                    for a in range(D):
                        acc += dqt[a] * Wq[a][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dx accumulated to a non-finite value")
                        acc += dkt[a] * Wk[a][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dx accumulated to a non-finite value")
                        acc += dvt[a] * Wv[a][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dx accumulated to a non-finite value")
                    dxt[j] = acc
        except ValueError:
            self._cache = None
            raise

        return dx, dWq, dWk, dWv, dWo

    def forward_cross(self, qx, kvx, mask=None):
        """投影多头交叉注意力前向，返回 (y, w) 并缓存反向所需快照。

        qx 须为非空 Tq×D、kvx 须为非空 Tk×D 的 F 矩阵；mask 须为 None 或
        Tq×Tk、元素 type 恰为 bool 且每行至少一个 True 的矩阵；
        self.Wq/Wk/Wv/Wo 须仍为 D×D 的 F 矩阵，否则抛 ValueError（校验方式
        与 forward 一致，mask 的形状与全屏蔽行契约由 multihead_attention
        沿用 attention 的同一校验保证）。依次求 q=qxWqᵀ、k=kvxWkᵀ、
        v=kvxWvᵀ，调用 multihead_attention(q, k, v, heads, mask) 得 (c, w)，
        再求 y=cWoᵀ。y 为 Tq×D、w 为 heads×Tq×Tk 的逐层新建 float 列表，
        不修改或复用输入与投影属性。任一中间量非有限抛 ValueError；任何失败
        都清空缓存。实参数量错误沿用 Python 自带的 TypeError。
        """
        D, heads = self.D, self.heads
        # 任何失败的 forward_cross 都使既有缓存失效。
        self._cache = None
        try:
            qxc = self._check_x(qx)
            Tq = len(qxc)
            kvxc = self._check_x(kvx)
            Tk = len(kvxc)
            Wq = _check_matrix(self.Wq, D, D, "Wq")
            Wk = _check_matrix(self.Wk, D, D, "Wk")
            Wv = _check_matrix(self.Wv, D, D, "Wv")
            Wo = _check_matrix(self.Wo, D, D, "Wo")
            Wq = [[float(v) for v in row] for row in Wq]
            Wk = [[float(v) for v in row] for row in Wk]
            Wv = [[float(v) for v in row] for row in Wv]
            Wo = [[float(v) for v in row] for row in Wo]

            q = self._project(Wq, qxc, Tq, D, "q")
            k = self._project(Wk, kvxc, Tk, D, "k")
            v = self._project(Wv, kvxc, Tk, D, "v")

            c, w = multihead_attention(q, k, v, heads, mask)

            # y[t][a] = Σ_j Wo[a][j]*c[t][j]，j 自 0.0 升序。
            y = [[0.0] * D for _ in range(Tq)]
            for t in range(Tq):
                ct = c[t]
                yt = y[t]
                for a in range(D):
                    Woa = Wo[a]
                    acc = 0.0
                    for j in range(D):
                        acc += Woa[j] * ct[j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "output projection accumulated to a "
                                "non-finite value")
                    yt[a] = acc

            # mask 独立快照：None 保持 None，否则逐行新建 bool 拷贝。
            if mask is None:
                mask_c = None
            else:
                mask_c = [list(mask[t]) for t in range(Tq)]
        except ValueError:
            self._cache = None
            raise

        self._cache = ("mha_cross", Tq, Tk, D, heads, qxc, kvxc, q, k, v,
                       c, w, mask_c, Wq, Wk, Wv, Wo)
        return y, w

    def backward_cross(self, dy):
        """投影多头交叉注意力反向，返回 (dqx, dkvx, dWq, dWk, dWv, dWo)。

        须紧随一次成功的 forward_cross（其后缓存未被任何其他 forward 类调用
        替换或被失败清空），否则抛 ValueError；dy 须为与前向 y 同形的
        Tq×D F 矩阵，否则抛 ValueError。按 backward 相同的公式与次序先求
            dWo[a][j] = Σ_t dy[t][a]*c[t][j]
            dc[t][j]  = Σ_a dy[t][a]*Wo[a][j]
        （求和下标分别按 t、a 自 0.0 升序），调用
        multihead_attention_backward(q, k, v, dc, heads, mask) 得
        (dq, dk, dv)，再求
            dWq[a][j] = Σ_t dq[t][a]*qx[t][j]（t 自 0.0 升序）
            dWk[a][j] = Σ_s dk[s][a]*kvx[s][j]（s 自 0.0 升序）
            dWv[a][j] = Σ_s dv[s][a]*kvx[s][j]
            dqx[t][j]  = Σ_a dq[t][a]*Wq[a][j]（a 升序）
            dkvx[s][j] = Σ_a (dk[s][a]*Wk[a][j] + dv[s][a]*Wv[a][j])
        （求和下标按 a 自 0.0 升序，k、v 两条路径之和）。dqx 为 Tq×D、
        dkvx 为 Tk×D，其余四个为 D×D 的逐层新建 float 列表，不修改 dy、缓存
        或投影属性；成功时缓存保留，重复调用结果相同。任一中间量非有限或反向
        失败均抛 ValueError 并清空缓存。实参数量错误沿用 Python 自带的
        TypeError。
        """
        cache = self._cache
        try:
            if type(cache) is not tuple or len(cache) != 17 \
                    or cache[0] != "mha_cross":
                raise ValueError(
                    "backward_cross requires a successful forward_cross "
                    "pass before it")
            (_, Tq, Tk, D, heads, qxc, kvxc, q, k, v, c, w, mask_c,
             Wq, Wk, Wv, Wo) = cache
            dy = _check_matrix(dy, Tq, D, "dy")
            dy = [[float(z) for z in row] for row in dy]

            # dWo[a][j] = Σ_t dy[t][a]*c[t][j]，t 自 0.0 升序。
            dWo = [[0.0] * D for _ in range(D)]
            for a in range(D):
                dWoa = dWo[a]
                for j in range(D):
                    acc = 0.0
                    for t in range(Tq):
                        acc += dy[t][a] * c[t][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dWo accumulated to a non-finite value")
                    dWoa[j] = acc

            # dc[t][j] = Σ_a dy[t][a]*Wo[a][j]，a 自 0.0 升序。
            dc = [[0.0] * D for _ in range(Tq)]
            for t in range(Tq):
                dyt = dy[t]
                dct = dc[t]
                for j in range(D):
                    acc = 0.0
                    for a in range(D):
                        acc += dyt[a] * Wo[a][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dc accumulated to a non-finite value")
                    dct[j] = acc

            dq, dk, dv = multihead_attention_backward(
                q, k, v, dc, heads, mask_c)

            # dWq[a][j] = Σ_t dq[t][a]*qx[t][j]，t 自 0.0 升序。
            dWq = [[0.0] * D for _ in range(D)]
            for a in range(D):
                dWqa = dWq[a]
                for j in range(D):
                    acc = 0.0
                    for t in range(Tq):
                        acc += dq[t][a] * qxc[t][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dWq accumulated to a non-finite value")
                    dWqa[j] = acc

            # dWk/dWv[a][j] = Σ_s d{k,v}[s][a]*kvx[s][j]，s 自 0.0 升序。
            dWk = [[0.0] * D for _ in range(D)]
            dWv = [[0.0] * D for _ in range(D)]
            for a in range(D):
                dWka, dWva = dWk[a], dWv[a]
                for j in range(D):
                    ak = av = 0.0
                    for s in range(Tk):
                        ak += dk[s][a] * kvxc[s][j]
                        if not math.isfinite(ak):
                            raise ValueError(
                                "dWk accumulated to a non-finite value")
                        av += dv[s][a] * kvxc[s][j]
                        if not math.isfinite(av):
                            raise ValueError(
                                "dWv accumulated to a non-finite value")
                    dWka[j], dWva[j] = ak, av

            # dqx[t][j] = Σ_a dq[t][a]*Wq[a][j]，a 自 0.0 升序。
            dqx = [[0.0] * D for _ in range(Tq)]
            for t in range(Tq):
                dqt = dq[t]
                dqxt = dqx[t]
                for j in range(D):
                    acc = 0.0
                    for a in range(D):
                        acc += dqt[a] * Wq[a][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dqx accumulated to a non-finite value")
                    dqxt[j] = acc

            # dkvx[s][j]：k、v 两路径之和，按 a 自 0.0 升序累加。
            dkvx = [[0.0] * D for _ in range(Tk)]
            for s in range(Tk):
                dks, dvs = dk[s], dv[s]
                dkvxs = dkvx[s]
                for j in range(D):
                    acc = 0.0
                    for a in range(D):
                        acc += dks[a] * Wk[a][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dkvx accumulated to a non-finite value")
                        acc += dvs[a] * Wv[a][j]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dkvx accumulated to a non-finite value")
                    dkvxs[j] = acc
        except ValueError:
            self._cache = None
            raise

        return dqx, dkvx, dWq, dWk, dWv, dWo

    def forward_padded(self, xs, lengths, mask=None):
        """变长批投影多头自注意力前向，返回 (y, w) 并保存各样本独立快照。

        xs 须为非空 B×T×D 的 F 嵌套列表（B、T>0，各样本等长 T、每行等长
        D，padding 位同样须为 F）；lengths 须为 B 长 list，各项 type 为非
        bool 的 int 且 1<=值<=T；mask 须为 None 或 B×T×T 的嵌套 list，元素
        type 恰为 bool，padding 行、列仅校验后忽略；样本 b 令 L=lengths[b]，
        其有效 L×L 块每行至少一个 True（None 等价全 True），否则抛
        ValueError。任何失败都使既有缓存失效（其后的 backward_padded 必须
        重新 forward_padded）。

        样本按 b 升序、仅对有效前缀逐样本执行现有投影与多头注意力（即对
        x=xs[b][:L] 与有效 L×L mask 块调用现有 forward 语义），其缓存原样
        存为该样本的独立快照。y 为 B×T×D，w 为 B×heads×T×T：有效区写入
        各样本结果，padding 行（及 w 的 padding 列）均为 0.0。输出均为逐层
        新建的 float 列表，不修改或复用输入与投影属性；任一中间量非有限抛
        ValueError。实参数量错误沿用 Python 自带的 TypeError。
        """
        D = self.D
        # 任何失败的 forward_padded 都使既有缓存（含普通 forward 缓存）失效。
        self._cache = None
        try:
            # xs：非空 B×T×D 的 F 嵌套列表，B、T>0，矩形且每行等长 D。
            if type(xs) is not list or len(xs) == 0:
                raise ValueError("xs must be a non-empty list")
            B = len(xs)
            first = xs[0]
            if type(first) is not list or len(first) == 0:
                raise ValueError("xs must be a list of shape B×T×D with T>0")
            T = len(first)
            checked_xs = []
            for b in range(B):
                seq = xs[b]
                if type(seq) is not list or len(seq) != T:
                    raise ValueError("xs must be a list of shape %d×%d×%d"
                                     % (B, T, D))
                checked_seq = []
                for row in seq:
                    if type(row) is not list or len(row) != D:
                        raise ValueError("xs must be a list of shape %d×%d×%d"
                                         % (B, T, D))
                    for v in row:
                        if not _is_f(v):
                            raise ValueError(
                                "xs entries must be finite numbers, got %r"
                                % (v,))
                    checked_seq.append([float(v) for v in row])
                checked_xs.append(checked_seq)

            # lengths：B 长 list，各项为非 bool int 且 1<=值<=T。
            if type(lengths) is not list or len(lengths) != B:
                raise ValueError("lengths must be a list of length %d" % B)
            for L in lengths:
                if type(L) is bool or type(L) is not int or L < 1 or L > T:
                    raise ValueError(
                        "lengths entries must be non-bool integers in [1, %d], "
                        "got %r" % (T, L))

            # mask：None 或 B×T×T 的嵌套 list，元素 type 恰为 bool；padding
            # 行、列在此一并校验，随后忽略。
            if mask is not None:
                if type(mask) is not list or len(mask) != B:
                    raise ValueError(
                        "mask must be a list of shape %d×%d×%d" % (B, T, T))
                for b in range(B):
                    mb = mask[b]
                    if type(mb) is not list or len(mb) != T:
                        raise ValueError(
                            "mask must be a list of shape %d×%d×%d"
                            % (B, T, T))
                    for row in mb:
                        if type(row) is not list or len(row) != T:
                            raise ValueError(
                                "mask must be a list of shape %d×%d×%d"
                                % (B, T, T))
                        for m in row:
                            if type(m) is not bool:
                                raise ValueError(
                                    "mask entries must be exactly bool, got %r"
                                    % (m,))

            heads = self.heads
            # 输出先铺零：有效区随后覆写，padding 行（及 w 的 padding 列）
            # 保持全新零行；w 每样本含 heads 个 T×T 矩阵。
            y = [[[0.0] * D for _ in range(T)] for _ in range(B)]
            w = [[[[0.0] * T for _ in range(T)] for _ in range(heads)]
                 for _ in range(B)]
            snaps = [None] * B
            for b in range(B):
                L = lengths[b]
                xb = [checked_xs[b][t] for t in range(L)]
                if mask is None:
                    block = None
                else:
                    block = [list(mask[b][i][:L]) for i in range(L)]

                # 现有 forward 完成全部投影、多头注意力与输出投影，并自行
                # 校验 W 矩阵与有效块每行至少一个 True；成功后其缓存即该样本
                # 的独立快照。其内部失败已自行清空缓存。
                yb, wb = self.forward(xb, block)
                snaps[b] = self._cache

                yb_row = y[b]
                wb_pad = w[b]
                for t in range(L):
                    for a in range(D):
                        yv = float(yb[t][a])
                        if not math.isfinite(yv):
                            raise ValueError(
                                "padded attention output became non-finite")
                        yb_row[t][a] = yv
                    for h in range(heads):
                        wbht = wb[h][t]
                        wrow = wb_pad[h][t]
                        for s in range(L):
                            wv = float(wbht[s])
                            if not math.isfinite(wv):
                                raise ValueError(
                                    "padded attention weight became non-finite")
                            wrow[s] = wv
        except ValueError:
            self._cache = None
            raise

        self._cache = ("mha_padded", B, T, list(lengths), snaps)
        return y, w

    def backward_padded(self, dy):
        """变长批投影多头自注意力反向，返回
        (dx, dWq, dWk, dWv, dWo)。

        须紧随一次成功的 forward_padded（其后缓存未被任何其他 forward 类
        调用替换或被失败清空），否则抛 ValueError；dy 须为 B×T×D 的 F 嵌套
        列表（与前向 y 同形，padding 位经校验后忽略），否则抛 ValueError。
        实参数量错误沿用 Python 自带的 TypeError。

        逐样本以其独立快照调用现有 backward：dx 为 B×T×D（有效前缀写入各
        样本 dx，padding 行为 D 个 0.0）；dWq、dWk、dWv、dWo 四个 D×D 梯度
        均自 0.0 起按 b、行、列升序累加。所有结果均为逐层新建的 float 列表，
        不修改 dy、缓存或投影属性；成功时缓存保留，可重复调用且结果确定。
        任一中间量非有限或反向失败均抛 ValueError 并清空缓存。
        """
        D = self.D
        cache = self._cache
        try:
            if type(cache) is not tuple or len(cache) != 5 \
                    or cache[0] != "mha_padded":
                raise ValueError(
                    "backward_padded requires a successful forward_padded "
                    "pass before it")
            _, B, T, lengths_c, snaps = cache

            # dy：B×T×D 的 F 嵌套列表；padding 位在此一并校验，随后忽略。
            if type(dy) is not list or len(dy) != B:
                raise ValueError("dy must be a list of shape %d×%d×%d"
                                 % (B, T, D))
            checked_dy = []
            for b in range(B):
                seq = dy[b]
                if type(seq) is not list or len(seq) != T:
                    raise ValueError("dy must be a list of shape %d×%d×%d"
                                     % (B, T, D))
                checked_seq = []
                for row in seq:
                    if type(row) is not list or len(row) != D:
                        raise ValueError("dy must be a list of shape %d×%d×%d"
                                         % (B, T, D))
                    for v in row:
                        if not _is_f(v):
                            raise ValueError(
                                "dy entries must be finite numbers, got %r"
                                % (v,))
                    checked_seq.append([float(v) for v in row])
                checked_dy.append(checked_seq)

            dx = [[[0.0] * D for _ in range(T)] for _ in range(B)]
            dWq = [[0.0] * D for _ in range(D)]
            dWk = [[0.0] * D for _ in range(D)]
            dWv = [[0.0] * D for _ in range(D)]
            dWo = [[0.0] * D for _ in range(D)]

            for b in range(B):
                L = lengths_c[b]
                dyb = [checked_dy[b][t] for t in range(L)]

                # 以该样本的独立快照调用现有 backward；失败时其内部已清空
                # self._cache，由本方法外层 except 统一处理。
                self._cache = snaps[b]
                dxb, gWq, gWk, gWv, gWo = self.backward(dyb)

                dx_b = dx[b]
                for t in range(L):
                    dxt = dx_b[t]
                    gxt = dxb[t]
                    for a in range(D):
                        gv = float(gxt[a])
                        if not math.isfinite(gv):
                            raise ValueError(
                                "dx accumulated to a non-finite value")
                        dxt[a] = gv

                # 四个参数梯度自 0.0 起按 b、行、列升序累加。
                for a in range(D):
                    dWqa, dWka, dWva, dWoa = (
                        dWq[a], dWk[a], dWv[a], dWo[a])
                    gWqa, gWka, gWva, gWoa = (
                        gWq[a], gWk[a], gWv[a], gWo[a])
                    for j in range(D):
                        dWqa[j] += float(gWqa[j])
                        if not math.isfinite(dWqa[j]):
                            raise ValueError(
                                "dWq accumulated to a non-finite value")
                        dWka[j] += float(gWka[j])
                        if not math.isfinite(dWka[j]):
                            raise ValueError(
                                "dWk accumulated to a non-finite value")
                        dWva[j] += float(gWva[j])
                        if not math.isfinite(dWva[j]):
                            raise ValueError(
                                "dWv accumulated to a non-finite value")
                        dWoa[j] += float(gWoa[j])
                        if not math.isfinite(dWoa[j]):
                            raise ValueError(
                                "dWo accumulated to a non-finite value")

            # 恢复批缓存，保证可重复反向。
            self._cache = cache
        except ValueError:
            self._cache = None
            raise

        return dx, dWq, dWk, dWv, dWo

    def forward_cross_padded(self, qxs, kvxs, q_lengths, kv_lengths,
                             mask=None):
        """变长批投影多头交叉注意力前向，返回 (y, w) 并保存各样本独立快照。

        qxs、kvxs 须为非空 B×Tq×D、B×Tk×D 的 F 嵌套列表（B、Tq、Tk>0，两
        者 B 相同，各样本等长、每行等长 D，padding 位同样须为 F）；
        q_lengths、kv_lengths 须为 B 长 list，各项 type 为非 bool 的 int
        且分别在 [1, Tq]、[1, Tk]；mask 须为 None 或 B×Tq×Tk 的嵌套
        list，元素 type 恰为 bool，padding 行、列仅校验后忽略；样本 b 令
        Lq=q_lengths[b]、Lk=kv_lengths[b]，其有效 Lq×Lk 块每行至少一个
        True（None 等价全 True），否则抛 ValueError。任何失败都使既有缓存
        失效（其后的 backward_cross_padded 必须重新 forward_cross_padded）。

        样本按 b 升序、仅对有效前缀逐样本执行现有交叉注意力（即对
        qx=qxs[b][:Lq]、kvx=kvxs[b][:Lk] 与有效 Lq×Lk mask 块调用现有
        forward_cross 语义），其缓存原样存为该样本的独立快照。y 为
        B×Tq×D，w 为 B×heads×Tq×Tk：有效区写入各样本结果，padding 行
        （及 w 的 padding 列）均为 0.0。输出均为逐层新建的 float 列表，
        不修改或复用输入与投影属性；任一中间量非有限抛 ValueError。实参
        数量错误沿用 Python 自带的 TypeError。
        """
        D = self.D
        # 任何失败的 forward_cross_padded 都使既有缓存失效。
        self._cache = None
        try:
            # qxs：非空 B×Tq×D 的 F 嵌套列表，B、Tq>0，矩形且每行等长 D。
            if type(qxs) is not list or len(qxs) == 0:
                raise ValueError("qxs must be a non-empty list")
            B = len(qxs)
            first = qxs[0]
            if type(first) is not list or len(first) == 0:
                raise ValueError(
                    "qxs must be a list of shape B×Tq×D with Tq>0")
            Tq = len(first)
            checked_qxs = []
            for b in range(B):
                seq = qxs[b]
                if type(seq) is not list or len(seq) != Tq:
                    raise ValueError("qxs must be a list of shape %d×%d×%d"
                                     % (B, Tq, D))
                checked_seq = []
                for row in seq:
                    if type(row) is not list or len(row) != D:
                        raise ValueError(
                            "qxs must be a list of shape %d×%d×%d"
                            % (B, Tq, D))
                    for v in row:
                        if not _is_f(v):
                            raise ValueError(
                                "qxs entries must be finite numbers, got %r"
                                % (v,))
                    checked_seq.append([float(v) for v in row])
                checked_qxs.append(checked_seq)

            # kvxs：非空 B×Tk×D 的 F 嵌套列表，B 与 qxs 相同，Tk>0。
            if type(kvxs) is not list or len(kvxs) != B:
                raise ValueError("kvxs must be a list of shape %d×Tk×%d"
                                 % (B, D))
            first = kvxs[0]
            if type(first) is not list or len(first) == 0:
                raise ValueError(
                    "kvxs must be a list of shape B×Tk×D with Tk>0")
            Tk = len(first)
            checked_kvxs = []
            for b in range(B):
                seq = kvxs[b]
                if type(seq) is not list or len(seq) != Tk:
                    raise ValueError("kvxs must be a list of shape %d×%d×%d"
                                     % (B, Tk, D))
                checked_seq = []
                for row in seq:
                    if type(row) is not list or len(row) != D:
                        raise ValueError(
                            "kvxs must be a list of shape %d×%d×%d"
                            % (B, Tk, D))
                    for v in row:
                        if not _is_f(v):
                            raise ValueError(
                                "kvxs entries must be finite numbers, got %r"
                                % (v,))
                    checked_seq.append([float(v) for v in row])
                checked_kvxs.append(checked_seq)

            # q_lengths、kv_lengths：B 长 list，各项为非 bool int 且分别
            # 在 [1, Tq]、[1, Tk]。
            if type(q_lengths) is not list or len(q_lengths) != B:
                raise ValueError("q_lengths must be a list of length %d" % B)
            for L in q_lengths:
                if type(L) is bool or type(L) is not int or L < 1 or L > Tq:
                    raise ValueError(
                        "q_lengths entries must be non-bool integers in "
                        "[1, %d], got %r" % (Tq, L))
            if type(kv_lengths) is not list or len(kv_lengths) != B:
                raise ValueError("kv_lengths must be a list of length %d" % B)
            for L in kv_lengths:
                if type(L) is bool or type(L) is not int or L < 1 or L > Tk:
                    raise ValueError(
                        "kv_lengths entries must be non-bool integers in "
                        "[1, %d], got %r" % (Tk, L))

            # mask：None 或 B×Tq×Tk 的嵌套 list，元素 type 恰为 bool；
            # padding 行、列在此一并校验，随后忽略。
            if mask is not None:
                if type(mask) is not list or len(mask) != B:
                    raise ValueError(
                        "mask must be a list of shape %d×%d×%d"
                        % (B, Tq, Tk))
                for b in range(B):
                    mb = mask[b]
                    if type(mb) is not list or len(mb) != Tq:
                        raise ValueError(
                            "mask must be a list of shape %d×%d×%d"
                            % (B, Tq, Tk))
                    for row in mb:
                        if type(row) is not list or len(row) != Tk:
                            raise ValueError(
                                "mask must be a list of shape %d×%d×%d"
                                % (B, Tq, Tk))
                        for m in row:
                            if type(m) is not bool:
                                raise ValueError(
                                    "mask entries must be exactly bool, got %r"
                                    % (m,))

            heads = self.heads
            # 输出先铺零：有效区随后覆写，padding 行（及 w 的 padding 列）
            # 保持全新零行；w 每样本含 heads 个 Tq×Tk 矩阵。
            y = [[[0.0] * D for _ in range(Tq)] for _ in range(B)]
            w = [[[[0.0] * Tk for _ in range(Tq)] for _ in range(heads)]
                 for _ in range(B)]
            snaps = [None] * B
            for b in range(B):
                Lq = q_lengths[b]
                Lk = kv_lengths[b]
                qxb = [checked_qxs[b][t] for t in range(Lq)]
                kvxb = [checked_kvxs[b][s] for s in range(Lk)]
                if mask is None:
                    block = None
                else:
                    block = [list(mask[b][i][:Lk]) for i in range(Lq)]

                # 现有 forward_cross 完成全部投影、多头注意力与输出投影，并
                # 自行校验 W 矩阵与有效块每行至少一个 True；成功后其缓存即
                # 该样本的独立快照。其内部失败已自行清空缓存。
                yb, wb = self.forward_cross(qxb, kvxb, block)
                snaps[b] = self._cache

                yb_row = y[b]
                wb_pad = w[b]
                for t in range(Lq):
                    for a in range(D):
                        yv = float(yb[t][a])
                        if not math.isfinite(yv):
                            raise ValueError(
                                "padded cross attention output became "
                                "non-finite")
                        yb_row[t][a] = yv
                    for h in range(heads):
                        wbht = wb[h][t]
                        wrow = wb_pad[h][t]
                        for s in range(Lk):
                            wv = float(wbht[s])
                            if not math.isfinite(wv):
                                raise ValueError(
                                    "padded cross attention weight became "
                                    "non-finite")
                            wrow[s] = wv
        except ValueError:
            self._cache = None
            raise

        self._cache = ("mha_cross_padded", B, Tq, Tk, list(q_lengths),
                       list(kv_lengths), snaps)
        return y, w

    def backward_cross_padded(self, dy):
        """变长批投影多头交叉注意力反向，返回
        (dqxs, dkvxs, dWq, dWk, dWv, dWo)。

        须紧随一次成功的 forward_cross_padded（其后缓存未被任何其他
        forward 类调用替换或被失败清空），否则抛 ValueError；dy 须为
        B×Tq×D 的 F 嵌套列表（与前向 y 同形，padding 位经校验后忽略），
        否则抛 ValueError。实参数量错误沿用 Python 自带的 TypeError。

        逐样本以其独立快照调用现有 backward_cross：dqxs 为 B×Tq×D、dkvxs
        为 B×Tk×D（有效前缀写入各样本结果，padding 行为 D 个 0.0）；
        dWq、dWk、dWv、dWo 四个 D×D 梯度均自 0.0 起按 b、行、列升序累加。
        所有结果均为逐层新建的 float 列表，不修改 dy、缓存或投影属性；
        成功时缓存保留，可重复调用且结果确定。任一中间量非有限或反向失败
        均抛 ValueError 并清空缓存。
        """
        D = self.D
        cache = self._cache
        try:
            if type(cache) is not tuple or len(cache) != 7 \
                    or cache[0] != "mha_cross_padded":
                raise ValueError(
                    "backward_cross_padded requires a successful "
                    "forward_cross_padded pass before it")
            _, B, Tq, Tk, q_lengths_c, kv_lengths_c, snaps = cache

            # dy：B×Tq×D 的 F 嵌套列表；padding 位在此一并校验，随后忽略。
            if type(dy) is not list or len(dy) != B:
                raise ValueError("dy must be a list of shape %d×%d×%d"
                                 % (B, Tq, D))
            checked_dy = []
            for b in range(B):
                seq = dy[b]
                if type(seq) is not list or len(seq) != Tq:
                    raise ValueError("dy must be a list of shape %d×%d×%d"
                                     % (B, Tq, D))
                checked_seq = []
                for row in seq:
                    if type(row) is not list or len(row) != D:
                        raise ValueError("dy must be a list of shape %d×%d×%d"
                                         % (B, Tq, D))
                    for v in row:
                        if not _is_f(v):
                            raise ValueError(
                                "dy entries must be finite numbers, got %r"
                                % (v,))
                    checked_seq.append([float(v) for v in row])
                checked_dy.append(checked_seq)

            dqxs = [[[0.0] * D for _ in range(Tq)] for _ in range(B)]
            dkvxs = [[[0.0] * D for _ in range(Tk)] for _ in range(B)]
            dWq = [[0.0] * D for _ in range(D)]
            dWk = [[0.0] * D for _ in range(D)]
            dWv = [[0.0] * D for _ in range(D)]
            dWo = [[0.0] * D for _ in range(D)]

            for b in range(B):
                Lq = q_lengths_c[b]
                Lk = kv_lengths_c[b]
                dyb = [checked_dy[b][t] for t in range(Lq)]

                # 以该样本的独立快照调用现有 backward_cross；失败时其内部已
                # 清空 self._cache，由本方法外层 except 统一处理。
                self._cache = snaps[b]
                dqxb, dkvxb, gWq, gWk, gWv, gWo = self.backward_cross(dyb)

                dqxs_b = dqxs[b]
                for t in range(Lq):
                    dqxt = dqxs_b[t]
                    gxt = dqxb[t]
                    for a in range(D):
                        gv = float(gxt[a])
                        if not math.isfinite(gv):
                            raise ValueError(
                                "dqxs accumulated to a non-finite value")
                        dqxt[a] = gv

                dkvxs_b = dkvxs[b]
                for s in range(Lk):
                    dkvs = dkvxs_b[s]
                    gxs = dkvxb[s]
                    for a in range(D):
                        gv = float(gxs[a])
                        if not math.isfinite(gv):
                            raise ValueError(
                                "dkvxs accumulated to a non-finite value")
                        dkvs[a] = gv

                # 四个参数梯度自 0.0 起按 b、行、列升序累加。
                for a in range(D):
                    dWqa, dWka, dWva, dWoa = (
                        dWq[a], dWk[a], dWv[a], dWo[a])
                    gWqa, gWka, gWva, gWoa = (
                        gWq[a], gWk[a], gWv[a], gWo[a])
                    for j in range(D):
                        dWqa[j] += float(gWqa[j])
                        if not math.isfinite(dWqa[j]):
                            raise ValueError(
                                "dWq accumulated to a non-finite value")
                        dWka[j] += float(gWka[j])
                        if not math.isfinite(dWka[j]):
                            raise ValueError(
                                "dWk accumulated to a non-finite value")
                        dWva[j] += float(gWva[j])
                        if not math.isfinite(dWva[j]):
                            raise ValueError(
                                "dWv accumulated to a non-finite value")
                        dWoa[j] += float(gWoa[j])
                        if not math.isfinite(dWoa[j]):
                            raise ValueError(
                                "dWo accumulated to a non-finite value")

            # 恢复批缓存，保证可重复反向。
            self._cache = cache
        except ValueError:
            self._cache = None
            raise

        return dqxs, dkvxs, dWq, dWk, dWv, dWo


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


class GRUCell(object):
    """单步 GRU 单元，参数 W/b 由 seed 确定性初始化。

    W 形状 3H×(I+H)，b 形状 3H。以 R = random.Random(seed)、
    s = 1/sqrt(I+H)，按 W 行列序令每项为 float((2*R.random()-1)*s)，
    b 初始化为全 0.0。
    """

    def __init__(self, I, H, seed=0):
        if type(I) is not int or type(H) is not int or I <= 0 or H <= 0:
            raise ValueError("I and H must be positive integers")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.I = I
        self.H = H
        R = random.Random(seed)
        s = 1.0 / math.sqrt(I + H)
        self.W = [[float((2 * R.random() - 1) * s)
                   for _ in range(I + H)] for _ in range(3 * H)]
        self.b = [0.0] * (3 * H)

    def forward(self, x, h_prev):
        """单步前向传播，返回 (h, cache)。

        x、h_prev 须分别为长度 I、H 的 F 列表；self.W、self.b 须仍满足
        3H×(I+H)、3H 的形状且元素均为 F，否则抛 ValueError。记 A_k(v) 为
        从 float(b[k]) 起按 j 升序累加 float(W[k][j])*float(v[j])，则
            z = x + h_prev
            r_i = sigmoid(A_i(z))，u_i = sigmoid(A_{H+i}(z))
            q = x + (r ⊙ h_prev)
            n_i = tanh(A_{2H+i}(q))
            h_i = (1 - u_i)*n_i + u_i*h_prev_i
        中间量出现非有限值同样抛 ValueError。返回的 h 为新的 H 长 float
        列表；cache 为 dict，键按 x、h_prev、z、r、u、q、n、h、W 顺序插入，
        向量值均为新的 float 列表，W 为二维 float 深拷贝；返回对象不复用或
        修改输入及属性。
        """
        I, H = self.I, self.H
        x = _check_vector(x, I, "x")
        h_prev = _check_vector(h_prev, H, "h_prev")
        W = _check_matrix(self.W, 3 * H, I + H, "W")
        b = _check_vector(self.b, 3 * H, "b")

        M = I + H
        z = x + h_prev

        # 重置门 r 与更新门 u：A_k(z)，k=0..2H-1，按列序累加。
        r = [0.0] * H
        u = [0.0] * H
        for k in range(2 * H):
            acc = float(b[k])
            row = W[k]
            for j in range(M):
                acc += float(row[j]) * float(z[j])
            if not math.isfinite(acc):
                raise ValueError("gate pre-activation accumulated to a "
                                 "non-finite value")
            if k < H:
                r[k] = LSTMCell._sigmoid(acc)
            else:
                u[k - H] = LSTMCell._sigmoid(acc)

        # q = x + (r ⊙ h_prev)（列表拼接，长度 I+H）。
        q = [float(v) for v in x]
        for i in range(H):
            qi = r[i] * float(h_prev[i])
            if not math.isfinite(qi):
                raise ValueError("candidate input became non-finite")
            q.append(qi)

        # n_i = tanh(A_{2H+i}(q))；h_i = (1-u_i)n_i + u_i*h_prev_i。
        n = [0.0] * H
        h = [0.0] * H
        for i in range(H):
            k = 2 * H + i
            acc = float(b[k])
            row = W[k]
            for j in range(M):
                acc += float(row[j]) * float(q[j])
            if not math.isfinite(acc):
                raise ValueError("candidate pre-activation accumulated to a "
                                 "non-finite value")
            ni = math.tanh(acc)
            n[i] = ni
            hi = (1.0 - u[i]) * ni + u[i] * float(h_prev[i])
            if not math.isfinite(hi):
                raise ValueError("hidden state became non-finite")
            h[i] = hi

        cache = {
            "x": [float(v) for v in x],
            "h_prev": [float(v) for v in h_prev],
            "z": [float(v) for v in z],
            "r": [float(v) for v in r],
            "u": [float(v) for v in u],
            "q": [float(v) for v in q],
            "n": [float(v) for v in n],
            "h": [float(v) for v in h],
            "W": [[float(v) for v in row] for row in W],
        }
        return [float(v) for v in h], cache

    def backward(self, dh, cache):
        """单步反向传播，固定返回 (dx, dh_prev, dW, db)。

        dh 须为长度 H 的 F 列表；cache 须为 dict 且严格符合 forward 的
        公开缓存契约：恰含 x、h_prev、z、r、u、q、n、h、W 共 9 个有序键
        （多、少、乱序均非法），其中 x 长 I，z、q 长 I+H，h_prev、r、u、
        n、h 长 H，W 为 3H×(I+H)，元素均为 F。任一容器、键序、形状或
        元素非法抛 ValueError；实参数量错误沿用 Python 自带的 TypeError。

        所有运算逐元素（变量取 cache 同名值）：
            dn   = dh*(1 - u)，du = dh*(h_prev - n)
            da_n = dn*(1 - n^2)
            dq[j] 对每个 j 以 0.0 起按 i 升序累加 W[2H+i][j]*da_n[i]
            dr   = dq[I:]*h_prev
            da_r = dr*r*(1-r)，da_u = du*u*(1-u)
            da 依次拼接 da_r、da_u、da_n 各 H 项
            dz[j] 对每个 j 以 0.0 起按 k=0..2H-1 升序累加 W[k][j]*da[k]
            dx[j] = dq[j] + dz[j]（j=0..I-1）
            dh_prev[i] 从 0.0 起依次加 dh[i]*u[i]、dq[I+i]*r[i]、dz[I+i]
            dW 前 2H 行取 da[k]*z[j]，末 H 行取 da[k]*q[j]，db[k] = da[k]
        四个返回值形状依次为 I、H、3H×(I+H)、3H，均为全新 float 列表
        （dW 逐层新建），不修改或复用 dh、cache 及其内容。任一中间结果或
        输出非有限同样抛 ValueError。
        """
        I, H = self.I, self.H
        M = I + H
        dh = _check_vector(dh, H, "dh")

        if type(cache) is not dict:
            raise ValueError("cache must be a dict returned by forward")
        expected_keys = ["x", "h_prev", "z", "r", "u", "q", "n", "h", "W"]
        if list(cache.keys()) != expected_keys:
            raise ValueError(
                "cache must contain exactly the 9 forward keys in order: %r"
                % expected_keys)
        _check_vector(cache["x"], I, "cache['x']")
        h_prev = _check_vector(cache["h_prev"], H, "cache['h_prev']")
        z = _check_vector(cache["z"], M, "cache['z']")
        r = _check_vector(cache["r"], H, "cache['r']")
        u = _check_vector(cache["u"], H, "cache['u']")
        q = _check_vector(cache["q"], M, "cache['q']")
        n = _check_vector(cache["n"], H, "cache['n']")
        _check_vector(cache["h"], H, "cache['h']")
        W = _check_matrix(cache["W"], 3 * H, M, "cache['W']")

        # h = (1-u)*n + u*h_prev：dn、du；n = tanh(A_n(q))：da_n。
        dn = [0.0] * H
        du = [0.0] * H
        da = [0.0] * (3 * H)
        for i in range(H):
            ui = float(u[i])
            ni = float(n[i])

            dni = float(dh[i]) * (1.0 - ui)
            if not math.isfinite(dni):
                raise ValueError("dn became non-finite")
            dn[i] = dni

            dui = float(dh[i]) * (float(h_prev[i]) - ni)
            if not math.isfinite(dui):
                raise ValueError("du became non-finite")
            du[i] = dui

            dani = dni * (1.0 - ni * ni)
            if not math.isfinite(dani):
                raise ValueError("candidate gradient became non-finite")
            da[2 * H + i] = dani

        # dq = W_nᵀ da_n：每个 j 独立以 0.0 起按 i 升序累加。
        dq = [0.0] * M
        for j in range(M):
            acc = 0.0
            for i in range(H):
                acc += W[2 * H + i][j] * da[2 * H + i]
                if not math.isfinite(acc):
                    raise ValueError("dq accumulated to a non-finite value")
            dq[j] = acc

        # q[I+i] = r_i*h_prev_i：dr = dq[I:]*h_prev；两门 sigmoid 导数。
        for i in range(H):
            dri = dq[I + i] * float(h_prev[i])
            if not math.isfinite(dri):
                raise ValueError("dr became non-finite")

            ri = float(r[i])
            dari = dri * ri * (1.0 - ri)
            if not math.isfinite(dari):
                raise ValueError("reset gate gradient became non-finite")
            da[i] = dari

            ui = float(u[i])
            daui = du[i] * ui * (1.0 - ui)
            if not math.isfinite(daui):
                raise ValueError("update gate gradient became non-finite")
            da[H + i] = daui

        # dz = [W_r; W_u]ᵀ da：每个 j 独立以 0.0 起按 k=0..2H-1 累加。
        dz = [0.0] * M
        for j in range(M):
            acc = 0.0
            for k in range(2 * H):
                acc += W[k][j] * da[k]
                if not math.isfinite(acc):
                    raise ValueError("dz accumulated to a non-finite value")
            dz[j] = acc

        # x 同时进入 z[:I] 与 q[:I]：dx = dq[:I] + dz[:I]。
        dx = [0.0] * I
        for j in range(I):
            v = dq[j] + dz[j]
            if not math.isfinite(v):
                raise ValueError("dx became non-finite")
            dx[j] = v

        # h_prev 的三项贡献：直接路径 u、经 q[I:] 的 r、经 z[I:]。
        dh_prev = [0.0] * H
        for i in range(H):
            acc = 0.0
            acc += float(dh[i]) * float(u[i])
            if not math.isfinite(acc):
                raise ValueError("dh_prev accumulated to a non-finite value")
            acc += dq[I + i] * float(r[i])
            if not math.isfinite(acc):
                raise ValueError("dh_prev accumulated to a non-finite value")
            acc += dz[I + i]
            if not math.isfinite(acc):
                raise ValueError("dh_prev accumulated to a non-finite value")
            dh_prev[i] = acc

        # dW：门段（r、u）对 z 求导，候选段（n）对 q 求导；db = da。
        dW = [[0.0] * M for _ in range(3 * H)]
        for k in range(3 * H):
            row = dW[k]
            dak = da[k]
            src = z if k < 2 * H else q
            for j in range(M):
                v = dak * src[j]
                if not math.isfinite(v):
                    raise ValueError("dW became non-finite")
                row[j] = v
        db = [float(v) for v in da]

        return dx, dh_prev, dW, db

    def _check_forward_cache(self, cache):
        """校验单个 cache 严格符合 forward 的公开缓存契约。

        须为 dict，恰含 x、h_prev、z、r、u、q、n、h、W 共 9 个有序键
        （多、少、乱序均非法），其中 x 长 I，h_prev、r、u、n、h 长 H，
        z、q 长 I+H，W 为 3H×(I+H)，元素均为 F。任一容器、键序、形状或
        元素非法抛 ValueError。
        """
        I, H = self.I, self.H
        if type(cache) is not dict:
            raise ValueError("cache must be a dict returned by forward")
        expected_keys = ["x", "h_prev", "z", "r", "u", "q", "n", "h", "W"]
        if list(cache.keys()) != expected_keys:
            raise ValueError(
                "cache must contain exactly the 9 forward keys in order: %r"
                % expected_keys)
        _check_vector(cache["x"], I, "cache['x']")
        _check_vector(cache["h_prev"], H, "cache['h_prev']")
        _check_vector(cache["z"], I + H, "cache['z']")
        _check_vector(cache["r"], H, "cache['r']")
        _check_vector(cache["u"], H, "cache['u']")
        _check_vector(cache["q"], I + H, "cache['q']")
        _check_vector(cache["n"], H, "cache['n']")
        _check_vector(cache["h"], H, "cache['h']")
        _check_matrix(cache["W"], 3 * H, I + H, "cache['W']")

    def backward_sequence(self, dhs, caches, dh_last=None, tbptt_steps=None):
        """沿整条序列反向传播（可选截断 BPTT），固定返回
        (dxs, dh0, dW, db)。

        caches 须为非空的时序 list，长度确定 T，每项均为 dict 且严格符合
        forward 的公开缓存契约（同 backward 对单个 cache 的要求）；dhs 须为
        与 caches 同长的 T×H 的 F 列表；dh_last 为 None（等价全零末端梯度）
        或长度 H 的 F 列表；tbptt_steps 为 None 或非 bool 的正整数 K。任一
        校验失败抛 ValueError；实参数量错误沿用 Python 自带的 TypeError。

        置 ph = dh_last 或零，自 t=T-1 至 0 依次调用
        现有 backward(dhs[t]+ph, caches[t])：dx 放回 dxs 的 t 位，返回的
        dh_prev 成为下一步的 ph；dW、db 各元素自 0.0 起按 t 降序、行列
        升序累加。给定 K 时窗口从末端对齐：每处理完 K 步且尚有更早的步，
        便将 ph 清零，故跨窗状态梯度为零。

        四个返回值形状依次为 T×I、H、3H×(I+H)、3H；dh0 即处理完 t=0
        一步后所得的状态梯度。所有结果均为全新 float 列表（矩阵逐层
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

        # tbptt_steps：None 或非 bool 正整数。
        if tbptt_steps is not None:
            if type(tbptt_steps) is bool or type(tbptt_steps) is not int \
                    or tbptt_steps <= 0:
                raise ValueError(
                    "tbptt_steps must be None or a positive integer, got %r"
                    % (tbptt_steps,))

        dxs = [None] * T
        dW = [[0.0] * (I + H) for _ in range(3 * H)]
        db = [0.0] * (3 * H)

        steps_done = 0
        for t in range(T - 1, -1, -1):
            # dhs[t] + ph：交由 backward 再做 F 校验（溢出即 ValueError）。
            dh_in = [dhs[t][j] + ph[j] for j in range(H)]
            dx, dh_prev, step_dW, step_db = self.backward(dh_in, caches[t])
            dxs[t] = [float(v) for v in dx]

            # 各参数梯度元素自 0.0 起按 t 降序、行列升序累加，累加后须有限。
            for k in range(3 * H):
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
            steps_done += 1

            # 窗口末端对齐：每满 K 步且尚有更早步，截断跨窗状态梯度。
            if (tbptt_steps is not None
                    and steps_done % tbptt_steps == 0 and t > 0):
                ph = [0.0] * H

        dh0 = [float(v) for v in ph]
        return dxs, dh0, dW, db


class BidirectionalGRU(object):
    """双向 GRU：两个独立 GRUCell 分别正序、倒序扫描同一序列。

    forward_cell = GRUCell(I, H, seed)，backward_cell = GRUCell(I, H, seed+1)。
    前向在时刻 t 拼接前支隐状态 h_f[t] 与映射至 t 的后支隐状态 h_b[t]
    （后支自序列末端倒序推进，其初态位于 t=T-1）。反向时两支各自沿自身
    扫描方向经 GRUCell.backward_sequence 求梯度，后支输入梯度再倒回时间
    正序，两支输入梯度逐元素相加。
    """

    def __init__(self, I, H, seed=0):
        # bool 是 int 的子类型，按契约须与非正整数一并拒绝。
        if type(I) is bool or type(I) is not int or I <= 0 \
                or type(H) is bool or type(H) is not int or H <= 0:
            raise ValueError("I and H must be positive integers")
        if type(seed) is bool or type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.I = I
        self.H = H
        self.forward_cell = GRUCell(I, H, seed)
        self.backward_cell = GRUCell(I, H, seed + 1)
        self._cache = None

    def forward(self, xs):
        """双向前向传播，返回 T×2H 的 F 列表并缓存。

        xs 须为非空 T×I 的 F 列表（T>0，首行长度确定 I，其余行等长）；
        两支 GRUCell 的参数亦须仍满足各自形状与 F 契约，否则抛 ValueError
        且缓存失效（其后的 backward 必须重新 forward）。两支均以全零 H
        向量为初态：前支按 t=0..T-1 正序调用
        forward_cell.forward(xs[t], h)，后支按 t=T-1..0 倒序调用
        backward_cell.forward(xs[t], h)。返回第 t 行为前支 h_f[t] 拼接
        映射至 t 的后支 h_b[t]（长度 2H）；输出为全新 float 列表，不修改
        或复用输入与单元参数。
        """
        I, H = self.I, self.H
        # 任何失败的 forward 都使既有缓存失效；成功时再以新缓存覆盖。
        self._cache = None

        if type(xs) is not list or len(xs) == 0:
            raise ValueError("xs must be a non-empty list")
        T = len(xs)
        xs = _check_matrix(xs, T, I, "xs")

        zero = [0.0] * H
        hf = [None] * T
        cf = [None] * T
        hb = [None] * T
        cb = [None] * T

        h = zero
        for t in range(T):
            hf[t], cf[t] = self.forward_cell.forward(xs[t], h)
            h = hf[t]

        h = zero
        for t in range(T - 1, -1, -1):
            hb[t], cb[t] = self.backward_cell.forward(xs[t], h)
            h = hb[t]

        outputs = [[float(v) for v in hf[t]] + [float(v) for v in hb[t]]
                   for t in range(T)]
        self._cache = (T, cf, cb)
        return outputs

    def backward(self, dys):
        """双向反向传播，固定返回
        (dxs, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b)。

        此前须有一次成功的 forward（其后无失败 forward 使缓存失效），否则
        抛 ValueError；dys 须为与前向同形的 T×2H 的 F 列表，否则抛
        ValueError。每行前 H 项为前支隐状态梯度、后 H 项为映射至同一时刻
        的后支隐状态梯度。前支以原序梯度、原序缓存调用
        forward_cell.backward_sequence；后支沿其扫描方向（时间倒序组织
        梯度与缓存）调用 backward_cell.backward_sequence，所得末端状态
        梯度即后支初态（位于 t=T-1）梯度 dh0_b，输入梯度再倒回时间正序。
        dxs 为 T×I，各元素自 0.0 起依次加前支、后支两支的输入梯度；
        dh0_f、dh0_b 长 H，dW_f、dW_b 形状 3H×(I+H)，db_f、db_b 长 3H。
        所有结果均为全新 float 列表（矩阵逐层深拷贝），不修改 dys、缓存
        或单元参数，重复调用结果确定；任一中间量非有限抛 ValueError，
        实参数量错误沿用 Python 自带的 TypeError。
        """
        I, H = self.I, self.H
        cache = self._cache
        if cache is None:
            raise ValueError(
                "backward requires a successful forward pass before it")
        T, cf, cb = cache
        dys = _check_matrix(dys, T, 2 * H, "dys")

        # 拆分两支梯度：逐行新建 float 列表，绝不复用 dys 的行。
        dhf = [[float(v) for v in row[:H]] for row in dys]
        dhb = [[float(v) for v in row[H:]] for row in dys]

        dxs_f, dh0_f, dW_f, db_f = \
            self.forward_cell.backward_sequence(dhf, cf)

        # 后支的“时序”沿扫描方向（t 降序）：梯度与缓存均按该方向排列。
        dxs_b_rev, dh0_b, dW_b, db_b = \
            self.backward_cell.backward_sequence(dhb[::-1], cb[::-1])
        # 输入梯度按扫描倒序给出，倒回时间正序以便与前支逐时刻相加。
        dxs_b = dxs_b_rev[::-1]

        dxs = [[0.0] * I for _ in range(T)]
        for t in range(T):
            row = dxs[t]
            row_f = dxs_f[t]
            row_b = dxs_b[t]
            for j in range(I):
                acc = 0.0
                acc += float(row_f[j])
                if not math.isfinite(acc):
                    raise ValueError(
                        "dxs accumulated to a non-finite value")
                acc += float(row_b[j])
                if not math.isfinite(acc):
                    raise ValueError(
                        "dxs accumulated to a non-finite value")
                row[j] = acc

        return dxs, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b

    def forward_padded(self, xs, lengths):
        """变长批双向前向，返回 B×T×2H 的全新 float 列表并保存批缓存。

        xs 须为非空 B×T×I 的 F 嵌套列表（B、T>0，各样本等长 T、每行等长
        I，padding 位同样须为 F）；lengths 须为 B 长 list，各项 type 为非
        bool 的 int 且 1<=值<=T，否则抛 ValueError。任何失败都使既有缓存
        失效（其后的 backward_padded 必须重新 forward_padded）。样本 b 仅
        对前 lengths[b] 行执行双向前向：两支均以全零 H 向量为初态，前支按
        t=0..L_b-1 正序调用 forward_cell.forward，后支按 t=L_b-1..0 倒序
        调用 backward_cell.forward。返回第 b 样本第 t 行为前支 h_f[t] 拼接
        映射至 t 的后支 h_b[t]（长度 2H）；有效前缀（t<L_b）为真实输出，
        其余行是 2H 个 0.0。输出为逐层新建的 float 列表，不修改或复用输入
        与单元参数；任一中间量非有限同样抛 ValueError。
        """
        I, H = self.I, self.H
        # 任何失败的 forward_padded 都使既有缓存（含普通 forward 缓存）失效。
        self._cache = None

        # xs：非空 B×T×I 的 F 嵌套列表，B、T>0，矩形且每行等长 I。
        if type(xs) is not list or len(xs) == 0:
            raise ValueError("xs must be a non-empty list")
        B = len(xs)
        first = xs[0]
        if type(first) is not list or len(first) == 0:
            raise ValueError("xs must be a list of shape B×T×I with T>0")
        T = len(first)
        checked_xs = []
        for b in range(B):
            seq = xs[b]
            if type(seq) is not list or len(seq) != T:
                raise ValueError("xs must be a list of shape %d×%d×%d"
                                 % (B, T, I))
            checked_seq = []
            for row in seq:
                if type(row) is not list or len(row) != I:
                    raise ValueError("xs must be a list of shape %d×%d×%d"
                                     % (B, T, I))
                for v in row:
                    if not _is_f(v):
                        raise ValueError(
                            "xs entries must be finite numbers, got %r" % (v,))
                checked_seq.append([float(v) for v in row])
            checked_xs.append(checked_seq)
        xs = checked_xs

        # lengths：B 长 list，各项为非 bool int 且 1<=值<=T。
        if type(lengths) is not list or len(lengths) != B:
            raise ValueError("lengths must be a list of length %d" % B)
        for L in lengths:
            if type(L) is bool or type(L) is not int or L < 1 or L > T:
                raise ValueError(
                    "lengths entries must be non-bool integers in [1, %d], "
                    "got %r" % (T, L))

        # 输出先铺零：有效前缀随后覆写，padding 行保持全新的 2H 零行。
        outputs = [[[0.0] * (2 * H) for _ in range(T)] for _ in range(B)]
        cfs = [None] * B
        cbs = [None] * B
        zero = [0.0] * H
        for b in range(B):
            L = lengths[b]
            cf = [None] * L
            hf = [None] * L
            h = zero
            for t in range(L):
                hf[t], cf[t] = self.forward_cell.forward(xs[b][t], h)
                h = hf[t]

            cb = [None] * L
            hb = [None] * L
            h = zero
            for t in range(L - 1, -1, -1):
                hb[t], cb[t] = self.backward_cell.forward(xs[b][t], h)
                h = hb[t]

            cfs[b] = cf
            cbs[b] = cb
            out_b = outputs[b]
            for t in range(L):
                out_b[t] = [float(v) for v in hf[t]] + \
                           [float(v) for v in hb[t]]

        self._cache = ("padded", B, T, list(lengths), cfs, cbs)
        return outputs

    def backward_padded(self, dys):
        """变长批双向反向，固定返回
        (dxs, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b)。

        最近一次前向须为成功的 forward_padded（其后无失败的 forward 类调用
        使缓存失效），否则抛 ValueError；dys 须为 B×T×2H 的 F 嵌套列表
        （与前向同形，padding 位经校验后忽略），否则抛 ValueError。各样本
        有效前缀按现有 backward 语义反传：前支以原序梯度、原序缓存调用
        forward_cell.backward_sequence，后支沿其扫描方向（时间倒序组织梯度
        与缓存）调用 backward_cell.backward_sequence，所得末端状态梯度即
        该样本后支初态（位于 t=L_b-1）梯度，输入梯度再倒回时间正序。

        dxs 为 B×T×I（有效前缀为前、后两支输入梯度逐元素相加，padding 行
        为 I 个 0.0）；dh0_f、dh0_b 均为 B×H，第 b 行为样本 b 的两支初态
        梯度；dW_f、dW_b 形状 3H×(I+H)，db_f、db_b 长 3H，各元素均自
        0.0 起按 b 升序逐元素累加。所有结果均为逐层新建的 float 列表，不
        修改 dys、缓存或单元参数，重复调用结果确定；任一中间量非有限抛
        ValueError，实参数量错误沿用 Python 自带的 TypeError。
        """
        I, H = self.I, self.H
        cache = self._cache
        if type(cache) is not tuple or len(cache) != 6 \
                or cache[0] != "padded":
            raise ValueError(
                "backward_padded requires a successful forward_padded pass "
                "before it")
        _, B, T, lengths, cfs, cbs = cache

        # dys：B×T×2H 的 F 嵌套列表；padding 位在此一并校验，随后忽略。
        dys = self._check_padded_3d(dys, B, T, 2 * H, "dys")

        # 各样本有效前缀的输出梯度直接取自 dys（padding 行忽略）。
        return self._padded_backprop(
            B, T, lengths, cfs, cbs,
            [[dys[b][t] for t in range(lengths[b])] for b in range(B)])

    @staticmethod
    def _check_padded_3d(values, B, T, cols, name):
        """校验 values 为 B×T×cols 的 F 嵌套列表，返回逐层 float 拷贝。"""
        if type(values) is not list or len(values) != B:
            raise ValueError("%s must be a list of shape %d×%d×%d"
                             % (name, B, T, cols))
        checked = []
        for b in range(B):
            seq = values[b]
            if type(seq) is not list or len(seq) != T:
                raise ValueError("%s must be a list of shape %d×%d×%d"
                                 % (name, B, T, cols))
            checked_seq = []
            for row in seq:
                if type(row) is not list or len(row) != cols:
                    raise ValueError("%s must be a list of shape %d×%d×%d"
                                     % (name, B, T, cols))
                for v in row:
                    if not _is_f(v):
                        raise ValueError(
                            "%s entries must be finite numbers, got %r"
                            % (name, v))
                checked_seq.append([float(v) for v in row])
            checked.append(checked_seq)
        return checked

    @staticmethod
    def _check_padded_2d(values, B, cols, name):
        """校验 values 为 B×cols 的 F 列表，返回逐层 float 拷贝。"""
        if type(values) is not list or len(values) != B:
            raise ValueError("%s must be a list of shape %d×%d"
                             % (name, B, cols))
        checked = []
        for row in values:
            if type(row) is not list or len(row) != cols:
                raise ValueError("%s must be a list of shape %d×%d"
                                 % (name, B, cols))
            for v in row:
                if not _is_f(v):
                    raise ValueError(
                        "%s entries must be finite numbers, got %r"
                        % (name, v))
            checked.append([float(v) for v in row])
        return checked

    def _padded_backprop(self, B, T, lengths, cfs, cbs, dy_valid):
        """forward_padded 与 forward_attn_padded 共用的批 GRU 反传。

        dy_valid[b] 为样本 b 有效前缀（长 L_b、行长 2H）的输出梯度，按
        backward_padded 的既有语义沿两支反传。固定返回
        (dxs, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b)：dxs 为 B×T×I
        （padding 行为 I 个 0.0），dh0_f、dh0_b 为 B×H，dW_f、dW_b 形状
        3H×(I+H)，db_f、db_b 长 3H，参数梯度自 0.0 起按 b 升序逐元素累加。
        所有结果均为逐层新建的 float 列表，不修改 dy_valid、缓存或单元
        参数；任一中间量非有限抛 ValueError。
        """
        I, H = self.I, self.H
        dxs = [[[0.0] * I for _ in range(T)] for _ in range(B)]
        dh0_f = [None] * B
        dh0_b = [None] * B
        dW_f = [[0.0] * (I + H) for _ in range(3 * H)]
        db_f = [0.0] * (3 * H)
        dW_b = [[0.0] * (I + H) for _ in range(3 * H)]
        db_b = [0.0] * (3 * H)

        # 参数梯度自 0.0 起按 b 升序、行列升序逐元素累加。
        for b in range(B):
            L = lengths[b]
            dy_b = dy_valid[b]
            dhf = [[float(v) for v in dy_b[t][:H]] for t in range(L)]
            dhb = [[float(v) for v in dy_b[t][H:]] for t in range(L)]

            dxs_f, dh0f_b, sdW_f, sdb_f = \
                self.forward_cell.backward_sequence(dhf, cfs[b])
            dxs_b_rev, dh0b_b, sdW_b, sdb_b = \
                self.backward_cell.backward_sequence(dhb[::-1], cbs[b][::-1])
            # 后支输入梯度按扫描倒序给出，倒回时间正序以便两支逐时刻相加。
            dxs_b = dxs_b_rev[::-1]

            dh0_f[b] = [float(v) for v in dh0f_b]
            dh0_b[b] = [float(v) for v in dh0b_b]

            rowx = dxs[b]
            for t in range(L):
                row_f = dxs_f[t]
                row_b = dxs_b[t]
                for j in range(I):
                    acc = 0.0
                    acc += float(row_f[j])
                    if not math.isfinite(acc):
                        raise ValueError(
                            "dxs accumulated to a non-finite value")
                    acc += float(row_b[j])
                    if not math.isfinite(acc):
                        raise ValueError(
                            "dxs accumulated to a non-finite value")
                    rowx[t][j] = acc

            for k in range(3 * H):
                row_f = dW_f[k]
                srow_f = sdW_f[k]
                row_b = dW_b[k]
                srow_b = sdW_b[k]
                for j in range(I + H):
                    row_f[j] += srow_f[j]
                    if not math.isfinite(row_f[j]):
                        raise ValueError(
                            "dW_f accumulated to a non-finite value")
                    row_b[j] += srow_b[j]
                    if not math.isfinite(row_b[j]):
                        raise ValueError(
                            "dW_b accumulated to a non-finite value")
                db_f[k] += sdb_f[k]
                if not math.isfinite(db_f[k]):
                    raise ValueError(
                        "db_f accumulated to a non-finite value")
                db_b[k] += sdb_b[k]
                if not math.isfinite(db_b[k]):
                    raise ValueError(
                        "db_b accumulated to a non-finite value")

        return dxs, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b

    def forward_attn_padded(self, xs, lengths, q):
        """变长批注意力汇聚前向，返回 (c, w) 并保存 attn 批缓存。

        xs、lengths 完全沿用 forward_padded 的契约（xs 为非空 B×T×I 的 F
        嵌套列表，lengths 各项为 1<=值<=T 的非 bool int）；q 须为 B×2H 的
        F 列表（每样本一行长度 2H 的查询），否则抛 ValueError。任何失败都
        使既有缓存（含 forward、forward_padded 缓存）失效。

        先按 forward_padded 的同一前向语义求批双向输出 y（有效前缀为真实
        输出、padding 行为 2H 个 0.0）；对样本 b 取 v = y[b][:lengths[b]]
        （逐行新建的 float 行），调用 attention([q[b]], v, v, None) 得
        (c_b, w_b)：c 为 B×2H，c[b] 即 c_b[0]；w 为 B×T，w[b] 有效前缀
        为 w_b[0]，padding 位为 0.0。缓存须足以支撑
        backward_attn_padded：保存批 GRU 缓存及 v、q 的快照。结果均为逐层
        新建的 float 列表，不修改或复用输入与单元参数；任一中间量非有限
        同样抛 ValueError，实参数量错误沿用 Python 自带的 TypeError。
        """
        I, H = self.I, self.H
        # 任何失败的 forward_attn_padded 都使既有缓存失效。
        self._cache = None

        # y 复用 forward_padded 的全部校验与前向计算；成功后其缓存即当前
        # 缓存（标签 "padded"），随后改写为带 v、q 快照的 attn 缓存。其内
        # 部失败已自行清空缓存；其后任一步失败也须清掉它留下的 padded 缓存。
        y = self.forward_padded(xs, lengths)
        try:
            _, B, T, lengths, cfs, cbs = self._cache

            # q：B×2H 的 F 列表，逐层拷贝；反向 attention_backward 需用。
            q = self._check_padded_2d(q, B, 2 * H, "q")

            c = [None] * B
            w = [[0.0] * T for _ in range(B)]
            vs = [None] * B
            for b in range(B):
                L = lengths[b]
                # v 为 y[b] 有效前缀的逐行新建 float 快照，独立于返回的 y。
                v = [[float(x) for x in y[b][t]] for t in range(L)]
                vs[b] = v
                cb, wb = attention([q[b]], v, v, None)
                c[b] = [float(x) for x in cb[0]]
                wb0 = wb[0]
                row = w[b]
                for t in range(L):
                    wv = wb0[t]
                    if not math.isfinite(wv):
                        raise ValueError("attention weight became non-finite")
                    row[t] = float(wv)
        except ValueError:
            self._cache = None
            raise

        self._cache = ("attn", B, T, list(lengths), cfs, cbs, vs, q)
        return c, w

    def backward_attn_padded(self, dc):
        """变长批注意力汇聚反向，固定返回
        (dxs, dq, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b)。

        最近一次成功前向须为 forward_attn_padded，且其后无任何其他 forward
        类调用（forward、forward_padded、forward_attn_padded 自身的成功或
        失败调用均会替换或使缓存失效），否则抛 ValueError；dc 须为 B×2H
        的 F 列表，否则抛 ValueError。实参数量错误沿用 Python 自带的
        TypeError。

        对样本 b 以缓存的查询 q[b]、有效前缀 v（长 L_b=lengths[b]）调用
        attention_backward([q[b]], v, v, [dc[b]], None) 得
        (gq, gk, gv)（形状分别为 1×2H、L_b×2H、L_b×2H）：dq[b] = gq[0]
        （长 2H）；有效前缀输出梯度 dy 自 0.0 起依次加 gk[t]、gv[t]（两行
        长均 2H，按 t、维升序），padding 位置 0.0 并随后忽略。再以该 dy 按
        backward_padded 的同一语义反传批双向 GRU。dxs 形状 B×T×I（padding
        行为 I 个 0.0），dq 形状 B×2H；dh0_f、dh0_b、dW_f、db_f、dW_b、
        db_b 六项完全沿用 backward_padded 的形状与累加语义。所有结果均为
        逐层新建的 float 列表，不修改 dc、缓存或单元参数，重复调用结果确定；
        任一中间量非有限抛 ValueError。
        """
        I, H = self.I, self.H
        cache = self._cache
        if type(cache) is not tuple or len(cache) != 8 \
                or cache[0] != "attn":
            raise ValueError(
                "backward_attn_padded requires a successful "
                "forward_attn_padded pass before it")
        _, B, T, lengths, cfs, cbs, vs, qs = cache

        # dc：B×2H 的 F 列表，逐层 float 拷贝。
        dc = self._check_padded_2d(dc, B, 2 * H, "dc")

        dq = [None] * B
        # 各样本有效前缀的输出梯度 dy：自 0.0 起依次加 gk[t]、gv[t]，padding
        # 位不构造（反传只取有效前缀，dxs 的 padding 行由反传铺零）。
        dy_valid = [None] * B
        for b in range(B):
            L = lengths[b]
            gq, gk, gv = attention_backward(
                [qs[b]], vs[b], vs[b], [dc[b]], None)
            dq[b] = [float(x) for x in gq[0]]
            dy_b = [[0.0] * (2 * H) for _ in range(L)]
            for t in range(L):
                row = dy_b[t]
                gkt = gk[t]
                gvt = gv[t]
                for a in range(2 * H):
                    acc = 0.0
                    acc += gkt[a]
                    if not math.isfinite(acc):
                        raise ValueError(
                            "dy accumulated to a non-finite value")
                    acc += gvt[a]
                    if not math.isfinite(acc):
                        raise ValueError(
                            "dy accumulated to a non-finite value")
                    row[a] = acc
            dy_valid[b] = dy_b

        # 批 GRU 反传完全沿用 backward_padded 的语义；padding 位梯度为零。
        dxs, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b = self._padded_backprop(
            B, T, lengths, cfs, cbs, dy_valid)

        return dxs, dq, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b

    def forward_self_attn_padded(self, xs, lengths):
        """变长批自注意力前向，返回 (c, w) 并保存 selfattn 批缓存。

        xs、lengths 完全沿用 forward_padded 的契约（xs 为非空 B×T×I 的 F
        嵌套列表，lengths 各项为 1<=值<=T 的非 bool int）。任何失败都使
        既有缓存（含 forward、forward_padded、forward_attn_padded 缓存）
        失效。

        先按 forward_padded 的同一前向语义求批双向输出 y（有效前缀为真实
        输出、padding 行为 2H 个 0.0）；对样本 b 令 L=lengths[b]、取
        v = y[b][:L]（逐行新建的 float 行），调用 attention(v, v, v, None)
        得 (c_b, w_b)（形状分别为 L×2H、L×L）。c 为 B×T×2H，c[b] 有效
        前缀逐行取自 c_b，padding 行是 2H 个 0.0；w 为 B×T×T，w[b] 的
        L×L 有效块取自 w_b，padding 行、列均为 0.0。缓存须足以支撑
        backward_self_attn_padded：保存批 GRU 缓存及 v 的快照。结果均为
        逐层新建的 float 列表，不修改或复用输入与单元参数；任一中间量非
        有限同样抛 ValueError，实参数量错误沿用 Python 自带的 TypeError。
        """
        I, H = self.I, self.H
        # 任何失败的 forward_self_attn_padded 都使既有缓存失效。
        self._cache = None

        # y 复用 forward_padded 的全部校验与前向计算；成功后其缓存即当前
        # 缓存（标签 "padded"），随后改写为带 v 快照的 selfattn 缓存。其
        # 内部失败已自行清空缓存；其后任一步失败也须清掉它留下的 padded 缓存。
        y = self.forward_padded(xs, lengths)
        try:
            _, B, T, lengths, cfs, cbs = self._cache

            # 输出先铺零：有效块随后覆写，padding 行（及 w 的 padding 列）
            # 保持全新的零行。
            c = [[[0.0] * (2 * H) for _ in range(T)] for _ in range(B)]
            w = [[[0.0] * T for _ in range(T)] for _ in range(B)]
            vs = [None] * B
            for b in range(B):
                L = lengths[b]
                # v 为 y[b] 有效前缀的逐行新建 float 快照，独立于返回的 y；
                # 自注意力中 q、k、v 同为 v。
                v = [[float(x) for x in y[b][t]] for t in range(L)]
                vs[b] = v
                cb, wb = attention(v, v, v, None)
                c_b = c[b]
                w_b = w[b]
                for t in range(L):
                    cbt = cb[t]
                    crow = c_b[t]
                    wbrow = w_b[t]
                    wbt = wb[t]
                    for a in range(2 * H):
                        cv = float(cbt[a])
                        if not math.isfinite(cv):
                            raise ValueError(
                                "self-attention context became non-finite")
                        crow[a] = cv
                    for s in range(L):
                        wv = float(wbt[s])
                        if not math.isfinite(wv):
                            raise ValueError(
                                "attention weight became non-finite")
                        wbrow[s] = wv
        except ValueError:
            self._cache = None
            raise

        self._cache = ("selfattn", B, T, list(lengths), cfs, cbs, vs)
        return c, w

    def backward_self_attn_padded(self, dc):
        """变长批自注意力反向，固定返回
        (dxs, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b)。

        最近一次成功前向须为 forward_self_attn_padded，且其后无任何其他
        forward 类调用（forward、forward_padded、forward_attn_padded、
        forward_self_attn_padded 自身的成功或失败调用均会替换或使缓存失效），
        否则抛 ValueError；dc 须为 B×T×2H 的 F 嵌套列表（与前向 c 同形，
        padding 位经校验后忽略），否则抛 ValueError。实参数量错误沿用 Python
        自带的 TypeError。

        对样本 b 令 L=lengths[b]，以缓存的有效前缀 v（长 L）对 dc 的有效
        前缀调用 attention_backward(v, v, v, dc[:L], None) 得
        (gq, gk, gv)（形状均为 L×2H）：有效前缀输出梯度 dy 按 t、维升序自
        0.0 起依次累加 gq[t][a]、gk[t][a]、gv[t][a]，padding 位置 0.0 并
        随后忽略。再以该 dy 按 backward_padded 的同一语义反传批双向 GRU。
        dxs 形状 B×T×I（padding 行为 I 个 0.0）；dh0_f、dh0_b、dW_f、
        db_f、dW_b、db_b 六项完全沿用 backward_padded 的形状与按 b 升序
        累加语义。所有结果均为逐层新建的 float 列表，不修改 dc、缓存或单元
        参数，重复调用结果确定；任一中间量非有限抛 ValueError。
        """
        I, H = self.I, self.H
        cache = self._cache
        if type(cache) is not tuple or len(cache) != 7 \
                or cache[0] != "selfattn":
            raise ValueError(
                "backward_self_attn_padded requires a successful "
                "forward_self_attn_padded pass before it")
        _, B, T, lengths, cfs, cbs, vs = cache

        # dc：B×T×2H 的 F 嵌套列表；padding 位在此一并校验，随后忽略。
        dc = self._check_padded_3d(dc, B, T, 2 * H, "dc")

        # 各样本有效前缀的输出梯度 dy：按 t、维升序自 0.0 依次累加 gq、gk、
        # gv；padding 位不构造（反传只取有效前缀，dxs 的 padding 行由反传
        # 铺零）。
        dy_valid = [None] * B
        for b in range(B):
            L = lengths[b]
            dc_prefix = [dc[b][t] for t in range(L)]
            gq, gk, gv = attention_backward(
                vs[b], vs[b], vs[b], dc_prefix, None)
            dy_b = [[0.0] * (2 * H) for _ in range(L)]
            for t in range(L):
                row = dy_b[t]
                gqt = gq[t]
                gkt = gk[t]
                gvt = gv[t]
                for a in range(2 * H):
                    acc = 0.0
                    acc += gqt[a]
                    if not math.isfinite(acc):
                        raise ValueError(
                            "dy accumulated to a non-finite value")
                    acc += gkt[a]
                    if not math.isfinite(acc):
                        raise ValueError(
                            "dy accumulated to a non-finite value")
                    acc += gvt[a]
                    if not math.isfinite(acc):
                        raise ValueError(
                            "dy accumulated to a non-finite value")
                    row[a] = acc
            dy_valid[b] = dy_b

        # 批 GRU 反传完全沿用 backward_padded 的语义；padding 位梯度为零。
        return self._padded_backprop(
            B, T, lengths, cfs, cbs, dy_valid)

    def forward_self_attn_masked(self, xs, lengths, mask):
        """变长批掩码自注意力前向，返回 (c, w) 并保存 selfattn_masked 批缓存。

        xs、lengths 完全沿用 forward_padded 的契约（xs 为非空 B×T×I 的 F
        嵌套列表，lengths 各项为 1<=值<=T 的非 bool int）；mask 须为 B×T×T
        的列表且元素 type 恰为 bool（True 参与、False 屏蔽），padding 行、
        列仅校验后忽略；样本 b 令 L=lengths[b]，其有效 L×L 块每行至少一个
        True，否则抛 ValueError。任何失败都使既有缓存（含 forward、
        forward_padded、forward_attn_padded、forward_self_attn_padded 缓存）
        失效。

        先按 forward_padded 的同一前向语义求批双向输出 y（有效前缀为真实
        输出、padding 行为 2H 个 0.0）；对样本 b 取 v = y[b][:L]（逐行新建
        的 float 行），以有效块为掩码调用 attention(v, v, v, 有效块) 得
        (c_b, w_b)（形状分别为 L×2H、L×L）。c 为 B×T×2H，c[b] 有效前缀
        逐行取自 c_b，padding 行是 2H 个 0.0；w 为 B×T×T，w[b] 的 L×L
        有效块取自 w_b，padding 行、列均为 0.0。缓存须足以支撑
        backward_self_attn_masked：保存批 GRU 缓存及 v、有效块的独立快照。
        结果均为逐层新建的 float 列表，不修改或复用输入与单元参数；任一
        中间量非有限同样抛 ValueError，实参数量错误沿用 Python 自带的
        TypeError。
        """
        I, H = self.I, self.H
        # 任何失败的 forward_self_attn_masked 都使既有缓存失效。
        self._cache = None

        # y 复用 forward_padded 的全部校验与前向计算；成功后其缓存即当前
        # 缓存（标签 "padded"），随后改写为带 v、有效块快照的
        # selfattn_masked 缓存。其内部失败已自行清空缓存；其后任一步失败
        # 也须清掉它留下的 padded 缓存。
        y = self.forward_padded(xs, lengths)
        try:
            _, B, T, lengths, cfs, cbs = self._cache

            # mask：B×T×T 的列表，元素 type 恰为 bool；padding 行、列在此
            # 一并校验，随后忽略。
            if type(mask) is not list or len(mask) != B:
                raise ValueError("mask must be a list of shape %d×%d×%d"
                                 % (B, T, T))
            for b in range(B):
                mb = mask[b]
                if type(mb) is not list or len(mb) != T:
                    raise ValueError("mask must be a list of shape %d×%d×%d"
                                     % (B, T, T))
                for row in mb:
                    if type(row) is not list or len(row) != T:
                        raise ValueError(
                            "mask must be a list of shape %d×%d×%d"
                            % (B, T, T))
                    for m in row:
                        if type(m) is not bool:
                            raise ValueError(
                                "mask entries must be exactly bool, got %r"
                                % (m,))

            # 有效块：逐样本取 L×L 前缀块的逐层新建 bool 拷贝（缓存由此
            # 独立于调用方持有的 mask），每行至少一个 True。
            blocks = [None] * B
            for b in range(B):
                L = lengths[b]
                mb = mask[b]
                block = []
                for i in range(L):
                    row = [mb[i][j] for j in range(L)]
                    if not any(row):
                        raise ValueError(
                            "mask row %d of sample %d is entirely False"
                            % (i, b))
                    block.append(row)
                blocks[b] = block

            # 输出先铺零：有效块随后覆写，padding 行（及 w 的 padding 列）
            # 保持全新的零行。
            c = [[[0.0] * (2 * H) for _ in range(T)] for _ in range(B)]
            w = [[[0.0] * T for _ in range(T)] for _ in range(B)]
            vs = [None] * B
            for b in range(B):
                L = lengths[b]
                # v 为 y[b] 有效前缀的逐行新建 float 快照，独立于返回的 y；
                # 自注意力中 q、k、v 同为 v。
                v = [[float(x) for x in y[b][t]] for t in range(L)]
                vs[b] = v
                cb, wb = attention(v, v, v, blocks[b])
                c_b = c[b]
                w_b = w[b]
                for t in range(L):
                    cbt = cb[t]
                    crow = c_b[t]
                    wbrow = w_b[t]
                    wbt = wb[t]
                    for a in range(2 * H):
                        cv = float(cbt[a])
                        if not math.isfinite(cv):
                            raise ValueError(
                                "self-attention context became non-finite")
                        crow[a] = cv
                    for s in range(L):
                        wv = float(wbt[s])
                        if not math.isfinite(wv):
                            raise ValueError(
                                "attention weight became non-finite")
                        wbrow[s] = wv
        except ValueError:
            self._cache = None
            raise

        self._cache = ("selfattn_masked", B, T, list(lengths), cfs, cbs,
                       vs, blocks)
        return c, w

    def backward_self_attn_masked(self, dc):
        """变长批掩码自注意力反向，固定返回
        (dxs, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b)。

        最近一次成功前向须为 forward_self_attn_masked，且其后无任何其他
        forward 类调用（forward、forward_padded、forward_attn_padded、
        forward_self_attn_padded、forward_self_attn_masked 自身的成功或
        失败调用均会替换或使缓存失效），否则抛 ValueError；dc 须为
        B×T×2H 的 F 嵌套列表（与前向 c 同形，padding 位经校验后忽略），
        否则抛 ValueError。实参数量错误沿用 Python 自带的 TypeError。

        反向中任何 ValueError（dc 校验、attention_backward、dy 累加或批
        GRU 反传失败）都使缓存失效：失败后未重新前向再调用本方法仍抛
        ValueError；成功时缓存保留，可重复调用且结果不变。

        对样本 b 令 L=lengths[b]，以缓存的有效前缀 v（长 L）与有效块掩码
        对 dc 的有效前缀调用 attention_backward(v, v, v, dc[:L], 有效块)
        得 (gq, gk, gv)（形状均为 L×2H）：有效前缀输出梯度 dy 按 t、维
        升序自 0.0 起依次累加 gq[t][a]、gk[t][a]、gv[t][a]，padding 位置
        0.0 并随后忽略。再以该 dy 按 backward_padded 的同一语义反传批双向
        GRU。dxs 形状 B×T×I（padding 行为 I 个 0.0）；dh0_f、dh0_b、
        dW_f、db_f、dW_b、db_b 六项完全沿用 backward_padded 的形状与按 b
        升序累加语义。所有结果均为逐层新建的 float 列表，不修改 dc、缓存
        或单元参数，重复调用结果确定；任一中间量非有限抛 ValueError。
        """
        I, H = self.I, self.H
        cache = self._cache
        if type(cache) is not tuple or len(cache) != 8 \
                or cache[0] != "selfattn_masked":
            raise ValueError(
                "backward_self_attn_masked requires a successful "
                "forward_self_attn_masked pass before it")
        _, B, T, lengths, cfs, cbs, vs, blocks = cache

        try:
            # dc：B×T×2H 的 F 嵌套列表；padding 位在此一并校验，随后忽略。
            dc = self._check_padded_3d(dc, B, T, 2 * H, "dc")

            # 各样本有效前缀的输出梯度 dy：按 t、维升序自 0.0 依次累加 gq、
            # gk、gv；padding 位不构造（反传只取有效前缀，dxs 的 padding
            # 行由反传铺零）。
            dy_valid = [None] * B
            for b in range(B):
                L = lengths[b]
                dc_prefix = [dc[b][t] for t in range(L)]
                gq, gk, gv = attention_backward(
                    vs[b], vs[b], vs[b], dc_prefix, blocks[b])
                dy_b = [[0.0] * (2 * H) for _ in range(L)]
                for t in range(L):
                    row = dy_b[t]
                    gqt = gq[t]
                    gkt = gk[t]
                    gvt = gv[t]
                    for a in range(2 * H):
                        acc = 0.0
                        acc += gqt[a]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dy accumulated to a non-finite value")
                        acc += gkt[a]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dy accumulated to a non-finite value")
                        acc += gvt[a]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dy accumulated to a non-finite value")
                        row[a] = acc
                dy_valid[b] = dy_b

            # 批 GRU 反传完全沿用 backward_padded 的语义；padding 位梯度为零。
            return self._padded_backprop(
                B, T, lengths, cfs, cbs, dy_valid)
        except ValueError:
            # 任何失败（dc 校验、attention_backward、dy 累加或批反传）都使
            # 缓存失效：未重新前向再调用本方法仍抛 ValueError。
            self._cache = None
            raise

    def forward_self_multihead_attn_masked(self, xs, lengths, heads, mask):
        """变长批多头掩码自注意力前向，返回 (c, w) 并保存批缓存。

        xs、lengths 完全沿用 forward_padded 的契约（xs 为非空 B×T×I 的 F
        嵌套列表，lengths 各项为 1<=值<=T 的非 bool int）；heads 须为非
        bool 的正 int 且整除 2H；mask 须为 B×T×T 的列表且元素 type 恰为
        bool（True 参与、False 屏蔽），padding 行、列仅校验后忽略；样本 b
        令 L=lengths[b]，其有效 L×L 块每行至少一个 True，否则抛
        ValueError。任何失败都使既有缓存（含 forward、forward_padded、
        forward_attn_padded、forward_self_attn_padded、
        forward_self_attn_masked 缓存）失效。

        先按 forward_padded 的同一前向语义求批双向输出 y（有效前缀为真实
        输出、padding 行为 2H 个 0.0）；对样本 b 取 v = y[b][:L]（逐行新建
        的 float 行），调用 multihead_attention(v, v, v, heads, 有效块) 得
        (c_b, w_b)（形状分别为 L×2H、heads×L×L）。c 为 B×T×2H，c[b] 有效
        前缀逐行取自 c_b，padding 行是 2H 个 0.0；w 为 B×heads×T×T，
        w[b][h] 的 L×L 有效块取自 w_b[h]，padding 行、列均为 0.0。缓存须
        足以支撑 backward_self_multihead_attn_masked：保存批 GRU 缓存及
        v、有效块、heads 的独立快照。结果均为逐层新建的 float 列表，不修改
        或复用输入与单元参数；任一中间量非有限同样抛 ValueError，实参数量
        错误沿用 Python 自带的 TypeError。
        """
        I, H = self.I, self.H
        # 任何失败的 forward_self_multihead_attn_masked 都使既有缓存失效。
        self._cache = None

        # y 复用 forward_padded 的全部校验与前向计算；成功后其缓存即当前
        # 缓存（标签 "padded"），随后改写为带 v、有效块、heads 快照的多头
        # 掩码缓存。其内部失败已自行清空缓存；其后任一步失败也须清掉它留下
        # 的 padded 缓存。
        y = self.forward_padded(xs, lengths)
        try:
            _, B, T, lengths, cfs, cbs = self._cache

            # heads：非 bool 的正 int 且整除 2H（D=Dv=2H）。
            if type(heads) is bool or type(heads) is not int or heads <= 0:
                raise ValueError(
                    "heads must be a non-bool positive int, got %r" % (heads,))
            if (2 * H) % heads != 0:
                raise ValueError(
                    "heads (%d) must divide 2H=%d" % (heads, 2 * H))

            # mask：B×T×T 的列表，元素 type 恰为 bool；padding 行、列在此
            # 一并校验，随后忽略。
            if type(mask) is not list or len(mask) != B:
                raise ValueError("mask must be a list of shape %d×%d×%d"
                                 % (B, T, T))
            for b in range(B):
                mb = mask[b]
                if type(mb) is not list or len(mb) != T:
                    raise ValueError("mask must be a list of shape %d×%d×%d"
                                     % (B, T, T))
                for row in mb:
                    if type(row) is not list or len(row) != T:
                        raise ValueError(
                            "mask must be a list of shape %d×%d×%d"
                            % (B, T, T))
                    for m in row:
                        if type(m) is not bool:
                            raise ValueError(
                                "mask entries must be exactly bool, got %r"
                                % (m,))

            # 有效块：逐样本取 L×L 前缀块的逐层新建 bool 拷贝（缓存由此
            # 独立于调用方持有的 mask），每行至少一个 True。
            blocks = [None] * B
            for b in range(B):
                L = lengths[b]
                mb = mask[b]
                block = []
                for i in range(L):
                    row = [mb[i][j] for j in range(L)]
                    if not any(row):
                        raise ValueError(
                            "mask row %d of sample %d is entirely False"
                            % (i, b))
                    block.append(row)
                blocks[b] = block

            # 输出先铺零：有效块随后覆写，padding 行（及 w 的 padding 列）
            # 保持全新的零行；w 每样本含 heads 个 T×T 矩阵。
            c = [[[0.0] * (2 * H) for _ in range(T)] for _ in range(B)]
            w = [[[[0.0] * T for _ in range(T)] for _ in range(heads)]
                 for _ in range(B)]
            vs = [None] * B
            for b in range(B):
                L = lengths[b]
                # v 为 y[b] 有效前缀的逐行新建 float 快照，独立于返回的 y；
                # 自注意力中 q、k、v 同为 v。
                v = [[float(x) for x in y[b][t]] for t in range(L)]
                vs[b] = v
                cb, wb = multihead_attention(v, v, v, heads, blocks[b])
                c_b = c[b]
                w_b = w[b]
                for t in range(L):
                    cbt = cb[t]
                    crow = c_b[t]
                    for a in range(2 * H):
                        cv = float(cbt[a])
                        if not math.isfinite(cv):
                            raise ValueError(
                                "multihead self-attention context became "
                                "non-finite")
                        crow[a] = cv
                    for h in range(heads):
                        wbrow = w_b[h][t]
                        wbt = wb[h][t]
                        for s in range(L):
                            wv = float(wbt[s])
                            if not math.isfinite(wv):
                                raise ValueError(
                                    "attention weight became non-finite")
                            wbrow[s] = wv
        except ValueError:
            self._cache = None
            raise

        self._cache = ("self_multihead_attn_masked", B, T, list(lengths),
                       cfs, cbs, vs, blocks, heads)
        return c, w

    def backward_self_multihead_attn_masked(self, dc):
        """变长批多头掩码自注意力反向，固定返回
        (dxs, dh0_f, dh0_b, dW_f, db_f, dW_b, db_b)。

        须紧随一次成功的 forward_self_multihead_attn_masked（其后无任何
        其他 forward 类调用替换或使缓存失效），否则抛 ValueError；dc 须为
        B×T×2H 的 F 嵌套列表（与前向 c 同形，padding 位经校验后忽略），
        否则抛 ValueError。实参数量错误沿用 Python 自带的 TypeError。

        反向中任何 ValueError（状态非法、dc 校验、
        multihead_attention_backward、dy 累加或批 GRU 反传失败）都使缓存
        失效：失败后未重新前向再调用本方法仍抛 ValueError；成功时缓存保留，
        可重复调用且结果不变。

        对样本 b 令 L=lengths[b]，以缓存的有效前缀 v（长 L）、heads 与有效
        块掩码对 dc 的有效前缀调用
        multihead_attention_backward(v, v, v, dc[:L], heads, 有效块) 得
        (gq, gk, gv)（形状均为 L×2H）：有效前缀输出梯度 dy 按 t、维升序自
        0.0 起依次累加 gq[t][a]、gk[t][a]、gv[t][a]，padding 位置 0.0 并
        随后忽略。再以该 dy 按 backward_padded 的同一语义反传批双向 GRU。
        dxs 形状 B×T×I（padding 行为 I 个 0.0）；dh0_f、dh0_b、dW_f、
        db_f、dW_b、db_b 六项完全沿用 backward_padded 的形状与按 b 升序
        累加语义。所有结果均为逐层新建的 float 列表，不修改 dc、缓存或单元
        参数，重复调用结果确定；任一中间量非有限抛 ValueError。
        """
        I, H = self.I, self.H
        cache = self._cache
        try:
            if type(cache) is not tuple or len(cache) != 9 \
                    or cache[0] != "self_multihead_attn_masked":
                raise ValueError(
                    "backward_self_multihead_attn_masked requires a "
                    "successful forward_self_multihead_attn_masked pass "
                    "before it")
            _, B, T, lengths, cfs, cbs, vs, blocks, heads = cache

            # dc：B×T×2H 的 F 嵌套列表；padding 位在此一并校验，随后忽略。
            dc = self._check_padded_3d(dc, B, T, 2 * H, "dc")

            # 各样本有效前缀的输出梯度 dy：按 t、维升序自 0.0 依次累加 gq、
            # gk、gv；padding 位不构造（反传只取有效前缀，dxs 的 padding
            # 行由反传铺零）。
            dy_valid = [None] * B
            for b in range(B):
                L = lengths[b]
                dc_prefix = [dc[b][t] for t in range(L)]
                gq, gk, gv = multihead_attention_backward(
                    vs[b], vs[b], vs[b], dc_prefix, heads, blocks[b])
                dy_b = [[0.0] * (2 * H) for _ in range(L)]
                for t in range(L):
                    row = dy_b[t]
                    gqt = gq[t]
                    gkt = gk[t]
                    gvt = gv[t]
                    for a in range(2 * H):
                        acc = 0.0
                        acc += gqt[a]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dy accumulated to a non-finite value")
                        acc += gkt[a]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dy accumulated to a non-finite value")
                        acc += gvt[a]
                        if not math.isfinite(acc):
                            raise ValueError(
                                "dy accumulated to a non-finite value")
                        row[a] = acc
                dy_valid[b] = dy_b

            # 批 GRU 反传完全沿用 backward_padded 的语义；padding 位梯度为零。
            return self._padded_backprop(
                B, T, lengths, cfs, cbs, dy_valid)
        except ValueError:
            # 任何失败（状态非法、dc 校验、multihead_attention_backward、
            # dy 累加或批反传）都使缓存失效：未重新前向再调用本方法仍抛
            # ValueError。
            self._cache = None
            raise


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


_LSTM_MHA_MODEL_KEYS = ["version", "vocab", "W", "b", "Wq", "Wk", "Wv", "Wo",
                        "heads", "Why", "by", "h0", "c0"]


def _load_perplexity_lstm_mha_model(path):
    """读取并校验 perplexity-lstm-mha 模型文件，返回解包后的十二元组。

    文件须为 UTF-8 编码的 JSON 对象，顶层键恰为
    version、vocab、W、b、Wq、Wk、Wv、Wo、heads、Why、by、h0、c0 且按此
    顺序出现（重复或多余均非法）：version 的 type 恰为 int 且值为 4；
    vocab 与 h0、c0 的契约同 perplexity-lstm；W、b 沿用 4H×(V+H) 与 4H
    形状；Wq、Wk、Wv、Wo 均为 H×H 的 F 矩阵；heads 为非 bool 的正 int
    且整除 H；Why、by 沿用 V×H 与 V 形状。任何读取、UTF-8、JSON 或校验
    失败均抛 ValueError（或 OSError）。
    """
    with open(path, "rb") as f:
        raw = f.read()
    # 先按严格 UTF-8 解码，再交由 json 解析（object_pairs_hook 保留键序与
    # 重复键，root 非对象时不会得到 (key, value) 二元组列表）。
    text = raw.decode("utf-8")
    pairs = json.loads(text, object_pairs_hook=list)
    if type(pairs) is not list or len(pairs) != len(_LSTM_MHA_MODEL_KEYS):
        raise ValueError("model must be a JSON object with exactly 13 keys")
    for pair, key in zip(pairs, _LSTM_MHA_MODEL_KEYS):
        if type(pair) is not tuple or len(pair) != 2 or pair[0] != key:
            raise ValueError("model keys must be exactly %r in order"
                             % _LSTM_MHA_MODEL_KEYS)
    model = dict(pairs)

    version = model["version"]
    if type(version) is bool or type(version) is not int or version != 4:
        raise ValueError("version must be exactly non-bool int 4, got %r"
                         % (version,))

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

    projections = {}
    for pname in ("Wq", "Wk", "Wv", "Wo"):
        P = model[pname]
        if type(P) is not list or len(P) != H:
            raise ValueError("%s must be a list of shape %d×%d"
                             % (pname, H, H))
        for row in P:
            if type(row) is not list or len(row) != H:
                raise ValueError("%s must be a list of shape %d×%d"
                                 % (pname, H, H))
            for v in row:
                if not _is_f(v):
                    raise ValueError(
                        "%s entries must be finite numbers, got %r"
                        % (pname, v))
        projections[pname] = P

    heads = model["heads"]
    if type(heads) is bool or type(heads) is not int or heads <= 0:
        raise ValueError(
            "heads must be a non-bool positive int, got %r" % (heads,))
    if H % heads != 0:
        raise ValueError("heads (%d) must divide H=%d" % (heads, H))

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

    return (vocab, W, b, projections["Wq"], projections["Wk"],
            projections["Wv"], projections["Wo"], heads, Why, by, h0, c0)


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


_GRU_MODEL_KEYS = ["version", "vocab", "W", "b", "Why", "by", "h0"]


def _load_perplexity_gru_model(path):
    """读取并校验 perplexity-gru 模型文件，返回解包后的六元组。

    文件须为 UTF-8 编码的 JSON 对象，顶层键恰为
    version、vocab、W、b、Why、by、h0 且按此顺序出现（重复或多余均非法）：
    version 的 type 恰为 int 且值为 3；vocab 的契约与 perplexity 相同
    （非空列表，每项是恰含一个码点的 str，元素唯一且按码点严格升序）；
    其余五项为 F 列表，形状依次为 3H×(V+H)、3H、V×H、V、H，其中
    V=len(vocab)、H=len(h0)>0。任何读取、UTF-8、JSON 或校验失败均抛
    ValueError（或 OSError）。
    """
    with open(path, "rb") as f:
        raw = f.read()
    # 先按严格 UTF-8 解码，再交由 json 解析（object_pairs_hook 保留键序与
    # 重复键，root 非对象时不会得到 (key, value) 二元组列表）。
    text = raw.decode("utf-8")
    pairs = json.loads(text, object_pairs_hook=list)
    if type(pairs) is not list or len(pairs) != len(_GRU_MODEL_KEYS):
        raise ValueError("model must be a JSON object with exactly 7 keys")
    for pair, key in zip(pairs, _GRU_MODEL_KEYS):
        if type(pair) is not tuple or len(pair) != 2 or pair[0] != key:
            raise ValueError("model keys must be exactly %r in order"
                             % _GRU_MODEL_KEYS)
    model = dict(pairs)

    version = model["version"]
    if type(version) is not int or version != 3:
        raise ValueError("version must be exactly int 3, got %r" % (version,))

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
    if type(W) is not list or len(W) != 3 * H:
        raise ValueError("W must be a list of shape %d×%d" % (3 * H, V + H))
    for row in W:
        if type(row) is not list or len(row) != V + H:
            raise ValueError("W must be a list of shape %d×%d" % (3 * H, V + H))
        for v in row:
            if not _is_f(v):
                raise ValueError("W entries must be finite numbers, got %r"
                                 % (v,))

    b = model["b"]
    if type(b) is not list or len(b) != 3 * H:
        raise ValueError("b must be a list of length %d" % (3 * H))
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

    return vocab, W, b, Why, by, h0


def _perplexity_gru(model_path, corpus_path):
    """计算 GRU 语言模型在给定语料上的困惑度，返回待写出的字符串。

    MODEL 为 version 3 的 GRU 模型：UTF-8 JSON 对象，顶层键恰为
    version、vocab、W、b、Why、by、h0（键序固定、无重复），W 形状
    3H×(V+H)、b 形状 3H；CORPUS 沿用 perplexity 的严格 UTF-8 全文码点、
    词表及至少 2 码点契约。

    置 h=h0、L=0.0、T=len(CORPUS)-1，t 升序：以当前字符的 V 长 one-hot
    为 x，调用装入 W、b 的 GRUCell.forward(x, h)，取返回的首项更新 h；
    随后按 perplexity 相同的下标、float 偏置与升序累加规则计算
    z_k = by_k + Σ_j Why_k,j*h_j，并以减最大值的 log-sum-exp 累加下一
    字符的负对数似然。任一中间量非有限（含最终 exp(L/T) 溢出）均抛
    ValueError。成功返回 format(exp(L/T), '.17g') + '\\n'。
    """
    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    cell = GRUCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    h = list(h0)
    L = 0.0
    T = len(ids) - 1
    for t in range(T):
        y = ids[t + 1]

        # 当前字符的 V 长 one-hot 输入。
        x = [0.0] * V
        x[ids[t]] = 1.0

        h, _cache = cell.forward(x, h)

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


def _train_gru(model_path, corpus_path, out_path):
    """对 GRU 模型做一次全语料 SGD 更新并把新模型写入 OUT。

    MODEL、CORPUS 完全沿用 perplexity-gru version 3 的七键、F、严格
    UTF-8、形状、词表及语料至少 2 码点契约。置 h=h0，t 升序：以当前字符
    的 V 长 one-hot 为 x，调用装入 W、b 的 GRUCell.forward(x, h)，以返回
    的首项更新 h 并缓存 cache；输出层 logit、softmax、g=p-onehot(y)、
    dWhy += g⊗h、dby += g 与 dhs = Whyᵀg 的公式、下标与 t/k/j 累加顺序
    完全沿用 train-lstm，仅隐状态换为 GRU 的 h。随后调用
    backward_sequence(dhs, caches) 取得整条序列的 dW、db。

    依 dW、db、dWhy、dby 行序求全局范数，超过 5.0 即统一缩放至 5.0；四组
    参数减去 0.1 倍梯度，h0 不变。任一中间量或结果非有限均抛 ValueError。

    OUT 复用 perplexity-gru 模型的七个键、键序与形状（version 恰为
    int 3），数组元素均转为 float；紧凑 JSON 参数、UTF-8 编码、末尾 LF
    及负零口径均沿用 train-lstm。
    """
    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    cell = GRUCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    # t 升序以 h0 为初态逐步前向，缓存每一步的隐状态与 cache。
    hs = []
    caches = []
    h = list(h0)
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0
        h, cache = cell.forward(x, h)
        hs.append(h)
        caches.append(cache)

    dWhy = [[0.0] * H for _ in range(V)]
    dby = [0.0] * V
    dhs = [[0.0] * H for _ in range(T)]

    for t in range(T):
        n = hs[t]
        y = ids[t + 1]

        # logit：与 train-lstm 相同，自 float 偏置起依 j 升序累加。
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

    _dxs, _dh0, dW, db = cell.backward_sequence(dhs, caches)

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

    # 四组参数减 0.1 倍（裁剪后的）梯度；h0 不变。结果非有限即失败。
    def _updated(old, grad):
        value = float(old) - 0.1 * (grad * scale)
        if not math.isfinite(value):
            raise ValueError("updated parameter became non-finite")
        return value

    new_W = [[_updated(W[k][j], dW[k][j]) for j in range(V + H)]
             for k in range(3 * H)]
    new_b = [_updated(b[k], db[k]) for k in range(3 * H)]
    new_Why = [[_updated(Why[k][j], dWhy[k][j]) for j in range(H)]
               for k in range(V)]
    new_by = [_updated(by[k], dby[k]) for k in range(V)]

    obj = {
        "version": 3,
        "vocab": vocab,
        "W": new_W,
        "b": new_b,
        "Why": new_Why,
        "by": new_by,
        "h0": [float(v) for v in h0],
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


def _sample_gru(model_path, start, seed_text, temperature_text, length_text):
    """从 GRU 语言模型采样 LENGTH 个码点，返回待写出的字符串。

    MODEL 严格沿用 perplexity-gru 的 version 3 七键顺序、F、形状及 UTF-8
    契约；START、SEED、TEMPERATURE、LENGTH 的词法与校验完全沿用 _sample。
    整次调用仅初始化一次 r=random.Random(int(SEED))，不写任何文件。

    置 h=list(h0)、x=START 索引。循环 LENGTH 次：以 x 的 V 长 one-hot 调用
    装入 W、b 的 GRUCell.forward(x, h)，取返回的首项更新 h；按 k、j 升序
    自 float(by[k]) 累加 Why[k][j]*h[j] 计算 logit z；温度缩放、稳定
    softmax、随机阈值、k 升序累计与选中规则逐项沿用 _sample；追加
    vocab[k] 并令 x=k。任一中间量非有限均抛 ValueError。成功返回 LENGTH
    个码点再加一个 LF。
    """
    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    cell = GRUCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    rng = random.Random(seed)
    h = list(h0)
    x = vocab.index(start)
    out = []

    for _t in range(length):
        # 当前字符的 V 长 one-hot 输入，推进 GRU 隐状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, _cache = cell.forward(xvec, h)

        # z_k = by_k + Σ_j Why_k,j*h_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, h)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
        m = max(a)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
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


def _sample_gru_attn(model_path, start, seed_text, temperature_text,
                     length_text, window_text):
    """带注意力上下文从 GRU 语言模型采样 LENGTH 个码点，返回待写出的字符串。

    MODEL、START、SEED、TEMPERATURE、LENGTH 的读取、词法、形状、F、确定性
    抽样及失败契约完全沿用 _sample_gru；WINDOW 整串匹配 [1-9][0-9]*（任意
    位数均合法，不转 int），安全截取沿用 perplexity-gru-attn，否则抛
    ValueError。整次调用仅初始化一次 r=random.Random(int(SEED))，不写任何
    文件。

    置 h=h0、x=START 索引、memory=[h0]。循环 LENGTH 次：以 x 的 V 长
    one-hot 调用装入 W、b 的 GRUCell.forward(x, h)，取返回的首项更新 h；
    M 取 memory 末尾 min(WINDOW, len(memory)) 项（顺序从旧到新），以
    attention([h], M, M, None) 返回首项 ctx，按 i 升序令
    u[i] = h[i] + ctx[0][i]。logit 仅以 u 替代 h，按 perplexity-gru-attn
    的顺序计算 Why/by 仿射；温度缩放、稳定 softmax、随机阈值、k 升序累计
    与选中规则逐项沿用 _sample_gru；追加 vocab[k]，令 x=k，再向 memory
    追加 h 的 float 副本。任一中间量非有限均抛 ValueError。成功返回
    LENGTH 个码点再加一个 LF。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    cell = GRUCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    rng = random.Random(seed)
    # 与 perplexity-gru-attn 相同：h0 保留模型原值，attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    x = vocab.index(start)
    out = []

    for _t in range(length):
        # 当前字符的 V 长 one-hot 输入，推进 GRU 隐状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, _cache = cell.forward(xvec, h)

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # z_k = by_k + Σ_j Why_k,j*u_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, u)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
        m = max(a)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
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


def _sample_gru_attn_anneal(model_path, start, seed_text, start_t_text,
                            end_t_text, length_text, window_text):
    """以线性退火温度、带注意力上下文从 GRU 语言模型采样 LENGTH 个码点。

    除温度外，MODEL、START、SEED、LENGTH、WINDOW、GRU 状态、注意力记忆、
    Why/by logit、稳定 softmax 及词表升序阈值抽样均沿用 _sample_gru_attn；
    WINDOW 整串匹配 [1-9][0-9]*（任意位数均合法，不转 int），安全截取沿用
    perplexity-gru-attn，否则抛 ValueError。START_T、END_T 各经 float()
    解析，结果须有限且严格大于 0。整次调用仅初始化一次
    r=random.Random(int(SEED))，不写任何文件。

    LENGTH 为 0 时不计算温度；为 1 时仅用 START_T；否则 t 自 0 升序，第
    t 步温度严格按 Python 表达式 START_T+(END_T-START_T)*t/(LENGTH-1)
    求值，结果非有限或不大于 0 即抛 ValueError。置 h=h0、x=START 索引、
    memory=[h0]，每步以该温度替换 _sample_gru_attn 的固定温度，沿用同一
    h、x 和 memory，生成后更新 x 并向 memory 追加 h 的 float 副本。任一
    中间量非有限均抛 ValueError。成功返回 LENGTH 个码点再加一个 LF。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    cell = GRUCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    rng = random.Random(seed)
    # 与 _sample_gru_attn 相同：h0 保留模型原值，attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    x = vocab.index(start)
    out = []

    for t in range(length):
        temperature = temperature_at(t)
        # 当前字符的 V 长 one-hot 输入，推进 GRU 隐状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, _cache = cell.forward(xvec, h)

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # z_k = by_k + Σ_j Why_k,j*u_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, u)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
        m = max(a)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
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


def _gru_cell_loaded(V, H, W, b):
    """不经随机初始化构造一个已装入模型 W、b 的 GRUCell。

    GRUCell.__init__ 会以 random.Random(0) 生成一份随即被覆盖的 W，使采样
    入口在采样随机源之外额外构造一次 random.Random；此处以 object.__new__
    跳过该随机初始化，仅设置 forward 所需的 I、H、W、b，令整次采样对
    random.Random 的唯一一次构造恰为 random.Random(int(SEED))。W 逐行复制
    为新列表、b 复制为新列表，行为与“GRUCell(V,H) 后覆盖 W、b”逐值一致。
    """
    cell = object.__new__(GRUCell)
    cell.I = V
    cell.H = H
    cell.W = [list(row) for row in W]
    cell.b = list(b)
    return cell


def _sample_gru_attn_topp(model_path, start, seed_text, start_t_text,
                          end_t_text, top_p_text, length_text, window_text):
    """以线性退火温度、top-p 核选样、带注意力上下文从 GRU 语言模型采样。

    除 TOP_P 及核选样外，MODEL、START、SEED、LENGTH、WINDOW 的校验，以及
    LENGTH 的 0/1 语义、GRU 状态、注意力记忆、线性温度、Why/by logit、稳定
    softmax 与有限性失败契约均沿用 _sample_gru_attn_anneal；WINDOW 整串匹配
    [1-9][0-9]*（任意位数均合法，不转 int），安全截取沿用
    perplexity-gru-attn，否则抛 ValueError。TOP_P 经 float() 解析，结果须
    有限且 0<TOP_P<=1，否则抛 ValueError。整次调用仅初始化一次
    r=random.Random(int(SEED))，不写任何文件。

    每步先按原顺序求 a_k=z_k/T、m=max(a)、e_k=exp(a_k-m)，d 自 0.0 依 k
    升序累加。再将索引按 (-e_k, k) 升序排列（e 降序、并列时 k 升序），依
    该序自 0.0 累加 e，截取首个使累计值 >=TOP_P*d 的最短前缀；s 为其自
    0.0 依该序累加所得之和。令 u=r.random()*s，再按前缀顺序自 0.0 累加
    e，选首个累计值严格大于 u 的索引；无则取前缀末项。其字符追加到输出并
    作为下一输入 x，随后向 memory 追加 h 的 float 副本。任一新增运算非有
    限均抛 ValueError。成功返回 LENGTH 个码点再加一个 LF。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    # TOP_P：float() 可解析且有限，0<TOP_P<=1。
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

    # 不经随机初始化装入 W、b：整次调用对 random.Random 的唯一一次构造即
    # 下方 random.Random(seed)，LENGTH 为 0、1 或选样回退时亦如此。
    cell = _gru_cell_loaded(V, H, W, b)

    rng = random.Random(seed)
    # 与 _sample_gru_attn 相同：h0 保留模型原值，attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    x = vocab.index(start)
    out = []

    for t in range(length):
        temperature = temperature_at(t)
        # 当前字符的 V 长 one-hot 输入，推进 GRU 隐状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, _cache = cell.forward(xvec, h)

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # z_k = by_k + Σ_j Why_k,j*u_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, u)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
        m = max(a)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
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

        # 核阈值 TOP_P*d 须有限。
        target = top_p * d
        if not math.isfinite(target):
            raise ValueError("top-p target accumulated non-finitely")

        # 索引按 (-e_k, k) 升序：e 降序、并列时 k 升序。
        order = sorted(range(V), key=lambda k: (-e[k], k))

        # 依该序自 0.0 累加 e，截取首个累计值 >=TOP_P*d 的最短前缀；s 为
        # 其自 0.0 依该序累加所得之和。浮点求和顺序不同可能令全量累计与
        # d 相差一 ULP，此时以全量索引为前缀（数学上其和恰为 d>=目标）。
        prefix = []
        s = 0.0
        reached = False
        for idx in order:
            prefix.append(idx)
            s += e[idx]
            if not math.isfinite(s):
                raise ValueError("top-p prefix accumulated non-finitely")
            if s >= target:
                reached = True
                break
        if not reached:
            prefix = list(order)

        # u=r.random()*s；按前缀顺序自 0.0 累加 e，选首个累计值严格大于
        # u 者；无则取前缀末项。
        threshold = rng.random() * s
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = prefix[-1]
        cum = 0.0
        for idx in prefix:
            cum += e[idx]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = idx
                break

        out.append(vocab[chosen])
        x = chosen
        memory.append([float(v) for v in h])

    return "".join(out) + "\n"


def _sample_gru_attn_topk(model_path, start, seed_text, start_t_text,
                          end_t_text, top_k_text, length_text, window_text):
    """以线性退火温度、top-k 选样、带注意力上下文从 GRU 语言模型采样。

    除 TOP_K 及候选选样外，MODEL、START、SEED、START_T、END_T、LENGTH、
    WINDOW 的校验、LENGTH 的 0/1 语义、GRU 状态、注意力记忆、线性温度、
    Why/by logit、稳定 softmax、随机源生命周期与有限性失败契约均沿用修复后
    的 _sample_gru_attn_topp；WINDOW 整串匹配 [1-9][0-9]*（任意位数均合
    法，不转 int），安全截取沿用 perplexity-gru-attn，否则抛 ValueError。
    整次调用仅构造一次 r=random.Random(int(SEED))（LENGTH 为 0、1 或选样
    回退时亦然），不写任何文件。

    TOP_K 整串匹配 [1-9][0-9]*，且数学值 K 不超过 V=len(vocab)。先按十
    进制位数及同长度字典序与 V 的十进制文本比较：位数更长、或同位数且字
    典序更大即越界失败；仅比较通过后才转 int，任意位数文本都不触发整数
    转换异常。

    每步先按原顺序求 a_k=z_k/T、m=max(a)、e_k=exp(a_k-m)，d 自 0.0 依 k
    升序累加。再将索引按 (-e_k, k) 升序排列（e 降序、并列时 k 升序），候
    选恰为该序前 K 项；s 自 0.0 按候选序累加 e。令 u=r.random()*s，再按
    候选序自 0.0 累加 e，选首个累计值严格大于 u 的索引；无则取候选末项。
    其字符追加到输出并作为下一输入 x，随后向 memory 追加 h 的 float 副
    本。任一新增运算非有限均抛 ValueError。成功返回 LENGTH 个码点再加一
    个 LF。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    # TOP_K：整串匹配 [1-9][0-9]*，且 K<=V。先以十进制位数、同位数字典序
    # 与 str(V) 比较，越界即失败；仅通过后才 int()，任何长度文本都不会触发
    # 整数转换异常（str(V) 受内存约束而位数有界）。
    if not _WINDOW_RE.match(top_k_text):
        raise ValueError("TOP_K must match [1-9][0-9]*")
    v_text = str(V)
    if len(top_k_text) > len(v_text) or (
            len(top_k_text) == len(v_text) and top_k_text > v_text):
        raise ValueError("TOP_K must not exceed len(vocab)")
    top_k = int(top_k_text)

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

    # 不经随机初始化装入 W、b：整次调用对 random.Random 的唯一一次构造即
    # 下方 random.Random(seed)，LENGTH 为 0、1 或选样回退时亦如此。
    cell = _gru_cell_loaded(V, H, W, b)

    rng = random.Random(seed)
    # 与 _sample_gru_attn 相同：h0 保留模型原值，attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    x = vocab.index(start)
    out = []

    for t in range(length):
        temperature = temperature_at(t)
        # 当前字符的 V 长 one-hot 输入，推进 GRU 隐状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, _cache = cell.forward(xvec, h)

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # z_k = by_k + Σ_j Why_k,j*u_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, u)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
        m = max(a)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
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

        # 索引按 (-e_k, k) 升序：e 降序、并列时 k 升序；候选恰为前 K 项。
        order = sorted(range(V), key=lambda k: (-e[k], k))
        candidates = order[:top_k]

        # s 自 0.0 按候选序累加 e。
        s = 0.0
        for idx in candidates:
            s += e[idx]
            if not math.isfinite(s):
                raise ValueError("top-k mass accumulated non-finitely")

        # u=r.random()*s；按候选序自 0.0 累加 e，选首个累计值严格大于 u
        # 者；无则取候选末项。
        threshold = rng.random() * s
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = candidates[-1]
        cum = 0.0
        for idx in candidates:
            cum += e[idx]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = idx
                break

        out.append(vocab[chosen])
        x = chosen
        memory.append([float(v) for v in h])

    return "".join(out) + "\n"


def _sample_gru_attn_topk_topp(model_path, start, seed_text, start_t_text,
                               end_t_text, top_k_text, top_p_text,
                               length_text, window_text):
    """以线性退火温度、top-k 截断后 top-p 核选样，带注意力从 GRU 模型采样。

    除 TOP_P 及 top-k 之后的 top-p 前缀筛选外，MODEL、START、SEED、
    START_T、END_T、LENGTH、WINDOW 的校验、LENGTH 的 0/1 语义、GRU 状态、
    注意力记忆、线性温度、Why/by logit、稳定 softmax、随机源生命周期与有
    限性失败契约均沿用 _sample_gru_attn_topk；WINDOW 整串匹配
    [1-9][0-9]*（任意位数均合法，不转 int），安全截取沿用
    perplexity-gru-attn，否则抛 ValueError。整次调用仅构造一次
    r=random.Random(int(SEED))（LENGTH 为 0、1 或选样回退时亦然），不写
    任何文件。

    TOP_K 整串匹配 [1-9][0-9]*，且数学值 K 不超过 V=len(vocab)。先按十
    进制位数及同长度字典序与 V 的十进制文本比较：位数更长、或同位数且字
    典序更大即越界失败；仅比较通过后才转 int，任意位数文本都不触发整数
    转换异常。TOP_P 经 float() 解析，结果须有限且 0<TOP_P<=1，否则抛
    ValueError。

    每步先按原顺序求 a_k=z_k/T、m=max(a)、e_k=exp(a_k-m)，d 自 0.0 依 k
    升序累加。再将索引按 (-e_k, k) 升序排列（e 降序、并列时 k 升序）并
    取前 K 项；sK 自 0.0 按该序累加 e，令 target=TOP_P*sK，再从 0.0 按
    该序累加 e，保留首个使累计值 >=target 的最短前缀，s 为此前缀累计和。
    令 u=r.random()*s，再按前缀序自 0.0 累加 e，选首个累计值严格大于 u
    的索引；无则取前缀末项。其字符追加到输出并作为下一输入 x，随后向
    memory 追加 h 的 float 副本。任一新增乘法或累加非有限均抛
    ValueError。成功返回 LENGTH 个码点再加一个 LF。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    # TOP_K：整串匹配 [1-9][0-9]*，且 K<=V。先以十进制位数、同位数字典序
    # 与 str(V) 比较，越界即失败；仅通过后才 int()，任何长度文本都不会触发
    # 整数转换异常（str(V) 受内存约束而位数有界）。
    if not _WINDOW_RE.match(top_k_text):
        raise ValueError("TOP_K must match [1-9][0-9]*")
    v_text = str(V)
    if len(top_k_text) > len(v_text) or (
            len(top_k_text) == len(v_text) and top_k_text > v_text):
        raise ValueError("TOP_K must not exceed len(vocab)")
    top_k = int(top_k_text)

    # TOP_P：float() 可解析且有限，0<TOP_P<=1。
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

    # 不经随机初始化装入 W、b：整次调用对 random.Random 的唯一一次构造即
    # 下方 random.Random(seed)，LENGTH 为 0、1 或选样回退时亦如此。
    cell = _gru_cell_loaded(V, H, W, b)

    rng = random.Random(seed)
    # 与 _sample_gru_attn 相同：h0 保留模型原值，attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    x = vocab.index(start)
    out = []

    for t in range(length):
        temperature = temperature_at(t)
        # 当前字符的 V 长 one-hot 输入，推进 GRU 隐状态。
        xvec = [0.0] * V
        xvec[x] = 1.0
        h, _cache = cell.forward(xvec, h)

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # z_k = by_k + Σ_j Why_k,j*u_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, u)

        # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k 升序累加。
        a = [0.0] * V
        for k in range(V):
            ak = z[k] / temperature
            if not math.isfinite(ak):
                raise ValueError("scaled logit became non-finite")
            a[k] = ak
        m = max(a)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
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

        # 索引按 (-e_k, k) 升序：e 降序、并列时 k 升序；取前 K 项。
        order = sorted(range(V), key=lambda k: (-e[k], k))
        candidates = order[:top_k]

        # sK 自 0.0 按候选序累加 e；target=TOP_P*sK 须有限。
        s_k = 0.0
        for idx in candidates:
            s_k += e[idx]
            if not math.isfinite(s_k):
                raise ValueError("top-k mass accumulated non-finitely")
        target = top_p * s_k
        if not math.isfinite(target):
            raise ValueError("top-p target accumulated non-finitely")

        # 再从 0.0 按候选序累加 e，保留首个累计值 >=target 的最短前缀；s
        # 为此前缀累计和。因累加顺序相同，末项累计恰为 sK>=target，前缀必
        # 存在。
        prefix = []
        s = 0.0
        for idx in candidates:
            prefix.append(idx)
            s += e[idx]
            if not math.isfinite(s):
                raise ValueError("top-p prefix accumulated non-finitely")
            if s >= target:
                break

        # u=r.random()*s；按前缀顺序自 0.0 累加 e，选首个累计值严格大于
        # u 者；无则取前缀末项。
        threshold = rng.random() * s
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = prefix[-1]
        cum = 0.0
        for idx in prefix:
            cum += e[idx]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = idx
                break

        out.append(vocab[chosen])
        x = chosen
        memory.append([float(v) for v in h])

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


def _mha_cross_context(n, memory, mha):
    """以 mha.forward_cross([n], M, None) 的首行输出逐项加进 n，返回 u。

    逐项令 u_i = n_i + ctx[0][i]，相加结果非有限即抛 ValueError。
    """
    ctx, _w = mha.forward_cross([n], memory, None)
    c0 = ctx[0]
    u = [0.0] * len(n)
    for i in range(len(n)):
        ui = n[i] + c0[i]
        if not math.isfinite(ui):
            raise ValueError("attention-adjusted hidden state became "
                             "non-finite")
        u[i] = ui
    return u


def _perplexity_lstm_mha(model_path, corpus_path, window_text):
    """LSTM 加投影多头注意力的字符困惑度，返回待写出的字符串。

    MODEL 沿用 perplexity-lstm 的 vocab、F 及六组数组契约，version 4
    新增 Wq、Wk、Wv、Wo 四个 H×H F 投影矩阵与非 bool 正 int 的 heads
    （整除 H），顶层键恰为
    version、vocab、W、b、Wq、Wk、Wv、Wo、heads、Why、by、h0、c0 且按
    此顺序出现；CORPUS 沿用严格 UTF-8 全文码点、词表及至少 2 码点契约；
    WINDOW 整串匹配 [1-9][0-9]*（任意位数均合法，不转 int），否则抛
    ValueError。

    置 h=h0、c=c0、L=0.0、T=len(CORPUS)-1、memory=[h0]，t 升序：以当前
    字符的 V 长 one-hot 为 x，调用装入 W、b 的 LSTMCell.forward(x, h, c)，
    取前两项更新 h、c；M 取 memory 末尾至多 WINDOW 项（顺序从旧到新），
    装入四组投影构造 MHA(H, heads)，以其 forward_cross([h], M, None)
    返回首项 ctx，按 i 升序令 u[i] = h[i] + ctx[0][i]。logit 仅以 u 替代
    perplexity-lstm 中的 h，其余 Why/by 仿射、稳定 log-sum-exp 及下一字符
    负对数似然的下标与累加顺序均与 perplexity-lstm 相同；随后向 memory
    追加 h 的 float 副本。任一运算非有限（含最终 exp(L/T) 溢出）均抛
    ValueError。成功返回 format(exp(L/T), '.17g') + '\\n'。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    (vocab, W, b, Wq, Wk, Wv, Wo, heads,
     Why, by, h0, c0) = _load_perplexity_lstm_mha_model(model_path)
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

    mha = MHA(H, heads)
    mha.Wq = [list(row) for row in Wq]
    mha.Wk = [list(row) for row in Wk]
    mha.Wv = [list(row) for row in Wv]
    mha.Wo = [list(row) for row in Wo]

    # 记忆序列 [h0, h_0, ..., h_{t-1}]；h0 保留模型原值（F 允许 int），
    # forward_cross 会先复制再投影；各步 h 本就是 float。
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
        u = _mha_cross_context(h, M, mha)

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


def _perplexity_gru_attn(model_path, corpus_path, window_text):
    """GRU 加注意力上下文的困惑度，返回待写出的字符串。

    MODEL、CORPUS 完全沿用 perplexity-gru version 3 的七键顺序、F、形状、
    严格 UTF-8、词表及语料至少 2 码点契约；WINDOW 的词法与任意位数安全截取
    完全沿用 perplexity-lstm-attn（整串匹配 [1-9][0-9]*，不转 int），否则
    抛 ValueError。

    置 h=h0、L=0.0、T=len(CORPUS)-1、memory=[h0]，t 升序：以当前字符的
    V 长 one-hot 为 x，调用装入 W、b 的 GRUCell.forward(x, h) 取首项更新
    h；M 取 memory 末尾至多 WINDOW 项（顺序从旧到新），以
    attention([h], M, M, None) 返回首项 ctx，按 i 升序令
    u[i] = h[i] + ctx[0][i]。logit 仅以 u 替代 perplexity-gru 中的 h，
    其余 Why/by 仿射、稳定 log-sum-exp 及下一字符负对数似然的下标与累加
    顺序均与 perplexity-gru 相同；随后向 memory 追加 h 的 float 副本。任一
    运算非有限（含最终 exp(L/T) 溢出）均抛 ValueError。成功返回
    format(exp(L/T), '.17g') + '\\n'。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    cell = GRUCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    # 记忆序列 [h0, h_0, ..., h_{t-1}]；h0 保留模型原值（F 允许 int），
    # attention 按原值“先乘后加”；各步 h 本就是 float。attention 不修改其
    # 输入，故直接共享行即可。
    memory = [h0]
    h = list(h0)
    L = 0.0
    T = len(ids) - 1
    for t in range(T):
        y = ids[t + 1]

        # 当前字符的 V 长 one-hot 输入，推进 GRU 隐状态。
        x = [0.0] * V
        x[ids[t]] = 1.0

        h, _cache = cell.forward(x, h)

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


def _eval_windows(model_path, corpus_path, windows_text):
    """对一组窗口分别求带注意力上下文的困惑度，返回待写出的字符串。

    MODEL、CORPUS 完全沿用 perplexity-lstm-attn 的读取、形状、F、严格
    UTF-8、词表及语料至少 2 码点契约。WINDOWS 为逗号分隔的非空窗口
    列表：每项须整串匹配 [1-9][0-9]*（任意位数均合法，不转 int），
    空项非法，超长项的安全截取沿用 perplexity-lstm-attn，重复项按序
    保留，否则抛 ValueError。

    每项独立计算：置 h=h0、c=c0、memory=[h0]，t 升序的单步计算与
    perplexity-lstm-attn 完全相同；令 T=语料码点数-1，L 为按 t 升序
    累加的总负对数似然。任一运算非有限（含 exp(L/T) 溢出）均抛
    ValueError。

    stdout 依次为每项一个 JSON 行，键序恰为
    window,steps,total_logprob,perplexity，值依次为原窗口文本 str、
    T（int）、format(-L,'.17g')、format(exp(L/T),'.17g') 字符串。
    每行恰由 json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n' 生成，各行直接拼接，末行保留 LF。全部行
    先完整构造再返回，故失败时不产生任何部分输出，不写文件。
    """
    windows = windows_text.split(",")
    for window_text in windows:
        if not _WINDOW_RE.match(window_text):
            raise ValueError("each WINDOWS item must match [1-9][0-9]*")

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

    lines = []
    for window_text in windows:
        # 每项均重置状态；单步计算严格复用 perplexity-lstm-attn。
        memory = [h0]
        h = list(h0)
        c = list(c0)
        L = 0.0
        for t in range(T):
            y = ids[t + 1]

            x = [0.0] * V
            x[ids[t]] = 1.0

            h, c = cell.forward(x, h, c)[:2]

            M = _window_tail(memory, window_text)
            u = _attn_context(h, M)

            z = _output_logits(Why, by, u)

            m = max(z)
            if not math.isfinite(m):
                raise ValueError("logit maximum is non-finite")
            d = 0.0
            for k in range(V):
                d += math.exp(z[k] - m)
                if not math.isfinite(d):
                    raise ValueError(
                        "softmax denominator accumulated non-finitely")

            step = m + math.log(d) - z[y]
            if not math.isfinite(step):
                raise ValueError("cross-entropy step is non-finite")
            L += step
            if not math.isfinite(L):
                raise ValueError(
                    "total cross-entropy accumulated non-finitely")

            memory.append([float(v) for v in h])

        perplexity = math.exp(L / T)
        if not math.isfinite(perplexity):
            raise ValueError("perplexity is non-finite")
        obj = {
            "window": window_text,
            "steps": T,
            "total_logprob": format(-L, ".17g"),
            "perplexity": format(perplexity, ".17g"),
        }
        lines.append(json.dumps(obj, ensure_ascii=True,
                                separators=(",", ":"), allow_nan=False))
    return "".join(line + "\n" for line in lines)


def _window_sensitivity_lstm_attn(model_path, corpus_path, short_text,
                                  long_text):
    """短、长两窗逐步负对数似然之差（长窗减短窗），返回待写出的字符串。

    MODEL、CORPUS 完全沿用 perplexity-lstm-attn 的读取、形状、F、严格
    UTF-8、词表及语料至少 2 码点契约。SHORT、LONG 各须整串匹配
    [1-9][0-9]*（任意位数均合法，不转 int），且数学值 SHORT<LONG；
    大小仅按十进制位数及同长字典序比较，不把无界整数文本转为 int，
    否则抛 ValueError。窗口截取沿用 perplexity-lstm-attn 的
    _window_tail。

    短、长两窗均从 h=h0、c=c0、memory=[h0] 独立重置，两条轨道的单步
    推进、注意力、logit 与稳定 log-sum-exp 与 perplexity-lstm-attn
    完全相同；令 T=语料码点数-1，第 t 步短、长窗的单步负对数似然依次
    为 s、l，d=s-l，D 从 0.0 按 t 升序累加 d。任一计算非有限（含 d
    或 D）均抛 ValueError。

    stdout 先写 T 个 JSON 项行，每行键序恰为
    t,target,short_nll,long_nll,delta，值依次为 int t、下一单码点 str、
    format(s,'.17g')、format(l,'.17g')、format(d,'.17g') 字符串；再写
    唯一汇总行，键序恰为 steps,total_delta，值依次为 int T、
    format(D,'.17g') 字符串。每行恰由 json.dumps(obj,ensure_ascii=True,
    separators=(',',':'),allow_nan=False)+'\\n' 生成，各行直接拼接，
    末行保留 LF。全部行先完整构造再返回，故失败时不产生任何部分输出，
    不写文件。
    """
    if not _WINDOW_RE.match(short_text):
        raise ValueError("SHORT must match [1-9][0-9]*")
    if not _WINDOW_RE.match(long_text):
        raise ValueError("LONG must match [1-9][0-9]*")
    # 位数多者数值大；同位数则字典序与数值序一致（两者均无前导零）。
    if not (len(short_text) < len(long_text)
            or (len(short_text) == len(long_text)
                and short_text < long_text)):
        raise ValueError("SHORT must be numerically smaller than LONG")

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

    def _step_nll(window_text, h, c, memory, t):
        """perplexity-lstm-attn 单步：推进状态并返回 (nll, h, c, memory)。"""
        y = ids[t + 1]

        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        z = _output_logits(Why, by, u)

        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        nll = m + math.log(d) - z[y]
        if not math.isfinite(nll):
            raise ValueError("cross-entropy step is non-finite")

        memory.append([float(v) for v in h])
        return nll, h, c, memory

    # 两窗独立重置：各自从 h0、c0、memory=[h0] 出发。
    memory_s = [h0]
    h_s = list(h0)
    c_s = list(c0)
    memory_l = [h0]
    h_l = list(h0)
    c_l = list(c0)
    D = 0.0
    T = len(ids) - 1
    lines = []
    for t in range(T):
        s, h_s, c_s, memory_s = _step_nll(
            short_text, h_s, c_s, memory_s, t)
        l, h_l, c_l, memory_l = _step_nll(
            long_text, h_l, c_l, memory_l, t)

        delta = s - l
        if not math.isfinite(delta):
            raise ValueError("per-step delta is non-finite")
        D += delta
        if not math.isfinite(D):
            raise ValueError("total delta accumulated non-finitely")

        lines.append(json.dumps(
            {
                "t": t,
                "target": corpus[t + 1],
                "short_nll": format(s, ".17g"),
                "long_nll": format(l, ".17g"),
                "delta": format(delta, ".17g"),
            },
            ensure_ascii=True, separators=(",", ":"), allow_nan=False))

    lines.append(json.dumps(
        {
            "steps": T,
            "total_delta": format(D, ".17g"),
        },
        ensure_ascii=True, separators=(",", ":"), allow_nan=False))
    return "".join(line + "\n" for line in lines)


def _compare_lstm_attn(model_a_path, model_b_path, corpus_path,
                       window_text):
    """两模型在同一语料、同一窗口上的逐步负对数似然之差（B 减 A）。

    MODEL_A、MODEL_B 各自沿用 perplexity-lstm-attn 的 version 2 八键、
    F、形状与严格 UTF-8 契约；两者 vocab 须逐项同序相等，H 可不同。
    CORPUS 沿用 perplexity-lstm-attn 的严格 UTF-8、词表（以共同 vocab
    为准）及至少 2 码点契约；WINDOW 沿用其 [1-9][0-9]* 词法及任意位数
    安全截取。

    两条轨道各自从自身的 h0、c0、memory=[h0] 重置，单步的状态推进、
    注意力、logit 与稳定 log-sum-exp 与 perplexity-lstm-attn 完全相同；
    令 T=语料码点数-1，第 t 步 A、B 的单步负对数似然依次为 a、b，
    d=b-a，D 从 0.0 按 t 升序累加 d。任一计算非有限（含 d 或 D）均
    抛 ValueError。

    stdout 先写 T 个 JSON 项行，每行键序恰为
    t,target,a_nll,b_nll,delta，值依次为 int t、下一单码点 str、
    format(a,'.17g')、format(b,'.17g')、format(d,'.17g') 字符串；再写
    唯一汇总行，键序恰为 steps,total_delta，值依次为 int T、
    format(D,'.17g') 字符串。每行恰由 json.dumps(obj,ensure_ascii=True,
    separators=(',',':'),allow_nan=False)+'\\n' 生成，各行直接拼接，
    末行保留 LF。全部行先完整构造再返回，故失败时不产生任何部分输出，
    不写文件。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    (vocab_a, W_a, b_a, Why_a, by_a, h0_a,
     c0_a) = _load_perplexity_lstm_model(model_a_path)
    (vocab_b, W_b, b_b, Why_b, by_b, h0_b,
     c0_b) = _load_perplexity_lstm_model(model_b_path)
    if vocab_a != vocab_b:
        raise ValueError("the two models must have identical vocabs in the "
                         "same order")
    vocab = vocab_a
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

    cell_a = LSTMCell(V, len(h0_a))
    cell_a.W = [list(row) for row in W_a]
    cell_a.b = list(b_a)
    cell_b = LSTMCell(V, len(h0_b))
    cell_b.W = [list(row) for row in W_b]
    cell_b.b = list(b_b)

    def _step_nll(cell, Why, by, h, c, memory, t):
        """perplexity-lstm-attn 单步：推进状态并返回 (nll, h, c, memory)。"""
        y = ids[t + 1]

        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        z = _output_logits(Why, by, u)

        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        nll = m + math.log(d) - z[y]
        if not math.isfinite(nll):
            raise ValueError("cross-entropy step is non-finite")

        memory.append([float(v) for v in h])
        return nll, h, c, memory

    # 两轨各自从自身的 h0、c0、memory=[h0] 独立重置。
    memory_a = [h0_a]
    h_a = list(h0_a)
    c_a = list(c0_a)
    memory_b = [h0_b]
    h_b = list(h0_b)
    c_b = list(c0_b)
    D = 0.0
    T = len(ids) - 1
    lines = []
    for t in range(T):
        nll_a, h_a, c_a, memory_a = _step_nll(
            cell_a, Why_a, by_a, h_a, c_a, memory_a, t)
        nll_b, h_b, c_b, memory_b = _step_nll(
            cell_b, Why_b, by_b, h_b, c_b, memory_b, t)

        delta = nll_b - nll_a
        if not math.isfinite(delta):
            raise ValueError("per-step delta is non-finite")
        D += delta
        if not math.isfinite(D):
            raise ValueError("total delta accumulated non-finitely")

        lines.append(json.dumps(
            {
                "t": t,
                "target": corpus[t + 1],
                "a_nll": format(nll_a, ".17g"),
                "b_nll": format(nll_b, ".17g"),
                "delta": format(delta, ".17g"),
            },
            ensure_ascii=True, separators=(",", ":"), allow_nan=False))

    lines.append(json.dumps(
        {
            "steps": T,
            "total_delta": format(D, ".17g"),
        },
        ensure_ascii=True, separators=(",", ":"), allow_nan=False))
    return "".join(line + "\n" for line in lines)


def _compare_suite(model_a_path, model_b_path, list_path):
    """两模型在一组 (语料, 窗口) 上的逐步负对数似然之差（B 减 A）汇总。

    MODEL_A、MODEL_B 的读取、vocab 逐项同序相等（H 可不同）契约，以及每
    项的两轨独立状态重置与逐步 NLL 计算，完全沿用 compare-lstm-attn。

    LIST 为严格 UTF-8 的 JSON 非空数组，每项恰为两个 str 组成的数组
    [corpus, window]：corpus 须为非空相对路径，按 LIST 文件的父目录解析；
    window 须整串匹配 [1-9][0-9]*（任意位数均合法，不转 int，安全截取沿
    用 perplexity-lstm-attn）；重复项按序保留，否则抛 ValueError。

    每项令 D=0.0，按 t 升序累加 b_nll-a_nll（任一计算非有限即抛
    ValueError）；D 为正、负、零时该项 winner 依次为 "A"、"B"、"tie"。
    G=0.0 并按清单项序累加各项 D，累加结果非有限即抛 ValueError。

    stdout 恰为单个 JSON 对象加 LF，顶层键序恰为 items,summary。items 按
    清单序，每项恰为 [corpus,window,steps,format(D,'.17g'),winner]：
    corpus、window 为清单原文 str，steps 为 int T，winner 为上述 str。
    summary 恰为 [groups,a_wins,b_wins,ties,format(G,'.17g'),winner]：
    前四项为 int（groups 为清单项数，a_wins、b_wins、ties 为对应 winner
    的项数），末项 winner 按 G 的正、负、零同样取 "A"、"B"、"tie"。
    序列化恰用 json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'。整个对象先完整构造再返回，故失败时不产生
    任何部分输出，不写文件。
    """
    with open(list_path, "rb") as f:
        suite = json.loads(f.read().decode("utf-8"))
    if not isinstance(suite, list) or not suite:
        raise ValueError("LIST must be a non-empty JSON array")
    entries = []
    for item in suite:
        if (not isinstance(item, list) or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)):
            raise ValueError("each LIST item must be a [corpus, window] "
                             "pair of strings")
        corpus_text, window_text = item
        if not corpus_text or os.path.isabs(corpus_text):
            raise ValueError("each corpus must be a non-empty relative path")
        if not _WINDOW_RE.match(window_text):
            raise ValueError("each window must match [1-9][0-9]*")
        entries.append((corpus_text, window_text))

    (vocab_a, W_a, b_a, Why_a, by_a, h0_a,
     c0_a) = _load_perplexity_lstm_model(model_a_path)
    (vocab_b, W_b, b_b, Why_b, by_b, h0_b,
     c0_b) = _load_perplexity_lstm_model(model_b_path)
    if vocab_a != vocab_b:
        raise ValueError("the two models must have identical vocabs in the "
                         "same order")
    vocab = vocab_a
    V = len(vocab)
    table = {ch: i for i, ch in enumerate(vocab)}

    cell_a = LSTMCell(V, len(h0_a))
    cell_a.W = [list(row) for row in W_a]
    cell_a.b = list(b_a)
    cell_b = LSTMCell(V, len(h0_b))
    cell_b.W = [list(row) for row in W_b]
    cell_b.b = list(b_b)

    base_dir = os.path.dirname(list_path)

    def _step_nll(cell, Why, by, h, c, memory, ids, t, window_text):
        """perplexity-lstm-attn 单步：推进状态并返回 (nll, h, c, memory)。"""
        y = ids[t + 1]

        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        z = _output_logits(Why, by, u)

        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        nll = m + math.log(d) - z[y]
        if not math.isfinite(nll):
            raise ValueError("cross-entropy step is non-finite")

        memory.append([float(v) for v in h])
        return nll, h, c, memory

    items = []
    a_wins = 0
    b_wins = 0
    ties = 0
    G = 0.0
    for corpus_text, window_text in entries:
        corpus_path = os.path.join(base_dir, corpus_text)
        with open(corpus_path, "rb") as f:
            corpus = f.read().decode("utf-8")
        if len(corpus) < 2:
            raise ValueError("corpus must contain at least 2 codepoints")

        ids = [0] * len(corpus)
        for t, ch in enumerate(corpus):
            ix = table.get(ch)
            if ix is None:
                raise ValueError(
                    "corpus contains an out-of-vocab character")
            ids[t] = ix

        T = len(ids) - 1

        # 两轨各自从自身的 h0、c0、memory=[h0] 独立重置。
        memory_a = [h0_a]
        h_a = list(h0_a)
        c_a = list(c0_a)
        memory_b = [h0_b]
        h_b = list(h0_b)
        c_b = list(c0_b)
        D = 0.0
        for t in range(T):
            nll_a, h_a, c_a, memory_a = _step_nll(
                cell_a, Why_a, by_a, h_a, c_a, memory_a, ids, t,
                window_text)
            nll_b, h_b, c_b, memory_b = _step_nll(
                cell_b, Why_b, by_b, h_b, c_b, memory_b, ids, t,
                window_text)

            delta = nll_b - nll_a
            if not math.isfinite(delta):
                raise ValueError("per-step delta is non-finite")
            D += delta
            if not math.isfinite(D):
                raise ValueError("total delta accumulated non-finitely")

        if D > 0:
            winner = "A"
            a_wins += 1
        elif D < 0:
            winner = "B"
            b_wins += 1
        else:
            winner = "tie"
            ties += 1

        G += D
        if not math.isfinite(G):
            raise ValueError("grand total accumulated non-finitely")

        items.append([corpus_text, window_text, T, format(D, ".17g"),
                      winner])

    if G > 0:
        suite_winner = "A"
    elif G < 0:
        suite_winner = "B"
    else:
        suite_winner = "tie"

    obj = {
        "items": items,
        "summary": [len(entries), a_wins, b_wins, ties,
                    format(G, ".17g"), suite_winner],
    }
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                       allow_nan=False)
            + "\n")


def _load_rank_inputs(models_path, list_path):
    """加载并校验三条 rank-window 命令共用的 MODELS、LIST、模型与语料。

    MODELS 为严格 UTF-8 的 JSON 数组，至少 2 项；每项须为非空相对路径
    str（绝对路径非法），按 MODELS 文件的父目录解析；重复项按序保留。
    LIST 为严格 UTF-8 的 JSON 非空数组，每项恰为二 str 数组
    [corpus, window]：corpus 为非空相对路径，按 LIST 文件的父目录解
    析；window 整串匹配 [1-9][0-9]*（任意位数安全，全程不转 int）；重
    复项按序保留。各模型的读取沿用 _load_perplexity_lstm_model 的
    version 2 八键、F、形状与严格 UTF-8 契约；所有模型 vocab 须逐项同
    序相等，H 可不同。每个语料按 LIST 父目录以严格 UTF-8 读取，至少含
    2 个码点且全部字符在公共 vocab 内，并按清单原序转成 id 列表。

    返回 (model_paths, entries, loaded, entry_ids)：model_paths 为
    MODELS 原文相对路径 str 列表；entries 为 LIST 原文 (corpus, window)
    二元组列表；loaded 为每模型 (cell, Why, by, h0, c0) 五元组列表，
    cell.W、cell.b 为模型数组的逐行/逐项拷贝；entry_ids 为与 entries
    同序的语料 id 列表。MODELS、LIST、模型或语料文件缺失、不可读抛
    OSError；UTF-8、JSON、结构、路径词法、WINDOW 词法、模型形状、词表
    及语料内容等其余非法一律抛 ValueError。
    """
    with open(models_path, "rb") as f:
        model_paths = json.loads(f.read().decode("utf-8"))
    if not isinstance(model_paths, list) or len(model_paths) < 2:
        raise ValueError("MODELS must be a JSON array with at least 2 items")
    for path_text in model_paths:
        if not isinstance(path_text, str) or not path_text \
                or os.path.isabs(path_text):
            raise ValueError("each model must be a non-empty relative path")

    with open(list_path, "rb") as f:
        suite = json.loads(f.read().decode("utf-8"))
    if not isinstance(suite, list) or not suite:
        raise ValueError("LIST must be a non-empty JSON array")
    entries = []
    for item in suite:
        if (not isinstance(item, list) or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)):
            raise ValueError("each LIST item must be a [corpus, window] "
                             "pair of strings")
        corpus_text, window_text = item
        if not corpus_text or os.path.isabs(corpus_text):
            raise ValueError("each corpus must be a non-empty relative path")
        if not _WINDOW_RE.match(window_text):
            raise ValueError("each window must match [1-9][0-9]*")
        entries.append((corpus_text, window_text))

    models_base = os.path.dirname(models_path)
    loaded = []
    vocab = None
    for path_text in model_paths:
        resolved = os.path.join(models_base, path_text)
        (one_vocab, W, b, Why, by, h0,
         c0) = _load_perplexity_lstm_model(resolved)
        if vocab is None:
            vocab = one_vocab
        elif one_vocab != vocab:
            raise ValueError("all models must have identical vocabs in the "
                             "same order")
        cell = LSTMCell(len(vocab), len(h0))
        cell.W = [list(row) for row in W]
        cell.b = list(b)
        loaded.append((cell, Why, by, h0, c0))

    table = {ch: i for i, ch in enumerate(vocab)}
    list_base = os.path.dirname(list_path)
    entry_ids = []
    for corpus_text, _window_text in entries:
        corpus_path = os.path.join(list_base, corpus_text)
        with open(corpus_path, "rb") as f:
            corpus = f.read().decode("utf-8")
        if len(corpus) < 2:
            raise ValueError("corpus must contain at least 2 codepoints")

        ids = [0] * len(corpus)
        for t, ch in enumerate(corpus):
            ix = table.get(ch)
            if ix is None:
                raise ValueError(
                    "corpus contains an out-of-vocab character")
            ids[t] = ix
        entry_ids.append(ids)

    return model_paths, entries, loaded, entry_ids


def _total_rank_nll(cell, Why, by, h0, c0, ids, window_text):
    """单个模型在单条语料上以给定窗口计算总 NLL，返回 float。

    每次调用均从传入的 h0、c0 与 memory=[h0] 独立重置（不保留任何跨调
    用状态），t 从 0 到 len(ids)-2 升序推进：one-hot x、LSTM 单步、
    _window_tail 截断的记忆、_attn_context 注意力、_output_logits
    logit，以及 k 升序的稳定 log-sum-exp 与 NLL；下标与累加顺序完全沿
    用各 rank 命令重构前的逐步算法，每步 nll 即时累加进从 0.0 起的总
    量。window_text 不匹配 [1-9][0-9]*、ids 非合法词表下标，或任一中
    间量、总量非有限，均抛 ValueError。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("window must match [1-9][0-9]*")
    V = len(by)
    steps = len(ids) - 1
    if steps < 1:
        raise ValueError("ids must contain at least 2 entries")

    # 每次均从自身 h0、c0、memory=[h0] 独立重置。
    memory = [h0]
    h = list(h0)
    c = list(c0)
    total = 0.0
    for t in range(steps):
        xt = ids[t]
        y = ids[t + 1]
        if (type(xt) is not int or type(y) is not int
                or not 0 <= xt < V or not 0 <= y < V):
            raise ValueError("ids must be in-vocab integer indices")

        x = [0.0] * V
        x[xt] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        z = _output_logits(Why, by, u)

        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        nll = m + math.log(d) - z[y]
        if not math.isfinite(nll):
            raise ValueError("cross-entropy step is non-finite")

        memory.append([float(v) for v in h])
        total += nll
        if not math.isfinite(total):
            raise ValueError("item total NLL accumulated non-finitely")

    return total


def _rank_suite(models_path, list_path):
    """多个 LSTM-attn 模型在同一组 (语料, 窗口) 上按总 NLL 排名。

    MODELS 为严格 UTF-8 的 JSON 数组，至少 2 项；每项须为非空相对路径
    str（绝对路径非法），按 MODELS 文件的父目录解析；重复项按序保留，
    否则抛 ValueError。LIST 的结构、语料相对路径（按 LIST 父目录解析）、
    WINDOW 的 [1-9][0-9]* 词法及重复项契约完全沿用 compare-suite。

    各模型的读取沿用 compare-lstm-attn 的 version 2 八键、F、形状与严格
    UTF-8 契约；所有模型 vocab 须逐项同序相等，H 可不同。每个模型在每
    个清单项均从自身 h0、c0、memory=[h0] 重置，单步状态推进、注意力、
    logit 与稳定 log-sum-exp 完全沿用 compare-lstm-attn。每项 NLL 总和
    L 从 0.0 按 t 升序累加（任一计算非有限即抛 ValueError）；模型总分
    S 从 0.0 按 LIST 序累加各项 L，累加结果非有限即抛 ValueError。

    按 (S, MODELS 原下标) 升序排名，S 以 float 精确比较。stdout 恰为
    单个 JSON 对象加 LF，顶层唯一键 ranking；其值为排名后的数组，每项
    恰为 [path, total_nll]，依次为 MODELS 原文 str 与
    format(S,'.17g') 字符串。序列化恰用 json.dumps(obj,ensure_ascii=
    True,separators=(',',':'),allow_nan=False)+'\\n'。整个对象先完整
    构造再返回，故失败时不产生任何部分输出，不写文件。
    """
    with open(models_path, "rb") as f:
        model_paths = json.loads(f.read().decode("utf-8"))
    if not isinstance(model_paths, list) or len(model_paths) < 2:
        raise ValueError("MODELS must be a JSON array with at least 2 items")
    for path_text in model_paths:
        if not isinstance(path_text, str) or not path_text \
                or os.path.isabs(path_text):
            raise ValueError("each model must be a non-empty relative path")

    with open(list_path, "rb") as f:
        suite = json.loads(f.read().decode("utf-8"))
    if not isinstance(suite, list) or not suite:
        raise ValueError("LIST must be a non-empty JSON array")
    entries = []
    for item in suite:
        if (not isinstance(item, list) or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)):
            raise ValueError("each LIST item must be a [corpus, window] "
                             "pair of strings")
        corpus_text, window_text = item
        if not corpus_text or os.path.isabs(corpus_text):
            raise ValueError("each corpus must be a non-empty relative path")
        if not _WINDOW_RE.match(window_text):
            raise ValueError("each window must match [1-9][0-9]*")
        entries.append((corpus_text, window_text))

    models_base = os.path.dirname(models_path)
    loaded = []
    vocab = None
    for path_text in model_paths:
        resolved = os.path.join(models_base, path_text)
        (one_vocab, W, b, Why, by, h0,
         c0) = _load_perplexity_lstm_model(resolved)
        if vocab is None:
            vocab = one_vocab
        elif one_vocab != vocab:
            raise ValueError("all models must have identical vocabs in the "
                             "same order")
        V = len(vocab)
        cell = LSTMCell(V, len(h0))
        cell.W = [list(row) for row in W]
        cell.b = list(b)
        loaded.append((cell, Why, by, h0, c0))
    V = len(vocab)
    table = {ch: i for i, ch in enumerate(vocab)}

    base_dir = os.path.dirname(list_path)

    def _step_nll(cell, Why, by, h, c, memory, ids, t, window_text):
        """perplexity-lstm-attn 单步：推进状态并返回 (nll, h, c, memory)。"""
        y = ids[t + 1]

        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        z = _output_logits(Why, by, u)

        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        nll = m + math.log(d) - z[y]
        if not math.isfinite(nll):
            raise ValueError("cross-entropy step is non-finite")

        memory.append([float(v) for v in h])
        return nll, h, c, memory

    # 每个模型的总分 S 从 0.0 起，按 LIST 序累加各项 L。
    scores = [0.0] * len(loaded)
    for corpus_text, window_text in entries:
        corpus_path = os.path.join(base_dir, corpus_text)
        with open(corpus_path, "rb") as f:
            corpus = f.read().decode("utf-8")
        if len(corpus) < 2:
            raise ValueError("corpus must contain at least 2 codepoints")

        ids = [0] * len(corpus)
        for t, ch in enumerate(corpus):
            ix = table.get(ch)
            if ix is None:
                raise ValueError(
                    "corpus contains an out-of-vocab character")
            ids[t] = ix

        T = len(ids) - 1

        for mi, (cell, Why, by, h0_m, c0_m) in enumerate(loaded):
            # 每项均从自身 h0、c0、memory=[h0] 独立重置。
            memory = [h0_m]
            h = list(h0_m)
            c = list(c0_m)
            L = 0.0
            for t in range(T):
                nll, h, c, memory = _step_nll(
                    cell, Why, by, h, c, memory, ids, t, window_text)
                L += nll
                if not math.isfinite(L):
                    raise ValueError(
                        "item total NLL accumulated non-finitely")

            scores[mi] += L
            if not math.isfinite(scores[mi]):
                raise ValueError("model score accumulated non-finitely")

    # 按 (S, MODELS 原下标) 升序排名；S 以 float 精确比较，下标决胜保序。
    order = sorted(range(len(loaded)),
                   key=lambda mi: (scores[mi], mi))
    ranking = [[model_paths[mi], format(scores[mi], ".17g")]
               for mi in order]

    obj = {"ranking": ranking}
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                       allow_nan=False)
            + "\n")


def _rank_suite_details(models_path, list_path):
    """rank-suite 的逐项明细版，ranking 与同参 rank-suite 逐字节一致。

    MODELS、LIST 的读取、校验、相对路径解析与重复项保序完全沿用
    rank-suite（MODELS 为至少 2 项的非空相对路径 str 数组；LIST 为非空
    [corpus, window] 数组；所有模型 vocab 须逐项同序相等，H 可不同）。

    按模型原下标 mi、清单原下标 gi 升序遍历 (mi, gi)；每个模型在每项均
    从自身 h0、c0、memory=[h0] 重置，单步状态推进、注意力、logit 与稳定
    log-sum-exp 完全沿用 rank-suite。每项 NLL 总和 L 从 0.0 按 t 升序累
    加；模型总分 S 从 0.0 按 gi 升序累加未格式化的 L，任一计算非有限即
    抛 ValueError。整个对象先完整构造再返回，失败时不产生任何部分输
    出，不写文件。

    stdout 恰为单个 JSON 对象加 LF，顶层键序 items,ranking。items 按
    (mi, gi) 展平，每项恰为
    [model,corpus,window,steps,item_nll]：前三项依次为 MODELS 原文 str、
    LIST 原文 corpus str、LIST 原文 window str；steps 为语料码点数减 1
    的 int；item_nll 为 format(L,'.17g') 字符串。ranking 按
    (S, mi) 升序（S 以 float 精确比较，下标决胜保序），每项恰为
    [model, format(S,'.17g')]，与同参 rank-suite 的 ranking 相等。序列
    化恰用 json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'。
    """
    with open(models_path, "rb") as f:
        model_paths = json.loads(f.read().decode("utf-8"))
    if not isinstance(model_paths, list) or len(model_paths) < 2:
        raise ValueError("MODELS must be a JSON array with at least 2 items")
    for path_text in model_paths:
        if not isinstance(path_text, str) or not path_text \
                or os.path.isabs(path_text):
            raise ValueError("each model must be a non-empty relative path")

    with open(list_path, "rb") as f:
        suite = json.loads(f.read().decode("utf-8"))
    if not isinstance(suite, list) or not suite:
        raise ValueError("LIST must be a non-empty JSON array")
    entries = []
    for item in suite:
        if (not isinstance(item, list) or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)):
            raise ValueError("each LIST item must be a [corpus, window] "
                             "pair of strings")
        corpus_text, window_text = item
        if not corpus_text or os.path.isabs(corpus_text):
            raise ValueError("each corpus must be a non-empty relative path")
        if not _WINDOW_RE.match(window_text):
            raise ValueError("each window must match [1-9][0-9]*")
        entries.append((corpus_text, window_text))

    models_base = os.path.dirname(models_path)
    loaded = []
    vocab = None
    for path_text in model_paths:
        resolved = os.path.join(models_base, path_text)
        (one_vocab, W, b, Why, by, h0,
         c0) = _load_perplexity_lstm_model(resolved)
        if vocab is None:
            vocab = one_vocab
        elif one_vocab != vocab:
            raise ValueError("all models must have identical vocabs in the "
                             "same order")
        V = len(vocab)
        cell = LSTMCell(V, len(h0))
        cell.W = [list(row) for row in W]
        cell.b = list(b)
        loaded.append((cell, Why, by, h0, c0))
    V = len(vocab)
    table = {ch: i for i, ch in enumerate(vocab)}

    base_dir = os.path.dirname(list_path)

    # 每个清单项的语料只读一次并转成 id 序列；计算结果与重复读取逐位相同。
    entry_ids = []
    for corpus_text, _window_text in entries:
        corpus_path = os.path.join(base_dir, corpus_text)
        with open(corpus_path, "rb") as f:
            corpus = f.read().decode("utf-8")
        if len(corpus) < 2:
            raise ValueError("corpus must contain at least 2 codepoints")

        ids = [0] * len(corpus)
        for t, ch in enumerate(corpus):
            ix = table.get(ch)
            if ix is None:
                raise ValueError(
                    "corpus contains an out-of-vocab character")
            ids[t] = ix
        entry_ids.append(ids)

    def _step_nll(cell, Why, by, h, c, memory, ids, t, window_text):
        """perplexity-lstm-attn 单步：推进状态并返回 (nll, h, c, memory)。"""
        y = ids[t + 1]

        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        z = _output_logits(Why, by, u)

        m = max(z)
        if not math.isfinite(m):
            raise ValueError("logit maximum is non-finite")
        d = 0.0
        for k in range(V):
            d += math.exp(z[k] - m)
            if not math.isfinite(d):
                raise ValueError(
                    "softmax denominator accumulated non-finitely")

        nll = m + math.log(d) - z[y]
        if not math.isfinite(nll):
            raise ValueError("cross-entropy step is non-finite")

        memory.append([float(v) for v in h])
        return nll, h, c, memory

    items = []
    # 每个模型的总分 S 从 0.0 起，按 gi 升序累加未格式化的各项 L。
    scores = [0.0] * len(loaded)
    for mi, (cell, Why, by, h0_m, c0_m) in enumerate(loaded):
        for gi, (corpus_text, window_text) in enumerate(entries):
            ids = entry_ids[gi]
            T = len(ids) - 1

            # 每项均从自身 h0、c0、memory=[h0] 独立重置。
            memory = [h0_m]
            h = list(h0_m)
            c = list(c0_m)
            L = 0.0
            for t in range(T):
                nll, h, c, memory = _step_nll(
                    cell, Why, by, h, c, memory, ids, t, window_text)
                L += nll
                if not math.isfinite(L):
                    raise ValueError(
                        "item total NLL accumulated non-finitely")

            scores[mi] += L
            if not math.isfinite(scores[mi]):
                raise ValueError("model score accumulated non-finitely")

            items.append([model_paths[mi], corpus_text, window_text, T,
                          format(L, ".17g")])

    # 按 (S, MODELS 原下标) 升序排名；S 以 float 精确比较，下标决胜保序。
    order = sorted(range(len(loaded)),
                   key=lambda mi: (scores[mi], mi))
    ranking = [[model_paths[mi], format(scores[mi], ".17g")]
               for mi in order]

    obj = {"items": items, "ranking": ranking}
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                       allow_nan=False)
            + "\n")


def _rank_window_sensitivity(models_path, list_path, base_text):
    """rank-suite-details 的窗口敏感度版，比较各项 window 与基准 BASE。

    MODELS、LIST 的读取、校验、相对路径解析、重复项保序、模型加载与语
    料 id 化完全沿用 _load_rank_inputs（MODELS 为至少 2 项的非空相对路
    径 str 数组；LIST 为非空 [corpus, window] 数组；所有模型 vocab 须逐
    项同序相等，H 可不同）。BASE 整串匹配 [1-9][0-9]*，任意位数均合
    法，全程不转为 int。

    按模型原下标 mi、清单原下标 gi 升序遍历 (mi, gi)；每项的 window 轨
    L 与 BASE 轨 B 均调用 _total_rank_nll 独立重置（h0、c0、
    memory=[h0]）并按 t 升序累加，两轨之间不共享任何状态，同一模型在
    不同项/不同命令调用之间亦不缓存状态。令 delta=L-B；每个模型的敏感
    度 A 从 0.0 按 gi 升序累加 abs(delta)，任一计算非有限即抛
    ValueError。整个对象先完整构造再返回，失败时不产生任何部分输出，
    不写文件。

    stdout 恰为单个 JSON 对象加 LF，顶层键序 items,ranking。items 按
    (mi, gi) 展平，每项恰为
    [model,corpus,window,steps,item_nll,base_nll,delta]：前三项依次为
    MODELS 原文 str、LIST 原文 corpus str、LIST 原文 window str；steps
    为语料码点数减 1 的 int；后三项依次为 format(L,'.17g')、
    format(B,'.17g')、format(delta,'.17g') 字符串（.17g 保留 -0.0）。
    ranking 按 (-A, mi) 升序（A 以 float 精确比较，下标决胜保序），每
    项恰为 [model, format(A,'.17g')]。序列化恰用 json.dumps(obj,
    ensure_ascii=True,separators=(',',':'),allow_nan=False)+'\\n'。
    """
    if not _WINDOW_RE.match(base_text):
        raise ValueError("BASE must match [1-9][0-9]*")

    model_paths, entries, loaded, entry_ids = _load_rank_inputs(
        models_path, list_path)

    items = []
    # 每个模型的敏感度 A 从 0.0 起，按 gi 升序累加各项 abs(delta)。
    sensitivity = [0.0] * len(loaded)
    for mi, (cell, Why, by, h0_m, c0_m) in enumerate(loaded):
        for gi, (corpus_text, window_text) in enumerate(entries):
            ids = entry_ids[gi]
            T = len(ids) - 1

            # 两轨均通过 _total_rank_nll 从自身 h0、c0、memory=[h0]
            # 独立重置，无跨调用缓存。
            L = _total_rank_nll(cell, Why, by, h0_m, c0_m, ids, window_text)
            B = _total_rank_nll(cell, Why, by, h0_m, c0_m, ids, base_text)

            delta = L - B
            if not math.isfinite(delta):
                raise ValueError("window delta is non-finite")

            sensitivity[mi] += abs(delta)
            if not math.isfinite(sensitivity[mi]):
                raise ValueError(
                    "model sensitivity accumulated non-finitely")

            items.append([model_paths[mi], corpus_text, window_text, T,
                          format(L, ".17g"), format(B, ".17g"),
                          format(delta, ".17g")])

    # 按 (-A, MODELS 原下标) 升序排名；A 以 float 精确比较，下标决胜保序。
    order = sorted(range(len(loaded)),
                   key=lambda mi: (-sensitivity[mi], mi))
    ranking = [[model_paths[mi], format(sensitivity[mi], ".17g")]
               for mi in order]

    obj = {"items": items, "ranking": ranking}
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                       allow_nan=False)
            + "\n")


def _rank_window_stability(models_path, list_path, bases_text):
    """rank-window-sensitivity 的多基准窗口稳定度版。

    MODELS、LIST、模型与语料的读取、校验、相对路径解析与重复项保序完
    全沿用 _load_rank_inputs（MODELS 为至少 2 项的非空相对路径 str 数
    组；LIST 为非空 [corpus, window] 数组；所有模型 vocab 须逐项同序
    相等，H 可不同）。BASES 为严格 UTF-8 的 JSON 非空数组，每项 type
    恰为 str 且整串匹配 [1-9][0-9]*（任意位数均合法，全程不转 int），
    重复项按序保留，否则失败。

    对每个 base（按 BASES 原序），按模型原下标 mi、清单原下标 gi 升序
    遍历 (mi, gi)，严格沿用 rank-window-sensitivity：每项的两轨总 NLL
    L、B 均调用 _total_rank_nll 独立重置（h0、c0、memory=[h0]），无跨
    调用缓存；A 从 0.0 按 gi 升序累加未格式化的 abs(L-B)，任一计算非
    有限即抛 ValueError。各 base 内按 (-A, mi) 升序排名（A 以 float
    精确比较，下标决胜保序），名次 r 从 1 起。对每个模型令
    Q=max(r)-min(r)，R 从 0 按 base 序累加 r。整个对象先完整构造再返
    回，失败时不产生任何部分输出，不写文件。

    stdout 恰为单个 JSON 对象加 LF，顶层键序 bases,ranking。bases 按
    BASES 原序，每项恰为 [base,rows]：base 为 BASES 原文 str；rows 按
    名次升序，每项恰为 [model,format(A,'.17g'),r]，model 为 MODELS 原
    文 str，r 为 int（.17g 保留 -0.0）。ranking 按 (Q,R,mi) 升序，每
    项恰为 [model,Q,R]，Q、R 为 int。序列化恰用 json.dumps(obj,
    ensure_ascii=True,separators=(',',':'),allow_nan=False)+'\\n'。
    """
    # BASES 为严格 UTF-8 的 JSON 文本；argv 中的孤立代理等非 UTF-8 可表
    # 示内容在此即失败。
    bases = json.loads(bases_text.encode("utf-8").decode("utf-8"))
    if not isinstance(bases, list) or not bases:
        raise ValueError("BASES must be a non-empty JSON array")
    for base_text in bases:
        if type(base_text) is not str or not _WINDOW_RE.match(base_text):
            raise ValueError("each BASES item must be a string matching "
                             "[1-9][0-9]*")

    model_paths, entries, loaded, entry_ids = _load_rank_inputs(
        models_path, list_path)

    n_models = len(loaded)
    # ranks[bi][mi] 为模型 mi 在第 bi 个 base 下的名次（从 1 起）。
    ranks = []
    bases_out = []
    for base_text in bases:
        # 每个模型的敏感度 A 从 0.0 起，按 gi 升序累加各项 abs(L-B)。
        sensitivity = [0.0] * n_models
        for mi, (cell, Why, by, h0_m, c0_m) in enumerate(loaded):
            for gi, (corpus_text, window_text) in enumerate(entries):
                ids = entry_ids[gi]

                # 两轨均通过 _total_rank_nll 从自身 h0、c0、memory=[h0]
                # 独立重置，无跨调用缓存。
                L = _total_rank_nll(cell, Why, by, h0_m, c0_m, ids,
                                    window_text)
                B = _total_rank_nll(cell, Why, by, h0_m, c0_m, ids,
                                    base_text)

                delta = L - B
                if not math.isfinite(delta):
                    raise ValueError("window delta is non-finite")

                sensitivity[mi] += abs(delta)
                if not math.isfinite(sensitivity[mi]):
                    raise ValueError(
                        "model sensitivity accumulated non-finitely")

        # 按 (-A, MODELS 原下标) 升序排名；A 以 float 精确比较，下标决胜。
        order = sorted(range(n_models),
                       key=lambda mi: (-sensitivity[mi], mi))
        rank_of = [0] * n_models
        rows = []
        for r, mi in enumerate(order, 1):
            rank_of[mi] = r
            rows.append([model_paths[mi],
                         format(sensitivity[mi], ".17g"), r])
        ranks.append(rank_of)
        bases_out.append([base_text, rows])

    ranking = []
    for mi in range(n_models):
        rs = [ranks[bi][mi] for bi in range(len(bases))]
        Q = max(rs) - min(rs)
        R = 0
        for r in rs:
            R += r
        ranking.append((Q, R, mi))
    ranking.sort()
    ranking = [[model_paths[mi], Q, R] for Q, R, mi in ranking]

    obj = {"bases": bases_out, "ranking": ranking}
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                       allow_nan=False)
            + "\n")


def _rank_window_stability_details(models_path, list_path, bases_text):
    """rank-window-stability 的逐项明细版。

    MODELS、LIST、BASES 及模型/语料的读取、校验、相对路径解析、重复项
    保序均沿用 rank-window-stability：模型/语料经 _load_rank_inputs
    加载，逐步总 NLL 经 _total_rank_nll 计算。按原下标 (bi,mi,gi) 升序
    遍历，每项分别以该清单项的 window 与第 bi 个 base 为窗口调用
    _total_rank_nll（均从自身 h0、c0、memory=[h0] 独立重置，无跨调用
    缓存）求总 NLL L、B；令 d=L-B、c=abs(d)，A[bi,mi] 从 0.0 按 gi 升
    序累加未格式化的 c，任一结果非有限即抛 ValueError。各 bi 内按
    (-A,mi) 升序排名（A 以 float 精确比较，下标决胜保序），名次 r 从
    1 起。对每个模型令 Q=max(r)-min(r)，R 从 0.0 按 bi 升序累加 r，最
    终按 (Q,R,mi) 升序。整个对象先完整构造再返回，失败时不产生任何部
    分输出，不写文件。

    stdout 恰为单个 JSON 对象加 LF，顶层键序 items,ranking。items 按
    (bi,mi,gi) 展平，每项恰为
    [base,model,corpus,window,steps,L,B,d,c]：base、model、corpus、
    window 为输入原文 str，steps 为语料码点数减 1 的 int，L、B、d、c
    均为 format(x,'.17g') 字符串（.17g 保留 -0.0）。ranking 按最终名
    次升序，每项恰为 [model,stats,Q,R]：stats 按 BASES 原序，每项恰为
    [base,format(A,'.17g'),r]；Q、R 为 int。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'。
    """
    # BASES 为严格 UTF-8 的 JSON 文本；argv 中的孤立代理等非 UTF-8 可表
    # 示内容在此即失败。
    bases = json.loads(bases_text.encode("utf-8").decode("utf-8"))
    if not isinstance(bases, list) or not bases:
        raise ValueError("BASES must be a non-empty JSON array")
    for base_text in bases:
        if type(base_text) is not str or not _WINDOW_RE.match(base_text):
            raise ValueError("each BASES item must be a string matching "
                             "[1-9][0-9]*")

    model_paths, entries, loaded, entry_ids = _load_rank_inputs(
        models_path, list_path)

    n_models = len(loaded)
    items = []
    # sens[bi][mi] 为第 bi 个 base 下模型 mi 的敏感度 A，从 0.0 按 gi
    # 升序累加各项 abs(L-B)。
    sens = []
    for base_text in bases:
        sensitivity = [0.0] * n_models
        for mi, (cell, Why, by, h0_m, c0_m) in enumerate(loaded):
            for gi, (corpus_text, window_text) in enumerate(entries):
                ids = entry_ids[gi]
                T = len(ids) - 1

                # 两轨均通过 _total_rank_nll 从自身 h0、c0、memory=[h0]
                # 独立重置，无跨调用缓存。
                L = _total_rank_nll(cell, Why, by, h0_m, c0_m, ids,
                                    window_text)
                B = _total_rank_nll(cell, Why, by, h0_m, c0_m, ids,
                                    base_text)

                delta = L - B
                if not math.isfinite(delta):
                    raise ValueError("window delta is non-finite")
                c_abs = abs(delta)

                sensitivity[mi] += c_abs
                if not math.isfinite(sensitivity[mi]):
                    raise ValueError(
                        "model sensitivity accumulated non-finitely")

                items.append([base_text, model_paths[mi], corpus_text,
                              window_text, T,
                              format(L, ".17g"), format(B, ".17g"),
                              format(delta, ".17g"), format(c_abs, ".17g")])
        sens.append(sensitivity)

    # ranks[bi][mi] 为模型 mi 在第 bi 个 base 下的名次（从 1 起）。
    ranks = []
    for bi in range(len(bases)):
        # 按 (-A, MODELS 原下标) 升序排名；A 以 float 精确比较，下标决胜。
        order = sorted(range(n_models),
                       key=lambda mi: (-sens[bi][mi], mi))
        rank_of = [0] * n_models
        for r, mi in enumerate(order, 1):
            rank_of[mi] = r
        ranks.append(rank_of)

    ranking = []
    for mi in range(n_models):
        rs = [ranks[bi][mi] for bi in range(len(bases))]
        Q = max(rs) - min(rs)
        R = 0.0
        for r in rs:
            R += r
        stats = [[bases[bi], format(sens[bi][mi], ".17g"), ranks[bi][mi]]
                 for bi in range(len(bases))]
        ranking.append((Q, R, mi, stats))
    ranking.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    ranking = [[model_paths[mi], stats, Q, int(R)]
               for Q, R, mi, stats in ranking]

    obj = {"items": items, "ranking": ranking}
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                       allow_nan=False)
            + "\n")


def _perplexity_lstm_attn_trace(model_path, corpus_path, window_text):
    """perplexity-lstm-attn 的逐步负对数似然轨迹，返回待写出的字符串。

    MODEL、CORPUS、WINDOW 的校验与逐步计算完全沿用
    _perplexity_lstm_attn。令 T=语料码点数-1；按 t 升序计算
    loss=m+log(d)-z_y，L 从 0.0 依序累加 loss。构造 JSON 对象：顶层键序
    恰为 version,items,total_nll,perplexity；version 为 int 1；items 为
    T 长列表，第 t 项键序恰为 t,target,nll，值依次为 int t、下一单码点
    str、format(loss,'.17g') 字符串；total_nll、perplexity 依次为
    format(L,'.17g')、format(exp(L/T),'.17g') 字符串。返回
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'，不写文件。
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

    memory = [h0]
    h = list(h0)
    c = list(c0)
    L = 0.0
    T = len(ids) - 1
    items = []
    for t in range(T):
        y = ids[t + 1]

        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        z = _output_logits(Why, by, u)

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

        items.append({
            "t": t,
            "target": corpus[t + 1],
            "nll": format(step, ".17g"),
        })

        memory.append([float(v) for v in h])

    perplexity = math.exp(L / T)
    if not math.isfinite(perplexity):
        raise ValueError("perplexity is non-finite")

    obj = {
        "version": 1,
        "items": items,
        "total_nll": format(L, ".17g"),
        "perplexity": format(perplexity, ".17g"),
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _lstm_attn_weights(model_path, corpus_path, window_text):
    """perplexity-lstm-attn 每步注意力权重轨迹，返回待写出的字符串。

    MODEL、CORPUS、WINDOW 的校验、状态推进（h、c 与 memory）与 M 截取
    完全沿用 _perplexity_lstm_attn，但不计算输出层与负对数似然（它们不
    影响状态）。令 T=语料码点数-1；t 升序先以当前字符 one-hot 调用
    LSTMCell.forward 更新 h、c，M 取 memory 末尾至多 WINDOW 项（从旧到
    新），在向 memory 追加 h 前以 attention([h], M, M, None) 取 w[0]，
    并令 start=t+1-len(M)。构造 JSON 对象：顶层键序恰为 version,items；
    version 为 int 1；items 为 T 长列表，第 t 项键序恰为 t,start,weights，
    前两值为 int t、int start，weights 为按 M 从旧到新排列的字符串列表，
    第 j 项恰为 format(w[0][j],'.17g')。memory 下标 0 为 h0、q≥1 为
    h_(q-1)，weights[j] 对应 start+j。返回
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'，不写文件。
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

    memory = [h0]
    h = list(h0)
    c = list(c0)
    T = len(ids) - 1
    items = []
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        _ctx, w = attention([h], M, M, None)
        start = t + 1 - len(M)

        items.append({
            "t": t,
            "start": start,
            "weights": [format(weight, ".17g") for weight in w[0]],
        })

        memory.append([float(v) for v in h])

    obj = {
        "version": 1,
        "items": items,
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _lstm_attn_entropy(model_path, corpus_path, window_text):
    """perplexity-lstm-attn 每步注意力权重的平均熵，返回待写出的字符串。

    MODEL、CORPUS、WINDOW 的校验、状态推进（h、c 与 memory）与 M 截取
    完全沿用 _lstm_attn_weights，但不计算输出层与负对数似然（它们不影响
    状态）。令 T=语料码点数-1、E=0.0；t 升序先以当前字符 one-hot 调用
    LSTMCell.forward 更新 h、c，M 取 memory 末尾至多 WINDOW 项（从旧到
    新），在向 memory 追加 h 前以 attention([h], M, M, None) 取 w[0]；
    第 t 步令 e=0.0，按 j 升序累加 e-=w[j]*log(w[j])（w[j] 恰为 0.0 时
    贡献 0.0，不取对数），再令 E+=e。任一 log、乘加或 E/T 非有限均抛
    ValueError。构造 JSON 对象：顶层键序恰为 version,steps,mean_entropy，
    值依次为 int 1、int T、format(E/T,'.17g') 字符串。返回
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'，不写文件。
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

    memory = [h0]
    h = list(h0)
    c = list(c0)
    T = len(ids) - 1
    E = 0.0
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        _ctx, w = attention([h], M, M, None)

        # 第 t 步熵：e 自 0.0 起按 j 升序累加 -w[j]*log(w[j])；
        # w[j] 恰为 0.0 时贡献 0.0（不取对数）。
        e = 0.0
        for wj in w[0]:
            if wj != 0.0:
                lj = math.log(wj)
                if not math.isfinite(lj):
                    raise ValueError("entropy log became non-finite")
                e -= wj * lj
                if not math.isfinite(e):
                    raise ValueError(
                        "entropy accumulated to a non-finite value")
        E += e
        if not math.isfinite(E):
            raise ValueError("total entropy accumulated to a non-finite value")

        memory.append([float(v) for v in h])

    mean = E / T
    if not math.isfinite(mean):
        raise ValueError("mean entropy is non-finite")

    obj = {
        "version": 1,
        "steps": T,
        "mean_entropy": format(mean, ".17g"),
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _lstm_attn_mean_lag(model_path, corpus_path, window_text):
    """perplexity-lstm-attn 每步注意力权重的平均滞后，返回待写出的字符串。

    MODEL、CORPUS、WINDOW 的校验、状态推进（h、c 与 memory）与 M 截取
    完全沿用 _lstm_attn_weights，但不计算输出层与负对数似然（它们不影响
    状态）。令 T=语料码点数-1、A=0.0；t 升序先以当前字符 one-hot 调用
    LSTMCell.forward 更新 h、c，M 取 memory 末尾至多 WINDOW 项（从旧到
    新），在向 memory 追加 h 前以 attention([h], M, M, None) 取 w[0]，
    并令 start=t+1-len(M)；第 t 步令 lag=0.0，按 j 升序令
    q=start+j、r=t+1-q，自 0.0 累加 lag+=w[0][j]*float(r)，再令
    A+=lag。任一乘加或 A/T 非有限均抛 ValueError。构造 JSON 对象：顶层
    键序恰为 version,items,mean_lag；version 为 int 1；items 为 T 长列
    表，第 t 项键序恰为 t,start,lag，值依次为 int t、int start、
    format(lag,'.17g') 字符串；mean_lag 为 format(A/T,'.17g') 字符串。
    返回 json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'，不写文件。
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

    memory = [h0]
    h = list(h0)
    c = list(c0)
    T = len(ids) - 1
    A = 0.0
    items = []
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        _ctx, w = attention([h], M, M, None)
        start = t + 1 - len(M)

        # 第 t 步滞后：lag 自 0.0 起按 j 升序累加 w[0][j]*float(t+1-(start+j))。
        lag = 0.0
        for j in range(len(M)):
            q = start + j
            r = t + 1 - q
            lag += w[0][j] * float(r)
            if not math.isfinite(lag):
                raise ValueError("lag accumulated to a non-finite value")
        A += lag
        if not math.isfinite(A):
            raise ValueError("total lag accumulated to a non-finite value")

        items.append({
            "t": t,
            "start": start,
            "lag": format(lag, ".17g"),
        })

        memory.append([float(v) for v in h])

    mean = A / T
    if not math.isfinite(mean):
        raise ValueError("mean lag is non-finite")

    obj = {
        "version": 1,
        "items": items,
        "mean_lag": format(mean, ".17g"),
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _lstm_attn_reach(model_path, corpus_path, window_text, mass_text):
    """perplexity-lstm-attn 每步注意力权重的质量回眺长度，返回待写出的字符串。

    MODEL、CORPUS、WINDOW 的校验、状态推进（h、c 与 memory）与 M 截取
    完全沿用 _lstm_attn_weights，但不计算输出层与负对数似然（它们不影响
    状态），也不写文件。MASS 经 float() 解析，须有限且 0<MASS<=1，否则
    抛 ValueError。令 T=语料码点数-1、A=0.0；t 升序先以当前字符 one-hot
    调用 LSTMCell.forward 更新 h、c，M 取 memory 末尾至多 WINDOW 项（从
    旧到新），在向 memory 追加 h 前以 attention([h], M, M, None) 取
    w[0]，并令 start=t+1-len(M)、c=0.0；j 自 len(M)-1 降至 0 累加
    c+=w[0][j]，首次 c>=MASS 即令 reach=len(M)-j 并停止，否则
    reach=len(M)。c 及按 t 升序执行的 A+=float(reach) 均须有限，否则抛
    ValueError。构造 JSON 对象：顶层键序恰为 version,mass,items,
    mean_reach；version 为 int 1；mass 为 format(MASS,'.17g') 字符串；
    items 为 T 长列表，按 t 升序，第 t 项为三个 int 的列表
    [t,start,reach]；mean_reach 为 format(A/T,'.17g') 字符串。返回
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")
    mass = float(mass_text)
    if not math.isfinite(mass) or not 0.0 < mass <= 1.0:
        raise ValueError("MASS must be finite and satisfy 0 < MASS <= 1")

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

    memory = [h0]
    h = list(h0)
    c = list(c0)
    T = len(ids) - 1
    A = 0.0
    items = []
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        _ctx, w = attention([h], M, M, None)
        start = t + 1 - len(M)

        # 第 t 步质量回眺：自最新向最旧累加权重，首次达到 MASS 时的项数。
        acc = 0.0
        reach = len(M)
        for j in range(len(M) - 1, -1, -1):
            acc += w[0][j]
            if not math.isfinite(acc):
                raise ValueError("mass accumulated to a non-finite value")
            if acc >= mass:
                reach = len(M) - j
                break
        A += float(reach)
        if not math.isfinite(A):
            raise ValueError("total reach accumulated to a non-finite value")

        items.append([t, start, reach])

        memory.append([float(v) for v in h])

    obj = {
        "version": 1,
        "mass": format(mass, ".17g"),
        "items": items,
        "mean_reach": format(A / T, ".17g"),
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _lstm_attn_reach_profile(model_path, corpus_path, window_text,
                             masses_path):
    """lstm-attn-reach 的多质量批量版本，返回待写出的字符串。

    MODEL、CORPUS、WINDOW 的校验、状态推进（h、c 与 memory）、M 截取与
    逐步权重计算完全沿用 _lstm_attn_reach，故每个 (t,m) 的 start、reach
    与单独调用 _lstm_attn_reach 逐位相同；不写文件。MASSES 为严格
    UTF-8 JSON 文件：顶层须为非空数组，元素 type 恰为 str，其 float()
    值 m 须有限且 0<m<=1，重复项按原序保留，否则抛 ValueError。

    令 T=语料码点数-1；t 升序的单次状态推进内，对原序每个 m 独立复用
    “自最新向最旧累计质量”规则（各 m 的累加器自 0.0 起互不相干），
    reach_m(t) 与单独调用一致；A_m 自 0.0 起按 t 升序累加
    float(reach_m(t))。任一累加中间量非有限即抛 ValueError。构造 JSON
    对象：顶层键序恰为 version,masses,items,mean_reach；version 为
    int 1；masses 为原序 format(m,'.17g') 字符串列表；items 按 t 升序，
    每项恰为 [t,start,reaches]，前两项为 int，reaches 为与 masses 对齐
    的 int 列表；mean_reach 为与 masses 对齐的
    format(A_m/T,'.17g') 字符串列表。返回
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    # MASSES：严格 UTF-8 解码后经 json 解析；顶层须为非空数组，元素须为
    # type 恰为 str 的有限质量文本（重复项按原序保留）。
    with open(masses_path, "rb") as f:
        masses_raw = f.read()
    masses_data = json.loads(masses_raw.decode("utf-8"))
    if type(masses_data) is not list or len(masses_data) == 0:
        raise ValueError("MASSES must be a non-empty JSON array")
    masses = []
    for element in masses_data:
        if type(element) is not str:
            raise ValueError("each MASSES element must be a JSON string")
        m = float(element)
        if not math.isfinite(m) or not 0.0 < m <= 1.0:
            raise ValueError(
                "each mass must be finite and satisfy 0 < mass <= 1")
        masses.append(m)

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

    n_masses = len(masses)
    memory = [h0]
    h = list(h0)
    c = list(c0)
    T = len(ids) - 1
    totals = [0.0] * n_masses
    items = []
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0

        h, c = cell.forward(x, h, c)[:2]

        M = _window_tail(memory, window_text)
        _ctx, w = attention([h], M, M, None)
        start = t + 1 - len(M)

        # 各 m 独立复用自最新向最旧累加权重的规则，首次达到 m 时的项数。
        reaches = [len(M)] * n_masses
        for k in range(n_masses):
            mass = masses[k]
            acc = 0.0
            reach = len(M)
            for j in range(len(M) - 1, -1, -1):
                acc += w[0][j]
                if not math.isfinite(acc):
                    raise ValueError(
                        "mass accumulated to a non-finite value")
                if acc >= mass:
                    reach = len(M) - j
                    break
            totals[k] += float(reach)
            if not math.isfinite(totals[k]):
                raise ValueError(
                    "total reach accumulated to a non-finite value")
            reaches[k] = reach

        items.append([t, start, reaches])

        memory.append([float(v) for v in h])

    obj = {
        "version": 1,
        "masses": [format(m, ".17g") for m in masses],
        "items": items,
        "mean_reach": [format(totals[k] / T, ".17g")
                       for k in range(n_masses)],
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


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


def _score_lstm_attn(model_path, start, text, temperature_text, window_text):
    """确定性评分指定候选 TEXT，返回待写出的字符串。

    MODEL、START、TEMPERATURE、WINDOW 的读取、词法、形状、F 及失败契约
    均沿用 sample-lstm-attn；TEXT 为可空 Unicode 字符串且每个码点须在
    vocab，否则抛 ValueError。本命令无 SEED、无随机源且不写文件。

    置 h=h0、c=c0、x=START 索引、memory=[h0]、total=0.0。按 t 升序遍历
    TEXT：以 x 的 V 长 one-hot 调用装入 W、b 的 LSTMCell.forward 更新
    h、c；M 取 memory 末尾至多 WINDOW 项（顺序从旧到新），以
    attention([h], M, M, None) 返回首项 ctx 得 u；按原顺序计算
    a[k]=z[k]/TEMPERATURE、m=max(a)、e[k]=exp(a[k]-m)，d 自 0.0 依 k
    升序累加 e[k]；不抽样，令 y 为 TEXT[t] 索引，
    lp=a[y]-m-math.log(d)，total 按 t 升序累加 lp，置 x=y，memory 追加
    h 的 float 副本。任一计算非有限均抛 ValueError。成功返回
    format(total, '.17g') + '\\n'（TEXT 为空时为 "0\\n"）。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # TEMPERATURE：float() 可解析且有限、严格大于 0；inf/nan/0/负数均失败。
    temperature = float(temperature_text)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("TEMPERATURE must be a finite positive float")

    # TEXT：可空 Unicode 字符串，每个码点须在词表内。
    if type(text) is not str:
        raise ValueError("TEXT must be a string")
    table = {ch: i for i, ch in enumerate(vocab)}
    ids = [0] * len(text)
    for t, ch in enumerate(text):
        ix = table.get(ch)
        if ix is None:
            raise ValueError("TEXT contains an out-of-vocab character")
        ids[t] = ix

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    memory = [h0]
    h = list(h0)
    c = list(c0)
    x = vocab.index(start)
    total = 0.0

    for t in range(len(ids)):
        # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
        xvec = [0.0] * V
        xvec[x] = 1.0

        h, c = cell.forward(xvec, h, c)[:2]

        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)

        # z_k = by_k + Σ_j Why_k,j*u_j，依 j 升序自 float 偏置累加。
        z = _output_logits(Why, by, u)

        # a_k=z_k/TEMPERATURE，m=max(a)，e_k=exp(a_k-m)，d 自 0.0 依 k
        # 升序累加。
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

        # 不抽样：y 为 TEXT[t] 索引，lp=a[y]-m-log(d)，total 依 t 升序累加。
        y = ids[t]
        lp = a[y] - m - math.log(d)
        if not math.isfinite(lp):
            raise ValueError("log probability is non-finite")
        total += lp
        if not math.isfinite(total):
            raise ValueError(
                "total log probability accumulated non-finitely")

        x = y
        memory.append([float(v) for v in h])

    return format(total, ".17g") + "\n"


def _score_lstm_attn_batch(model_path, start, candidates_path,
                           temperature_text, window_text):
    """确定性评分 CANDIDATES 文件中的多个候选串，返回待写出的字符串。

    MODEL、START、TEMPERATURE、WINDOW 的读取、词法、形状、F 及失败契约
    均沿用 score-lstm-attn。CANDIDATES 为严格 UTF-8 的 JSON 文件，顶层须
    为非空数组；每个元素 type 恰为 str（可为空串），且每个码点须在
    vocab，否则抛 ValueError。本命令无 SEED、无随机源且不写文件。

    逐个候选独立评分：每个候选均重置 h=h0、c=c0、x=START 索引、
    memory=[h0]、total=0.0，候选间不共享任何状态。按 t 升序遍历候选码
    点：以 x 的 V 长 one-hot 调用装入 W、b 的 LSTMCell.forward 更新
    h、c；M 取 memory 末尾至多 WINDOW 项（顺序从旧到新），以
    attention([h], M, M, None) 返回首项 ctx 得 u；按原顺序计算
    a[k]=z[k]/TEMPERATURE、m=max(a)，d 自 0.0 依 k 升序累加
    exp(a[k]-m)；令 y 为候选第 t 个码点索引，
    lp=a[y]-m-math.log(d)，total 按 t 升序累加 lp，置 x=y，memory 追加
    h 的 float 副本。任一计算非有限均抛 ValueError。

    成功返回单个 JSON 对象加 LF，键序 items,best。items 按 CANDIDATES
    原序，每项恰为 [text,steps,format(total,'.17g')]：text 为候选原文
    str（空串为 ""），steps 为码点数 int；best 为最高 total 的 int 下
    标，平分时取较小下标。序列化恰用 json.dumps(obj,ensure_ascii=True,
    separators=(',',':'),allow_nan=False)+'\\n'。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # TEMPERATURE：float() 可解析且有限、严格大于 0；inf/nan/0/负数均失败。
    temperature = float(temperature_text)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("TEMPERATURE must be a finite positive float")

    # CANDIDATES：严格 UTF-8 的 JSON 非空数组，元素为可空 str。
    with open(candidates_path, "rb") as f:
        candidates = json.loads(f.read().decode("utf-8"))
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("CANDIDATES must be a non-empty JSON array")

    table = {ch: i for i, ch in enumerate(vocab)}
    candidates_ids = []
    for text in candidates:
        if type(text) is not str:
            raise ValueError("each candidate must be a string")
        ids = [0] * len(text)
        for t, ch in enumerate(text):
            ix = table.get(ch)
            if ix is None:
                raise ValueError(
                    "candidate contains an out-of-vocab character")
            ids[t] = ix
        candidates_ids.append(ids)

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)
    start_id = vocab.index(start)

    items = []
    best = 0
    best_total = None
    for ci, ids in enumerate(candidates_ids):
        # 每个候选独立重置 h0、c0、START 与 memory，候选间不共享状态。
        memory = [h0]
        h = list(h0)
        c = list(c0)
        x = start_id
        total = 0.0

        for t in range(len(ids)):
            # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
            xvec = [0.0] * V
            xvec[x] = 1.0

            h, c = cell.forward(xvec, h, c)[:2]

            M = _window_tail(memory, window_text)
            u = _attn_context(h, M)

            # z_k = by_k + Σ_j Why_k,j*u_j，依 j 升序自 float 偏置累加。
            z = _output_logits(Why, by, u)

            # a_k=z_k/TEMPERATURE，m=max(a)；d 自 0.0 依 k 升序累加
            # exp(a_k-m)。
            a = [0.0] * V
            m = None
            for k in range(V):
                ak = z[k] / temperature
                if not math.isfinite(ak):
                    raise ValueError("scaled logit became non-finite")
                a[k] = ak
                if m is None or ak > m:
                    m = ak
            d = 0.0
            for k in range(V):
                try:
                    ek = math.exp(a[k] - m)
                except OverflowError:
                    raise ValueError("softmax exp overflowed")
                if not math.isfinite(ek):
                    raise ValueError("softmax exp became non-finite")
                d += ek
                if not math.isfinite(d):
                    raise ValueError(
                        "softmax denominator accumulated non-finitely")

            # 不抽样：y 为候选第 t 个码点索引，lp=a[y]-m-log(d)，total
            # 依 t 升序累加。
            y = ids[t]
            lp = a[y] - m - math.log(d)
            if not math.isfinite(lp):
                raise ValueError("log probability is non-finite")
            total += lp
            if not math.isfinite(total):
                raise ValueError(
                    "total log probability accumulated non-finitely")

            x = y
            memory.append([float(v) for v in h])

        items.append([candidates[ci], len(ids), format(total, ".17g")])
        # best 为最高 total 的下标；仅严格大于才更新，平分保留小下标。
        if best_total is None or total > best_total:
            best_total = total
            best = ci

    obj = {"items": items, "best": best}
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                       allow_nan=False)
            + "\n")


def _score_lstm_attn_temps(model_path, start, candidates_path,
                           temps_text, window_text):
    """确定性评分 CANDIDATES 文件中多个候选串在多个温度下的得分。

    除 TEMPS、遍历顺序及输出外，MODEL、START、CANDIDATES、WINDOW 的读取、
    词法、形状、F、逐步 one-hot 推进、窗口注意力、a[k]=z[k]/TEMPERATURE、
    m=max(a)、d 自 0.0 依 k 升序累加 exp(a[k]-m)、
    lp=a[y]-m-math.log(d) 与失败契约均沿用 _score_lstm_attn_batch。本命令
    无 SEED、无随机源且不写文件。

    TEMPS 为 argv 中严格 UTF-8 的 JSON 文本（经 encode('utf-8') 再 decode
    以拒绝孤立代理等非 UTF-8 可表示内容），顶层须为非空数组；每个元素
    type 恰为 str，其 float() 值须有限且严格大于 0，重复项按序保留，否则
    抛 ValueError。

    按 (ti, ci) 升序遍历温度与候选：每组均从 h=h0、c=c0、x=START 索引、
    memory=[h0]、total=0.0 独立重置，组间不共享任何状态，沿用 t、k 顺序
    累加未格式化的 total，任一计算非有限均抛 ValueError。

    成功返回单个 JSON 对象加 LF，键序 temperatures,items,best。
    temperatures 为 TEMPS 原序的 format(temp,'.17g') 字符串列表；items 按
    (ti,ci) 展平，每项恰为 [ti,ci,text,steps,format(total,'.17g')]：
    ti、ci、steps 为 int，text 为候选原文 str；best 为最高 total 的
    [ti,ci]，平分时取较小 ti，再取较小 ci。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # TEMPS：严格 UTF-8 的 JSON 非空数组，元素 type 恰为 str，float() 值
    # 须有限且严格大于 0；重复项按序保留。
    temps = json.loads(temps_text.encode("utf-8").decode("utf-8"))
    if not isinstance(temps, list) or not temps:
        raise ValueError("TEMPS must be a non-empty JSON array")
    temperatures = []
    for temp_text in temps:
        if type(temp_text) is not str:
            raise ValueError("each temperature must be a string")
        temperature = float(temp_text)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                "each temperature must be a finite positive float")
        temperatures.append(temperature)

    # CANDIDATES：严格 UTF-8 的 JSON 非空数组，元素为可空 str。
    with open(candidates_path, "rb") as f:
        candidates = json.loads(f.read().decode("utf-8"))
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("CANDIDATES must be a non-empty JSON array")

    table = {ch: i for i, ch in enumerate(vocab)}
    candidates_ids = []
    for text in candidates:
        if type(text) is not str:
            raise ValueError("each candidate must be a string")
        ids = [0] * len(text)
        for t, ch in enumerate(text):
            ix = table.get(ch)
            if ix is None:
                raise ValueError(
                    "candidate contains an out-of-vocab character")
            ids[t] = ix
        candidates_ids.append(ids)

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)
    start_id = vocab.index(start)

    temps_out = [format(temperature, ".17g") for temperature in temperatures]
    items = []
    best_ti = 0
    best_ci = 0
    best_total = None
    # 按 (ti, ci) 升序遍历；每组独立重置，组间不共享状态。
    for ti, temperature in enumerate(temperatures):
        for ci, ids in enumerate(candidates_ids):
            # 每组均重置 h0、c0、START 与 memory。
            memory = [h0]
            h = list(h0)
            c = list(c0)
            x = start_id
            total = 0.0

            for t in range(len(ids)):
                # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
                xvec = [0.0] * V
                xvec[x] = 1.0

                h, c = cell.forward(xvec, h, c)[:2]

                M = _window_tail(memory, window_text)
                u = _attn_context(h, M)

                # z_k = by_k + Σ_j Why_k,j*u_j，依 j 升序自 float 偏置累加。
                z = _output_logits(Why, by, u)

                # a_k=z_k/TEMPERATURE，m=max(a)；d 自 0.0 依 k 升序累加
                # exp(a_k-m)。
                a = [0.0] * V
                m = None
                for k in range(V):
                    ak = z[k] / temperature
                    if not math.isfinite(ak):
                        raise ValueError("scaled logit became non-finite")
                    a[k] = ak
                    if m is None or ak > m:
                        m = ak
                d = 0.0
                for k in range(V):
                    try:
                        ek = math.exp(a[k] - m)
                    except OverflowError:
                        raise ValueError("softmax exp overflowed")
                    if not math.isfinite(ek):
                        raise ValueError("softmax exp became non-finite")
                    d += ek
                    if not math.isfinite(d):
                        raise ValueError(
                            "softmax denominator accumulated non-finitely")

                # 不抽样：y 为候选第 t 个码点索引，lp=a[y]-m-log(d)，total
                # 依 t 升序累加。
                y = ids[t]
                lp = a[y] - m - math.log(d)
                if not math.isfinite(lp):
                    raise ValueError("log probability is non-finite")
                total += lp
                if not math.isfinite(total):
                    raise ValueError(
                        "total log probability accumulated non-finitely")

                x = y
                memory.append([float(v) for v in h])

            items.append([ti, ci, candidates[ci], len(ids),
                          format(total, ".17g")])
            # best 为最高 total 的 [ti,ci]；仅严格大于才更新，平分时遍历
            # 序 (ti,ci) 保证保留较小 ti、再较小 ci。
            if best_total is None or total > best_total:
                best_total = total
                best_ti = ti
                best_ci = ci

    obj = {"temperatures": temps_out, "items": items,
           "best": [best_ti, best_ci]}
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                       allow_nan=False)
            + "\n")


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


def _sample_lstm_attn_topp_run(model_path, start, seed_text, start_t_text,
                               end_t_text, top_p_text, length_text,
                               window_text):
    """sample-lstm-attn-top-p 与 -scored 共享的采样核心。

    校验、状态推进、温度退火、top-p 前缀截取、随机源初始化与消费、选索引
    规则及有限性失败契约均与 _sample_lstm_attn_topp 文档一致。返回
    (chars, logprobs, total)：chars 为生成码点列表；logprobs 为每步选中
    索引 k 对应的 a[k]-m-log(s)（s 为 top-p 前缀质量）；total 自 0.0 按 t
    升序累加各 lp。任一 lp 或 total 非有限均抛 ValueError。

    除 TOP_P 及下述选样外，MODEL、START、SEED、LENGTH、WINDOW、LSTM 状态、
    注意力记忆、Why/by logit、稳定 softmax、线性温度退火（LENGTH 为 1 时
    仅用 START_T，为 0 时不计算温度）与有限性失败契约均沿用
    _sample_lstm_attn_anneal；WINDOW 词法与安全截取沿用
    perplexity-lstm-attn。TOP_P 经 float() 解析，结果须有限且
    0<TOP_P<=1，否则抛 ValueError。整次调用仅初始化一次
    r=random.Random(int(SEED))，不写任何文件。

    每步先按原顺序求温度缩放后的 e_k=exp(a_k-max(a))，d 自 0.0 依 k 升序
    累加。再将索引按 (-e_k, k) 升序排列（e 降序、并列时 k 升序），依该序
    自 0.0 累加 e，截取首个使累计值 >=TOP_P*d 的最短前缀；s 为该前缀按
    该序自 0.0 累加所得之和。令 u=r.random()*s，再按前缀顺序自 0.0 累加
    e，选首个累计值严格大于 u 的索引；无则取前缀末项。其字符追加到输出
    并作为下一输入 x，随后向 memory 追加 h 的 float 副本。任一新增运算
    非有限均抛 ValueError。
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

    # TOP_P：float() 可解析且有限，0<TOP_P<=1。
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
    logprobs = []
    total = 0.0

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

        # 核阈值 TOP_P*d 须有限。
        target = top_p * d
        if not math.isfinite(target):
            raise ValueError("top-p target accumulated non-finitely")

        # 索引按 (-e_k, k) 升序：e 降序、并列时 k 升序。
        order = sorted(range(V), key=lambda k: (-e[k], k))

        # 依该序自 0.0 累加 e，截取首个累计值 >=TOP_P*d 的最短前缀；s 为
        # 其自 0.0 依该序累加所得之和。浮点求和顺序不同可能令全量累计与
        # d 相差一 ULP，此时以全量索引为前缀（数学上其和恰为 d>=目标）。
        prefix = []
        s = 0.0
        reached = False
        for idx in order:
            prefix.append(idx)
            s += e[idx]
            if not math.isfinite(s):
                raise ValueError("top-p prefix accumulated non-finitely")
            if s >= target:
                reached = True
                break
        if not reached:
            prefix = list(order)

        # u=r.random()*s；按前缀顺序自 0.0 累加 e，选首个累计值严格大于
        # u 者；无则取前缀末项。
        threshold = rng.random() * s
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = prefix[-1]
        cum = 0.0
        for idx in prefix:
            cum += e[idx]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = idx
                break

        # 选中索引 k 后，以既有 a、m 与前缀质量 s 计算选中项对数概率，
        # total 自 0.0 按 t 升序累加；任一结果非有限即失败。
        lp = a[chosen] - m - math.log(s)
        if not math.isfinite(lp):
            raise ValueError("selected log-prob became non-finite")
        total += lp
        if not math.isfinite(total):
            raise ValueError("total log-prob accumulated non-finitely")
        logprobs.append(lp)

        out.append(vocab[chosen])
        x = chosen
        memory.append([float(v) for v in h])

    return "".join(out), logprobs, total


def _sample_lstm_attn_topp(model_path, start, seed_text, start_t_text,
                           end_t_text, top_p_text, length_text, window_text):
    """sample-lstm-attn-top-p：成功返回 LENGTH 个码点再加一个 LF。

    采样与校验全部沿用 _sample_lstm_attn_topp_run，本包装仅取其生成文本。
    """
    text, _logprobs, _total = _sample_lstm_attn_topp_run(
        model_path, start, seed_text, start_t_text, end_t_text, top_p_text,
        length_text, window_text)
    return text + "\n"


def _sample_lstm_attn_topp_scored(model_path, start, seed_text, start_t_text,
                                  end_t_text, top_p_text, length_text,
                                  window_text):
    """sample-lstm-attn-top-p-scored：输出文本、逐步对数概率与累计对数概率。

    全部校验、状态推进、温度退火、top-p 前缀截取、随机源初始化与消费、选
    索引规则及错误协议均沿用 sample-lstm-attn-top-p；同参须消费相同随机
    序列并生成与原入口一致的 text。每步选中索引 k 后，以既有 a、m 和前缀
    质量 s 计算 lp=a[k]-m-log(s)；total 从 0.0 按 t 升序累加 lp，任一结
    果非有限即失败。stdout 恰为单个 JSON 对象加 LF，键序
    text,logprobs,total_logprob；text 为生成字符串（不含尾随 LF），
    logprobs 为 LENGTH 长字符串列表、第 t 项为 format(lp,'.17g')，
    total_logprob 为 format(total,'.17g')。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'。不写文件。
    """
    text, logprobs, total = _sample_lstm_attn_topp_run(
        model_path, start, seed_text, start_t_text, end_t_text, top_p_text,
        length_text, window_text)
    obj = {
        "text": text,
        "logprobs": [format(lp, ".17g") for lp in logprobs],
        "total_logprob": format(total, ".17g"),
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _sample_lstm_attn_topp_batch(model_path, start, seeds_path,
                                 start_t_text, end_t_text, top_p_text,
                                 length_text, window_text):
    """sample-lstm-attn-top-p-batch：对 SEEDS 文件中的多个种子独立采样。

    SEEDS 为严格 UTF-8 的 JSON 非空数组文件；每个元素 type 恰为 str，且
    整串匹配 _INT_RE（0|-?[1-9][0-9]*，禁止空白、+ 前缀、前导零、下划
    线），重复项按原序保留；文件读取、UTF-8/JSON 解析或元素非法即按既
    有协议失败。MODEL、START、START_T、END_T、TOP_P、LENGTH、WINDOW 与
    其余校验、解码、逐步得分、total 累加及错误协议均沿用
    sample-lstm-attn-top-p-scored。

    按 SEEDS 原序逐项独立调用 _sample_lstm_attn_topp_run，每项逐值等同
    于以该 seed 单独调用 sample-lstm-attn-top-p-scored 入口，项间不共享
    任何随机源或采样状态。令 d_i=Σ_j H(text_i,text_j)，H 为两串逐码点
    不等位置数（较短串长度之外的位置均计为不等），j 按原序自 0 累加；
    medoid 按 (d_i,-total_i,i) 升序取首项下标，total 使用累加所得的未
    格式化 float。不写任何文件。

    stdout 恰为单个 JSON 对象加 LF，键序 runs,medoid；runs[i] 恰为
    [SEEDS[i],text_i,logs_i,format(total_i,'.17g'),d_i]，SEEDS[i] 为原
    始种子 str，text_i 为生成字符串（不含尾随 LF），logs_i 为逐步
    format(lp,'.17g') 字符串列表，d_i 为 int；medoid 为所选 int 下标。
    序列化恰用 json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n' 的 UTF-8 字节。
    """
    # SEEDS：严格 UTF-8 的 JSON 非空数组，元素为匹配整数词法的 str。
    with open(seeds_path, "rb") as f:
        seeds = json.loads(f.read().decode("utf-8"))
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("SEEDS must be a non-empty JSON array")
    for seed_text in seeds:
        if type(seed_text) is not str or not _INT_RE.match(seed_text):
            raise ValueError(
                "each seed must match 0|-?[1-9][0-9]*")

    runs = []
    texts = []
    totals = []
    for seed_text in seeds:
        # 每项均以独立调用采样，随机源在核心内按 seed 新建，项间不共享
        # 任何状态；返回值逐值等同于单独调用 -scored 入口。
        text, logprobs, total = _sample_lstm_attn_topp_run(
            model_path, start, seed_text, start_t_text, end_t_text,
            top_p_text, length_text, window_text)
        runs.append([
            seed_text,
            text,
            [format(lp, ".17g") for lp in logprobs],
            format(total, ".17g"),
            0,
        ])
        texts.append(text)
        totals.append(total)

    n = len(texts)
    for i in range(n):
        # H(text_i,text_j)：等长部分逐码点比较，长度差位置全部计不等；
        # d_i 按 j 原序自 0 累加。
        d_i = 0
        ti = texts[i]
        for tj in texts:
            if len(ti) >= len(tj):
                longer, shorter = ti, tj
            else:
                longer, shorter = tj, ti
            dij = len(longer) - len(shorter)
            for k, ch in enumerate(shorter):
                if ch != longer[k]:
                    dij += 1
            d_i += dij
        runs[i][4] = d_i

    # medoid 按 (d_i,-total_i,i) 升序取首项；total 用未格式化 float。
    medoid = min(range(n), key=lambda i: (runs[i][4], -totals[i], i))

    obj = {"runs": runs, "medoid": medoid}
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _sample_lstm_attn_topk_run(model_path, start, seed_text, start_t_text,
                               end_t_text, top_k_text, length_text,
                               window_text):
    """sample-lstm-attn-top-k 与 -scored 共享的采样核心。

    校验、状态推进、温度退火、top-k 候选截取、随机源初始化与消费、选索引
    规则及有限性失败契约均与 _sample_lstm_attn_topk 文档一致。返回
    (chars, logprobs, total)：chars 为生成码点列表；logprobs 为每步选中
    索引 k 对应的 a[k]-m-log(s)（s 为 top-k 候选质量）；total 自 0.0 按
    t 升序累加各 lp。任一 lp 或 total 非有限均抛 ValueError。

    除 TOP_K 及下述候选截取外，MODEL、START、SEED、LENGTH、WINDOW、LSTM
    状态、注意力记忆、Why/by logit、稳定 softmax、线性温度退火（LENGTH 为
    1 时仅用 START_T，为 0 时不计算温度）与有限性失败契约均沿用
    _sample_lstm_attn_topp；WINDOW 词法与安全截取沿用
    perplexity-lstm-attn。整次调用仅初始化一次
    r=random.Random(int(SEED))，不写任何文件。

    TOP_K 整串匹配 [1-9][0-9]*，且数学值 K 不超过 V=len(vocab)。先按十
    进制位数及同长度字典序与 V 的十进制文本比较：位数更长、或同位数且字
    典序更大即越界失败；仅比较通过后才转 int，任意位数文本都不触发整数
    转换异常。

    每步先按原顺序求温度缩放后的 e_k=exp(a_k-max(a))，d 自 0.0 依 k 升
    序累加。再将索引按 (-e_k, k) 升序排列（e 降序、并列时 k 升序），候
    选恰为该序前 K 项；s 自 0.0 按候选序累加 e。令 u=r.random()*s，再
    按候选序自 0.0 累加 e，选首个累计值严格大于 u 的索引；无则取候选末
    项。其字符追加到输出并作为下一输入 x，随后向 memory 追加 h 的 float
    副本。任一新增运算非有限均抛 ValueError。
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

    # TOP_K：整串匹配 [1-9][0-9]*，且 K<=V。先以十进制位数、同位数字典序
    # 与 str(V) 比较，越界即失败；仅通过后才 int()，任何长度文本都不会触发
    # 整数转换异常（str(V) 受内存约束而位数有界）。
    if not _WINDOW_RE.match(top_k_text):
        raise ValueError("TOP_K must match [1-9][0-9]*")
    v_text = str(V)
    if len(top_k_text) > len(v_text) or (
            len(top_k_text) == len(v_text) and top_k_text > v_text):
        raise ValueError("TOP_K must not exceed len(vocab)")
    top_k = int(top_k_text)

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
    logprobs = []
    total = 0.0

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

        # 索引按 (-e_k, k) 升序：e 降序、并列时 k 升序；候选恰为前 K 项。
        order = sorted(range(V), key=lambda k: (-e[k], k))
        candidates = order[:top_k]

        # s 自 0.0 按候选序累加 e。
        s = 0.0
        for idx in candidates:
            s += e[idx]
            if not math.isfinite(s):
                raise ValueError("top-k mass accumulated non-finitely")

        # u=r.random()*s；按候选序自 0.0 累加 e，选首个累计值严格大于 u
        # 者；无则取候选末项。
        threshold = rng.random() * s
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = candidates[-1]
        cum = 0.0
        for idx in candidates:
            cum += e[idx]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = idx
                break

        # 选中索引 k 后，以既有 a、m 与候选质量 s 计算选中项对数概率，
        # total 自 0.0 按 t 升序累加；任一结果非有限即失败。
        lp = a[chosen] - m - math.log(s)
        if not math.isfinite(lp):
            raise ValueError("selected log-prob became non-finite")
        total += lp
        if not math.isfinite(total):
            raise ValueError("total log-prob accumulated non-finitely")
        logprobs.append(lp)

        out.append(vocab[chosen])
        x = chosen
        memory.append([float(v) for v in h])

    return "".join(out), logprobs, total


def _sample_lstm_attn_topk(model_path, start, seed_text, start_t_text,
                           end_t_text, top_k_text, length_text, window_text):
    """sample-lstm-attn-top-k：成功返回 LENGTH 个码点再加一个 LF。

    采样与校验全部沿用 _sample_lstm_attn_topk_run，本包装仅取其生成文本。
    """
    text, _logprobs, _total = _sample_lstm_attn_topk_run(
        model_path, start, seed_text, start_t_text, end_t_text, top_k_text,
        length_text, window_text)
    return text + "\n"


def _sample_lstm_attn_topk_scored(model_path, start, seed_text, start_t_text,
                                  end_t_text, top_k_text, length_text,
                                  window_text):
    """sample-lstm-attn-top-k-scored：输出文本、逐步对数概率与累计对数概率。

    全部校验、状态推进、温度退火、top-k 候选截取、随机源初始化与消费、选
    索引规则及错误协议均沿用 sample-lstm-attn-top-k；同参须消费相同随机
    序列并生成与原入口一致的 text。每步选中索引 k 后，以既有 a、m 和候选
    质量 s 计算 lp=a[k]-m-log(s)；total 从 0.0 按 t 升序累加 lp，任一结
    果非有限即失败。stdout 恰为单个 JSON 对象加 LF，键序
    text,logprobs,total_logprob；text 为生成字符串（不含尾随 LF），
    logprobs 为 LENGTH 长字符串列表、第 t 项为 format(lp,'.17g')，
    total_logprob 为 format(total,'.17g')。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'。不写文件。
    """
    text, logprobs, total = _sample_lstm_attn_topk_run(
        model_path, start, seed_text, start_t_text, end_t_text, top_k_text,
        length_text, window_text)
    obj = {
        "text": text,
        "logprobs": [format(lp, ".17g") for lp in logprobs],
        "total_logprob": format(total, ".17g"),
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _sample_lstm_attn_topk_topp_run(model_path, start, seed_text,
                                    start_t_text, end_t_text, top_k_text,
                                    top_p_text, length_text, window_text):
    """sample-lstm-attn-top-k-top-p 与 -scored 共享的采样核心。

    校验、状态推进、温度退火、top-k 截断后 top-p 前缀截取、随机源初始化
    与消费、选索引规则及有限性失败契约均与 _sample_lstm_attn_topk_topp
    文档一致。返回 (text, logprobs, total)：text 为生成字符串；logprobs
    为每步选中索引 k 对应的 a[k]-m-log(s)（s 为最终前缀质量）；total 自
    0.0 按 t 升序累加各 lp。任一 lp 或 total 非有限均抛 ValueError。

    除 TOP_K、TOP_P 及下述选样外，MODEL、START、SEED、LENGTH、WINDOW、
    LSTM 状态、注意力记忆、Why/by logit、稳定 softmax、线性温度退火
    （LENGTH 为 1 时仅用 START_T，为 0 时不计算温度）与有限性失败契约均
    沿用 _sample_lstm_attn_topk；WINDOW 词法与安全截取沿用
    perplexity-lstm-attn。整次调用仅初始化一次
    r=random.Random(int(SEED))，不写任何文件。

    TOP_K 整串匹配 [1-9][0-9]*，且数学值 K 不超过 V=len(vocab)：先按十进
    制位数及同长度字典序与 V 的十进制文本比较，越界即失败，仅通过后才转
    int，任意位数文本都不触发整数转换异常。TOP_P 经 float() 解析，结果须
    有限且 0<TOP_P<=1，否则抛 ValueError。

    每步先按原顺序求温度缩放后的 e_k=exp(a_k-max(a))，d 自 0.0 依 k 升序
    累加。再将索引按 (-e_k, k) 升序排列（e 降序、并列时 k 升序）并取前 K
    项；sK 自 0.0 按该序累加 e，令 target=TOP_P*sK，再从 0.0 按该序累加
    e，保留首个使累计值 >=target 的最短前缀，s 为此前缀累计和。令
    u=r.random()*s，再按前缀序自 0.0 累加 e，选首个累计值严格大于 u 的
    索引；无则取前缀末项。其字符追加到输出并作为下一输入 x，随后向
    memory 追加 h 的 float 副本。任一新增乘法或累加非有限均抛
    ValueError。
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

    # TOP_K：整串匹配 [1-9][0-9]*，且 K<=V。先以十进制位数、同位数字典序
    # 与 str(V) 比较，越界即失败；仅通过后才 int()，任何长度文本都不会触发
    # 整数转换异常（str(V) 受内存约束而位数有界）。
    if not _WINDOW_RE.match(top_k_text):
        raise ValueError("TOP_K must match [1-9][0-9]*")
    v_text = str(V)
    if len(top_k_text) > len(v_text) or (
            len(top_k_text) == len(v_text) and top_k_text > v_text):
        raise ValueError("TOP_K must not exceed len(vocab)")
    top_k = int(top_k_text)

    # TOP_P：float() 可解析且有限，0<TOP_P<=1。
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
    logprobs = []
    total = 0.0

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

        # 索引按 (-e_k, k) 升序：e 降序、并列时 k 升序；取前 K 项。
        order = sorted(range(V), key=lambda k: (-e[k], k))
        candidates = order[:top_k]

        # sK 自 0.0 按候选序累加 e；target=TOP_P*sK 须有限。
        s_k = 0.0
        for idx in candidates:
            s_k += e[idx]
            if not math.isfinite(s_k):
                raise ValueError("top-k mass accumulated non-finitely")
        target = top_p * s_k
        if not math.isfinite(target):
            raise ValueError("top-p target accumulated non-finitely")

        # 再从 0.0 按候选序累加 e，保留首个累计值 >=target 的最短前缀；s
        # 为此前缀累计和。因累加顺序相同，末项累计恰为 sK>=target，前缀必
        # 存在。
        prefix = []
        s = 0.0
        for idx in candidates:
            prefix.append(idx)
            s += e[idx]
            if not math.isfinite(s):
                raise ValueError("top-p prefix accumulated non-finitely")
            if s >= target:
                break

        # u=r.random()*s；按前缀顺序自 0.0 累加 e，选首个累计值严格大于
        # u 者；无则取前缀末项。
        threshold = rng.random() * s
        if not math.isfinite(threshold):
            raise ValueError("sample threshold became non-finite")
        chosen = prefix[-1]
        cum = 0.0
        for idx in prefix:
            cum += e[idx]
            if not math.isfinite(cum):
                raise ValueError("cumulative probability accumulated "
                                 "non-finitely")
            if cum > threshold:
                chosen = idx
                break

        # 选中索引 k 后，以既有 a、m 与最终前缀质量 s 计算选中项对数概
        # 率，total 自 0.0 按 t 升序累加；任一结果非有限即失败。
        lp = a[chosen] - m - math.log(s)
        if not math.isfinite(lp):
            raise ValueError("selected log-prob became non-finite")
        total += lp
        if not math.isfinite(total):
            raise ValueError("total log-prob accumulated non-finitely")
        logprobs.append(lp)

        out.append(vocab[chosen])
        x = chosen
        memory.append([float(v) for v in h])

    return "".join(out), logprobs, total


def _sample_lstm_attn_topk_topp(model_path, start, seed_text, start_t_text,
                                end_t_text, top_k_text, top_p_text,
                                length_text, window_text):
    """sample-lstm-attn-top-k-top-p：成功返回 LENGTH 个码点再加一个 LF。

    采样与校验全部沿用 _sample_lstm_attn_topk_topp_run，本包装仅取其生成
    文本。
    """
    text, _logprobs, _total = _sample_lstm_attn_topk_topp_run(
        model_path, start, seed_text, start_t_text, end_t_text, top_k_text,
        top_p_text, length_text, window_text)
    return text + "\n"


def _sample_lstm_attn_topk_topp_scored(model_path, start, seed_text,
                                       start_t_text, end_t_text, top_k_text,
                                       top_p_text, length_text, window_text):
    """sample-lstm-attn-top-k-top-p-scored：文本、逐步对数概率与累计值。

    全部校验、状态推进、温度退火、top-k 截断后 top-p 前缀截取、随机源初
    始化与消费、选索引规则及错误协议均沿用 sample-lstm-attn-top-k-top-p；
    同参须消费相同随机序列并生成与原入口一致的 text。每步选中索引 k 后，
    以既有 a、m 和最终前缀质量 s 计算 lp=a[k]-m-log(s)；total 从 0.0 按
    t 升序累加 lp，任一结果非有限即失败。stdout 恰为单个 JSON 对象加
    LF，键序 text,logprobs,total_logprob；text 为生成字符串（不含尾随
    LF），logprobs 为 LENGTH 长字符串列表、第 t 项为
    format(lp,'.17g')，total_logprob 为 format(total,'.17g')；LENGTH 为
    0 时三值依次为 ""、[]、"0"。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'。不写文件。
    """
    text, logprobs, total = _sample_lstm_attn_topk_topp_run(
        model_path, start, seed_text, start_t_text, end_t_text, top_k_text,
        top_p_text, length_text, window_text)
    obj = {
        "text": text,
        "logprobs": [format(lp, ".17g") for lp in logprobs],
        "total_logprob": format(total, ".17g"),
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _sample_lstm_attn_topk_topp_batch(model_path, start, seeds_path,
                                      start_t_text, end_t_text, top_k_text,
                                      top_p_text, length_text, window_text):
    """sample-lstm-attn-top-k-top-p-batch：多 seed 独立采样并汇总 medoid。

    SEEDS 的严格 UTF-8 JSON、非空数组、元素为匹配 _INT_RE（
    0|-?[1-9][0-9]*，禁止空白、+ 前缀、前导零、下划线）的 str 及重复按原
    序保留契约均沿用 sample-lstm-attn-top-p-batch；文件读取、UTF-8/JSON
    解析或元素非法即按既有协议失败。MODEL、START、START_T、END_T、TOP_K、
    TOP_P、LENGTH、WINDOW 与其余校验、温度退火、top-k 后 top-p 前缀截
    取、随机源初始化与消费、逐步得分、total 累加及非有限错误协议均沿用
    sample-lstm-attn-top-k-top-p-scored。

    按 SEEDS 原序逐项独立调用 _sample_lstm_attn_topk_topp_run，每项逐值
    等同于以该 seed 单独调用 sample-lstm-attn-top-k-top-p-scored 入口，
    项间不共享任何随机源或采样状态。令 d_i=Σ_j H(text_i,text_j)，H 为两
    串逐码点不等位置数（较短串长度之外的位置均计为不等），j 按原序自 0
    累加；medoid 按 (d_i,-total_i,i) 升序取首项下标，total 使用累加所得
    的未格式化 float。不写任何文件。

    stdout 恰为单个 JSON 对象加 LF，键序 runs,medoid；runs[i] 恰为
    [SEEDS[i],text_i,logs_i,format(total_i,'.17g'),d_i]，SEEDS[i] 为原始
    种子 str，text_i 为生成字符串（不含尾随 LF），logs_i 为逐步
    format(lp,'.17g') 字符串列表，d_i 为 int；medoid 为所选 int 下标。
    序列化恰用 json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n' 的 UTF-8 字节。
    """
    # SEEDS：严格 UTF-8 的 JSON 非空数组，元素为匹配整数词法的 str。
    with open(seeds_path, "rb") as f:
        seeds = json.loads(f.read().decode("utf-8"))
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("SEEDS must be a non-empty JSON array")
    for seed_text in seeds:
        if type(seed_text) is not str or not _INT_RE.match(seed_text):
            raise ValueError(
                "each seed must match 0|-?[1-9][0-9]*")

    runs = []
    texts = []
    totals = []
    for seed_text in seeds:
        # 每项均以独立调用采样，随机源在核心内按 seed 新建，项间不共享
        # 任何状态；返回值逐值等同于单独调用 -scored 入口。
        text, logprobs, total = _sample_lstm_attn_topk_topp_run(
            model_path, start, seed_text, start_t_text, end_t_text,
            top_k_text, top_p_text, length_text, window_text)
        runs.append([
            seed_text,
            text,
            [format(lp, ".17g") for lp in logprobs],
            format(total, ".17g"),
            0,
        ])
        texts.append(text)
        totals.append(total)

    n = len(texts)
    for i in range(n):
        # H(text_i,text_j)：等长部分逐码点比较，长度差位置全部计不等；
        # d_i 按 j 原序自 0 累加。
        d_i = 0
        ti = texts[i]
        for tj in texts:
            if len(ti) >= len(tj):
                longer, shorter = ti, tj
            else:
                longer, shorter = tj, ti
            dij = len(longer) - len(shorter)
            for k, ch in enumerate(shorter):
                if ch != longer[k]:
                    dij += 1
            d_i += dij
        runs[i][4] = d_i

    # medoid 按 (d_i,-total_i,i) 升序取首项；total 用未格式化 float。
    medoid = min(range(n), key=lambda i: (runs[i][4], -totals[i], i))

    obj = {"runs": runs, "medoid": medoid}
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _sample_lstm_attn_consensus(model_path, start, seeds_path,
                                start_t_text, end_t_text, top_k_text,
                                top_p_text, length_text, window_text):
    """sample-lstm-attn-consensus：多 seed 独立采样、medoid 与逐位共识。

    除输出与下述逐位统计外，SEEDS 校验、文件读取、UTF-8/JSON 解析、
    MODEL、START、START_T、END_T、TOP_K、TOP_P、LENGTH、WINDOW 与其余校
    验、温度退火、top-k 后 top-p 前缀截取、随机源初始化与消费、逐步得
    分、total 累加及非有限错误协议，以及 runs 与 medoid 的构造，均严格沿
    用 sample-lstm-attn-top-k-top-p-batch；同参 runs、medoid 逐值相同。

    令 N 为种子数。每个位置 t 按 vocab 索引 k 升序计数 n，仅输出 n>0 的
    [vocab[k],n]；winner 取 n 最大者，并列取最小 k。令 p=n/N，e 从 0.0
    按 k 升序累加 -p*math.log(p)（跳过 p==0 项），E 从 0.0 按 t 升序累加
    e，任一中间值非有限即抛 ValueError 失败。stdout 为键序
    runs,medoid,positions,consensus,mean_entropy 的紧凑 JSON 加 LF；
    positions 按 t 升序，每项恰为
    [t,counts,winner,format(e,'.17g')]，counts 为上述 [vocab[k],n] 列表，
    winner 为 vocab[k]；consensus 连接各位置 winner；mean_entropy 为
    format(E/LENGTH,'.17g')。LENGTH=0 时 positions、consensus、
    mean_entropy 依次为 []、""、"0"。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n' 的 UTF-8 字节。不写任何文件；失败时 stdout 为
    空，错误协议沿用原 batch。
    """
    # SEEDS：严格 UTF-8 的 JSON 非空数组，元素为匹配整数词法的 str。
    with open(seeds_path, "rb") as f:
        seeds = json.loads(f.read().decode("utf-8"))
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("SEEDS must be a non-empty JSON array")
    for seed_text in seeds:
        if type(seed_text) is not str or not _INT_RE.match(seed_text):
            raise ValueError(
                "each seed must match 0|-?[1-9][0-9]*")

    runs = []
    texts = []
    totals = []
    for seed_text in seeds:
        # 每项均以独立调用采样，随机源在核心内按 seed 新建，项间不共享
        # 任何状态；返回值逐值等同于单独调用 -scored 入口。
        text, logprobs, total = _sample_lstm_attn_topk_topp_run(
            model_path, start, seed_text, start_t_text, end_t_text,
            top_k_text, top_p_text, length_text, window_text)
        runs.append([
            seed_text,
            text,
            [format(lp, ".17g") for lp in logprobs],
            format(total, ".17g"),
            0,
        ])
        texts.append(text)
        totals.append(total)

    n_runs = len(texts)
    for i in range(n_runs):
        # H(text_i,text_j)：等长部分逐码点比较，长度差位置全部计不等；
        # d_i 按 j 原序自 0 累加。
        d_i = 0
        ti = texts[i]
        for tj in texts:
            if len(ti) >= len(tj):
                longer, shorter = ti, tj
            else:
                longer, shorter = tj, ti
            dij = len(longer) - len(shorter)
            for k, ch in enumerate(shorter):
                if ch != longer[k]:
                    dij += 1
            d_i += dij
        runs[i][4] = d_i

    # medoid 按 (d_i,-total_i,i) 升序取首项；total 用未格式化 float。
    medoid = min(range(n_runs), key=lambda i: (runs[i][4], -totals[i], i))

    # LENGTH 经与采样核心相同的词法校验后转 int（0|[1-9][0-9]*）；采样已
    # 先于本处执行，非法 LENGTH 会在核心内失败，故此处恒合法。
    length = int(length_text)

    # 取与采样一致的 vocab，其下标顺序即逐位计数所需的 k 升序。
    vocab = _load_perplexity_lstm_model(model_path)[0]

    # 各生成串等长 LENGTH（采样核心每步恰追加一个码点），据此逐位统计。
    positions = []
    consensus_parts = []
    E = 0.0
    for t in range(length):
        counts_by_ch = {}
        for text in texts:
            ch = text[t]
            counts_by_ch[ch] = counts_by_ch.get(ch, 0) + 1
        counts = []
        winner = None
        winner_n = -1
        e = 0.0
        for ch in vocab:
            cnt = counts_by_ch.get(ch, 0)
            if cnt > 0:
                counts.append([ch, cnt])
                p = cnt / n_runs
                term = -p * math.log(p)
                if not math.isfinite(term):
                    raise ValueError(
                        "per-position entropy term became non-finite")
                e += term
                if not math.isfinite(e):
                    raise ValueError(
                        "per-position entropy accumulated non-finitely")
                if cnt > winner_n:
                    winner_n = cnt
                    winner = ch
        E += e
        if not math.isfinite(E):
            raise ValueError("mean entropy accumulated non-finitely")
        consensus_parts.append(winner)
        positions.append([t, counts, winner, format(e, ".17g")])

    if length == 0:
        positions = []
        consensus = ""
        mean_entropy_text = "0"
    else:
        consensus = "".join(consensus_parts)
        mean_entropy = E / length
        if not math.isfinite(mean_entropy):
            raise ValueError("mean entropy became non-finite")
        mean_entropy_text = format(mean_entropy, ".17g")

    obj = {
        "runs": runs,
        "medoid": medoid,
        "positions": positions,
        "consensus": consensus,
        "mean_entropy": mean_entropy_text,
    }
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def _beam_lstm_attn(model_path, start, start_t_text, end_t_text,
                    beam_text, length_text, window_text):
    """以线性退火温度、带注意力上下文从 LSTM 模型做确定性束搜索。

    除 BEAM 与下述确定性选束外，MODEL、START、LENGTH、WINDOW、LSTM 状态、
    注意力记忆、Why/by logit、稳定 softmax、线性温度退火（LENGTH 为 1 时
    仅用 START_T，为 0 时不计算温度）与有限性失败契约均沿用
    _sample_lstm_attn_anneal；WINDOW 词法与安全截取沿用
    perplexity-lstm-attn。本命令无 SEED、无随机源且不写任何文件。

    BEAM 整串匹配 [1-9][0-9]*（任意位数均合法），否则抛 ValueError；每轮
    仅当其数学值小于候选数时转 int（此时其位数不超过候选数十进制位数，
    不触发整数文本位数上限），否则保留全部候选。

    初始束为 (0.0, "", h0, c0, START 索引, [h0])，生成索引元组为空。第 t
    轮逐束按既有温度推进 h、c 并求缩放 logit a、m=max(a)、
    e[k]=exp(a[k]-m)，d 自 0.0 按 k 升序累加；每个 k 生成子束：分数加
    a[k]-m-log(d)，文本追加 vocab[k]，置 x=k，memory 追加 h 的 float
    副本。全部子束按 (-分数, 生成索引元组) 升序排列，保留前
    min(BEAM, 候选数) 项；同轮等长，使用原始累计分数，长度归一化因子固定
    为 1。任一新增运算非有限均抛 ValueError。LENGTH 为 0 时仅输出 LF，
    否则输出最终首束文本加一个 LF。相同输入输出逐字节相同。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    # BEAM：整串匹配 [1-9][0-9]*（任意位数均合法，不预先转 int）。
    if not _WINDOW_RE.match(beam_text):
        raise ValueError("BEAM must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

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

    # 初始束 (0.0, "", h0, c0, START 索引, [h0])；h0 保留模型原值，
    # attention 与 LSTMCell.forward 均不修改其输入，故可直接共享。indices
    # 为与各束并行的生成索引元组（仅用于排序的确定性决胜）。
    beams = [(0.0, "", h0, c0, vocab.index(start), [h0])]
    indices = [()]

    for t in range(length):
        temperature = temperature_at(t)
        children = []
        child_indices = []
        for bi in range(len(beams)):
            score, text, h, c, x, memory = beams[bi]

            # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
            xvec = [0.0] * V
            xvec[x] = 1.0
            nh, nc = cell.forward(xvec, h, c)[:2]

            M = _window_tail(memory, window_text)
            u = _attn_context(nh, M)

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

            ld = math.log(d)
            if not math.isfinite(ld):
                raise ValueError("log softmax denominator became non-finite")

            # 子束共享父束推进后的 nh、nc 与追加后的 memory（均不被修改）。
            new_memory = memory + [[float(v) for v in nh]]
            base_idx = indices[bi]
            for k in range(V):
                step = a[k] - m - ld
                if not math.isfinite(step):
                    raise ValueError("beam score step became non-finite")
                child_score = score + step
                if not math.isfinite(child_score):
                    raise ValueError("beam score became non-finite")
                children.append((child_score, text + vocab[k], nh, nc, k,
                                 new_memory))
                child_indices.append(base_idx + (k,))

        # 全部子束按 (-分数, 生成索引元组) 升序；同轮等长，使用原始累计
        # 分数（长度归一化因子固定为 1）。
        order = sorted(range(len(children)),
                       key=lambda i: (-children[i][0], child_indices[i]))

        # 仅当 BEAM 的数学值小于候选数时转 int（位数不超过 str(候选数)，
        # 不触发整数文本位数上限），否则保留全部候选。
        n_candidates = len(children)
        limit_text = str(n_candidates)
        if len(beam_text) < len(limit_text) or (
                len(beam_text) == len(limit_text)
                and beam_text < limit_text):
            keep = int(beam_text)
        else:
            keep = n_candidates
        beams = [children[i] for i in order[:keep]]
        indices = [child_indices[i] for i in order[:keep]]

    return beams[0][1] + "\n"


def _beam_lstm_attn_topk_topp_run(model_path, start, start_t_text, end_t_text,
                                  top_k_text, top_p_text, beam_text,
                                  length_text, window_text):
    """top-k 截断再 top-p 截取的确定性束搜索核心，返回最终 (beams, indices)。

    除 TOP_K、TOP_P 及下述候选截取外，MODEL、START、LENGTH、WINDOW、LSTM
    状态、注意力记忆、Why/by logit、稳定 softmax、线性温度退火（LENGTH 为
    1 时仅用 START_T，为 0 时不计算温度）与有限性失败契约均沿用
    _beam_lstm_attn；WINDOW 词法与安全截取沿用 perplexity-lstm-attn。本
    命令无 SEED、无随机源且不写任何文件。

    TOP_K 整串匹配 [1-9][0-9]*，且数学值 K 不超过 V=len(vocab)：先按十进
    制位数及同长度字典序与 V 的十进制文本比较，越界即失败，仅通过后才转
    int，任意位数文本都不触发整数转换异常。TOP_P 经 float() 解析，结果须
    有限且 0<TOP_P<=1，否则抛 ValueError。BEAM 整串匹配 [1-9][0-9]*（任
    意位数均合法），每轮仅当其数学值小于候选数时转 int，否则保留全部候
    选。

    初始束为 (0.0, "", h0, c0, START 索引, [h0])，生成索引元组为空。第 t
    轮逐束按既有顺序求缩放 logit a、m=max(a)、e[k]=exp(a[k]-m)；索引按
    (-e[k], k) 升序排列并取前 K 项，sK 自 0.0 按该序累加 e，令
    target=TOP_P*sK，再从 0.0 按该序累加 e，取累计值首次 >=target 的最短
    前缀，s 为其累计和。仅为前缀中每个 k 生成子束：分数加
    a[k]-m-log(s)，文本追加 vocab[k]，置 x=k，memory 追加 h 的 float 副
    本。上述乘加、log 与分数非有限均抛 ValueError。全部子束按
    (-分数, 生成索引元组) 升序排列，保留前 min(BEAM, 候选数) 项；长度归
    一化因子固定为 1。搜索结束后返回 (beams, indices)：LENGTH 为 0 时仅
    含初始束，beams[0][1] 为空串。相同输入结果逐字节确定。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    # BEAM：整串匹配 [1-9][0-9]*（任意位数均合法，不预先转 int）。
    if not _WINDOW_RE.match(beam_text):
        raise ValueError("BEAM must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0, c0 = _load_perplexity_lstm_model(model_path)
    V = len(vocab)
    H = len(h0)

    # START：恰为词表内的一个码点。
    if type(start) is not str or len(start) != 1 or start not in vocab:
        raise ValueError("START must be a single in-vocab codepoint")

    # START_T、END_T：float() 可解析且有限、严格大于 0。
    start_t = float(start_t_text)
    if not math.isfinite(start_t) or start_t <= 0.0:
        raise ValueError("START_T must be a finite positive float")
    end_t = float(end_t_text)
    if not math.isfinite(end_t) or end_t <= 0.0:
        raise ValueError("END_T must be a finite positive float")

    # TOP_K：整串匹配 [1-9][0-9]*，且 K<=V。先以十进制位数、同位数字典序
    # 与 str(V) 比较，越界即失败；仅通过后才 int()，任何长度文本都不会触发
    # 整数转换异常（str(V) 受内存约束而位数有界）。
    if not _WINDOW_RE.match(top_k_text):
        raise ValueError("TOP_K must match [1-9][0-9]*")
    v_text = str(V)
    if len(top_k_text) > len(v_text) or (
            len(top_k_text) == len(v_text) and top_k_text > v_text):
        raise ValueError("TOP_K must not exceed len(vocab)")
    top_k = int(top_k_text)

    # TOP_P：float() 可解析且有限，0<TOP_P<=1。
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

    # 初始束 (0.0, "", h0, c0, START 索引, [h0])；h0 保留模型原值，
    # attention 与 LSTMCell.forward 均不修改其输入，故可直接共享。indices
    # 为与各束并行的生成索引元组（仅用于排序的确定性决胜）。
    beams = [(0.0, "", h0, c0, vocab.index(start), [h0])]
    indices = [()]

    for t in range(length):
        temperature = temperature_at(t)
        children = []
        child_indices = []
        for bi in range(len(beams)):
            score, text, h, c, x, memory = beams[bi]

            # 当前字符的 V 长 one-hot 输入，推进 LSTM 隐状态与细胞状态。
            xvec = [0.0] * V
            xvec[x] = 1.0
            nh, nc = cell.forward(xvec, h, c)[:2]

            M = _window_tail(memory, window_text)
            u = _attn_context(nh, M)

            # logit 仅以 u 替代 h；下标与累加顺序同 _sample_lstm。
            z = _output_logits(Why, by, u)

            # a_k=z_k/T，m=max(a)，e_k=exp(a_k-m)（求值顺序同
            # _beam_lstm_attn）。
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
            for k in range(V):
                try:
                    ek = math.exp(a[k] - m)
                except OverflowError:
                    raise ValueError("softmax exp overflowed")
                if not math.isfinite(ek):
                    raise ValueError("softmax exp became non-finite")
                e[k] = ek

            # 索引按 (-e_k, k) 升序：e 降序、并列时 k 升序；取前 K 项。
            order = sorted(range(V), key=lambda k: (-e[k], k))
            candidates = order[:top_k]

            # sK 自 0.0 按候选序累加 e；target=TOP_P*sK 须有限。
            s_k = 0.0
            for idx in candidates:
                s_k += e[idx]
                if not math.isfinite(s_k):
                    raise ValueError("top-k mass accumulated non-finitely")
            target = top_p * s_k
            if not math.isfinite(target):
                raise ValueError("top-p target became non-finite")

            # 再从 0.0 按候选序累加 e，取累计值首次 >=target 的最短前缀；
            # s 为其累计和。因累加顺序相同，末项累计恰为 sK>=target，前缀
            # 必存在。
            prefix = []
            s = 0.0
            for idx in candidates:
                prefix.append(idx)
                s += e[idx]
                if not math.isfinite(s):
                    raise ValueError("top-p prefix accumulated non-finitely")
                if s >= target:
                    break

            ls = math.log(s)
            if not math.isfinite(ls):
                raise ValueError("log top-p mass became non-finite")

            # 仅为前缀中每个 k 生成子束；子束共享父束推进后的 nh、nc 与追
            # 加后的 memory（均不被修改），其余更新沿用 _beam_lstm_attn。
            new_memory = memory + [[float(v) for v in nh]]
            base_idx = indices[bi]
            for k in prefix:
                step = a[k] - m - ls
                if not math.isfinite(step):
                    raise ValueError("beam score step became non-finite")
                child_score = score + step
                if not math.isfinite(child_score):
                    raise ValueError("beam score became non-finite")
                children.append((child_score, text + vocab[k], nh, nc, k,
                                 new_memory))
                child_indices.append(base_idx + (k,))

        # 全部子束按 (-分数, 生成索引元组) 升序；同轮等长，使用原始累计
        # 分数（长度归一化因子固定为 1）。
        order = sorted(range(len(children)),
                       key=lambda i: (-children[i][0], child_indices[i]))

        # 仅当 BEAM 的数学值小于候选数时转 int（位数不超过 str(候选数)，
        # 不触发整数文本位数上限），否则保留全部候选。
        n_candidates = len(children)
        limit_text = str(n_candidates)
        if len(beam_text) < len(limit_text) or (
                len(beam_text) == len(limit_text)
                and beam_text < limit_text):
            keep = int(beam_text)
        else:
            keep = n_candidates
        beams = [children[i] for i in order[:keep]]
        indices = [child_indices[i] for i in order[:keep]]

    return beams, indices


def _beam_lstm_attn_topk_topp(model_path, start, start_t_text, end_t_text,
                              top_k_text, top_p_text, beam_text, length_text,
                              window_text):
    """beam-lstm-attn-topk-topp：返回最终首束文本加一个 LF。

    搜索过程（含全部校验、候选裁剪、累计分数与索引元组决胜）严格沿用
    _beam_lstm_attn_topk_topp_run；LENGTH 为 0 时最终仅初始束，首束文本
    为空串，故仅输出 LF。
    """
    beams, _indices = _beam_lstm_attn_topk_topp_run(
        model_path, start, start_t_text, end_t_text, top_k_text, top_p_text,
        beam_text, length_text, window_text)
    return beams[0][1] + "\n"


def _beam_lstm_attn_topk_topp_nbest(model_path, start, start_t_text,
                                    end_t_text, top_k_text, top_p_text,
                                    beam_text, n_text, length_text,
                                    window_text):
    """beam-lstm-attn-topk-topp-nbest：输出最终束前 N 个候选（JSON 行）。

    除 N 及输出外，全部参数校验与逐轮搜索行为严格沿用
    beam-lstm-attn-topk-topp：候选裁剪、累计分数与索引元组决胜完全一致，
    无 SEED、无随机源且不写文件。

    N 整串匹配 [1-9][0-9]*（任意位数合法），否则抛 ValueError。最终束形
    成后，先按十进制位数及同长度字典序比较 N 与最终束数：仅当 N 数学上
    较小时才转 int 并取前 N 束（此时其位数不超过束数十进制位数，不触发
    整数文本位数上限），否则取全部束；超长 N 文本不触发整数转换异常。

    按最终束既有顺序，每个候选独占一个 JSON 行，其值仅为生成文本，恰以
    json.dumps(text, ensure_ascii=True, separators=(',',':'),
    allow_nan=False)+'\\n' 序列化；各行直接拼接，末行保留 LF。LENGTH 为 0
    时最终仅初始束（文本为空串），故输出一个空字符串 JSON 行。
    """
    # N：整串匹配 [1-9][0-9]*（任意位数均合法，不预先转 int）。
    if not _WINDOW_RE.match(n_text):
        raise ValueError("N must match [1-9][0-9]*")

    beams, _indices = _beam_lstm_attn_topk_topp_run(
        model_path, start, start_t_text, end_t_text, top_k_text, top_p_text,
        beam_text, length_text, window_text)

    # 以十进制位数及同长度字典序与最终束数比较；仅当 N 较小时才转 int。
    n_beams = len(beams)
    limit_text = str(n_beams)
    if len(n_text) < len(limit_text) or (
            len(n_text) == len(limit_text) and n_text < limit_text):
        keep = int(n_text)
    else:
        keep = n_beams

    parts = []
    for beam in beams[:keep]:
        parts.append(json.dumps(beam[1], ensure_ascii=True,
                                separators=(",", ":"), allow_nan=False))
    return "\n".join(parts) + "\n"


def _beam_lstm_attn_topk_topp_nbest_scored(model_path, start, start_t_text,
                                           end_t_text, top_k_text, top_p_text,
                                           beam_text, n_text, length_text,
                                           window_text):
    """beam-lstm-attn-topk-topp-nbest-scored：输出最终束前 N 个候选及其分数。

    除输出外，全部参数校验、逐轮搜索、候选裁剪、累计分数、索引元组决胜
    及 N 的超长十进制处理均严格沿用 beam-lstm-attn-topk-topp-nbest；无
    SEED、无随机源且不写文件。

    按最终束既有顺序取前 min(N, 束数) 项，每项独占一个 JSON 对象行，键
    序恰为 text、score：值分别为生成字符串与累计分数的
    format(score, '.17g') 字符串（负零写作 "-0"）。每行恰由
    json.dumps(obj, ensure_ascii=True, separators=(',',':'),
    allow_nan=False)+'\\n' 生成，各行直接拼接且末行保留 LF，不输出额外
    空白。LENGTH 为 0 时最终仅初始束，故唯一一行恰为
    {"text":"","score":"0"}\\n。
    """
    # N：整串匹配 [1-9][0-9]*（任意位数均合法，不预先转 int）。
    if not _WINDOW_RE.match(n_text):
        raise ValueError("N must match [1-9][0-9]*")

    beams, _indices = _beam_lstm_attn_topk_topp_run(
        model_path, start, start_t_text, end_t_text, top_k_text, top_p_text,
        beam_text, length_text, window_text)

    # 以十进制位数及同长度字典序与最终束数比较；仅当 N 较小时才转 int。
    n_beams = len(beams)
    limit_text = str(n_beams)
    if len(n_text) < len(limit_text) or (
            len(n_text) == len(limit_text) and n_text < limit_text):
        keep = int(n_text)
    else:
        keep = n_beams

    parts = []
    for beam in beams[:keep]:
        obj = {"text": beam[1], "score": format(beam[0], ".17g")}
        parts.append(json.dumps(obj, ensure_ascii=True,
                                separators=(",", ":"), allow_nan=False))
    return "\n".join(parts) + "\n"


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


def _train_lstm_attn(model_path, corpus_path, out_path, window_text):
    """带注意力上下文对 LSTM 模型做一次全语料 SGD 更新并写入 OUT。

    MODEL、CORPUS 完全沿用 train-lstm（perplexity-lstm version 2 的八键、
    F、严格 UTF-8、形状、词表及语料至少 2 码点契约）；WINDOW 整串匹配
    [1-9][0-9]*（任意位数均合法，不转 int），安全截取沿用
    perplexity-lstm-attn，否则抛 ValueError。

    前向严格复用 perplexity-lstm-attn：置 h=h0、c=c0、memory=[h0]，t 升序
    以当前字符的 V 长 one-hot 调用装入 W、b 的 LSTMCell.forward 更新 h、c
    并缓存每步 cache；M_t 取 memory 末尾至多 WINDOW 项（顺序从旧到新），
    u_t 逐项等于 h 加 attention([h], M_t, M_t, None) 的首行上下文，随后向
    memory 追加 h 的 float 副本；logit 仅以 u_t 替代隐状态，其余下标、
    稳定 softmax 顺序均与 train-lstm 相同。

    令 g_t = p_t - onehot(y_t)，按 t 升序累加 dWhy += g_t⊗u_t、dby += g_t
    （组内下标顺序同 train-lstm），并对每个 j 自 0.0 按 k 升序求
    du_t = Whyᵀg_t。置 dhs 为全零，按 t 降序调用
    attention_context_backward(h_t, M_t, du_t)：dn 按 i 升序加至 dhs[t]；
    再按 M_t 从旧到新、i 升序把 dmemory 各行映射到完整记忆
    [h0, h_0, ..., h_{t-1}]——首行若为 h0 则梯度丢弃，第 q（q>=1）行加至
    dhs[q-1]。任一累加非有限抛 ValueError。随后调用
    backward_sequence(dhs, caches) 取得整条序列的 dW、db；梯度组序 dW、
    db、dWhy、dby，5.0 全局裁剪、0.1 更新、h0/c0 不变、OUT 键序/形状/
    紧凑 JSON 加 LF 及负零口径均沿用 train-lstm。
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

    T = len(ids) - 1

    cell = LSTMCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    # 前向严格复用 perplexity-lstm-attn：t 升序以 (h0, c0) 为初态逐步
    # 推进 h、c 并缓存每步 cache；M_t 为 memory 末尾至多 WINDOW 项，
    # u_t = h + 首行上下文，随后向 memory 追加 h 的 float 副本。
    hs = []
    caches = []
    m_list = []
    u_list = []
    memory = [h0]
    h = list(h0)
    c = list(c0)
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0
        h, c, cache = cell.forward(x, h, c)
        hs.append(h)
        caches.append(cache)
        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)
        m_list.append(M)
        u_list.append(u)
        memory.append([float(v) for v in h])

    dWhy = [[0.0] * H for _ in range(V)]
    dby = [0.0] * V
    du_list = [None] * T

    for t in range(T):
        u = u_list[t]
        y = ids[t + 1]

        # logit：仅以 u 替代 h，其余与 train-lstm 相同。
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

        # 按 t 升序累加 dWhy += g⊗u、dby += g（组内顺序同 train-lstm）。
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
        # [h0, h_0, ..., h_{t-1}] 中的下标；0 即 h0（梯度丢弃），
        # q>=1 对应 h_{q-1}。
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


def _train_lstm_mha(model_path, corpus_path, out_path, window_text):
    """带投影多头注意力对 LSTM 模型做一次全语料 SGD 更新并写入 OUT。

    MODEL、CORPUS 完全沿用 perplexity-lstm-mha version 4 的十三键、F、严格
    UTF-8、形状、词表及语料至少 2 码点契约；WINDOW 整串匹配
    [1-9][0-9]*（任意位数均合法，不转 int），安全截取沿用
    perplexity-lstm-mha，否则抛 ValueError。

    前向严格复用 perplexity-lstm-mha：置 h=h0、c=c0、memory=[h0]，t 升序
    以当前字符的 V 长 one-hot 调用装入 W、b 的 LSTMCell.forward 更新 h、c
    并缓存每步 cache；M_t 取 memory 末尾至多 WINDOW 项（顺序从旧到新），
    装入四组投影构造 MHA(H, heads)，u_t 逐项等于 h 加
    forward_cross([h], M_t, None) 的首行上下文，随后向 memory 追加 h 的
    float 副本；logit 仅以 u_t 替代隐状态，其余下标、稳定 softmax 顺序均
    与 train-lstm 相同。

    令 g_t = p_t - onehot(y_t)，按 t 升序累加 dWhy += g_t⊗u_t、dby += g_t
    （组内下标顺序同 train-lstm），并对每个 j 自 0.0 按 k 升序求
    du_t = Whyᵀg_t。置 dhs 为全零，按 t 降序调用
    mha.backward_cross([du_t]) 得 dqx、dkvx 与四组投影梯度：dqx 仅一行，
    残差直连与查询路径之和 du_t + dqx[0] 按 i 升序加至 dhs[t]；dkvx 再按
    M_t 从旧到新、i 升序映射到完整记忆 [h0, h_0, ..., h_{t-1}]——首行
    若为 h0 则梯度丢弃，第 q（q>=1）行加至 dhs[q-1]；dWq、dWk、dWv、dWo
    各自 0.0 起按 t 降序、组内行（a）列（j）序累加。任一累加非有限抛
    ValueError。随后调用 LSTMCell.backward_sequence(dhs, caches) 取得
    dW、db。梯度组序依次为 dW、db、dWq、dWk、dWv、dWo、dWhy、dby，
    5.0 全局裁剪、0.1 更新、h0/c0/heads 不变；OUT 键序恰为
    version、vocab、W、b、Wq、Wk、Wv、Wo、heads、Why、by、h0、c0
    （version 为 int 4，heads 沿用原 int，其余数组元素均转为 float），
    紧凑 JSON 加 LF 及负零口径均沿用 train-lstm。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")

    (vocab, W, b, Wq, Wk, Wv, Wo, heads,
     Why, by, h0, c0) = _load_perplexity_lstm_mha_model(model_path)
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

    mha = MHA(H, heads)
    mha.Wq = [list(row) for row in Wq]
    mha.Wk = [list(row) for row in Wk]
    mha.Wv = [list(row) for row in Wv]
    mha.Wo = [list(row) for row in Wo]

    # 前向严格复用 perplexity-lstm-mha：t 升序以 (h0, c0) 为初态逐步
    # 推进 h、c 并缓存每步 cache；M_t 为 memory 末尾至多 WINDOW 项，
    # u_t = h + 首行交叉注意力上下文，随后向 memory 追加 h 的 float 副本。
    hs = []
    caches = []
    m_list = []
    u_list = []
    cross_caches = []
    memory = [h0]
    h = list(h0)
    c = list(c0)
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0
        h, c, cache = cell.forward(x, h, c)
        hs.append(h)
        caches.append(cache)
        M = _window_tail(memory, window_text)
        u = _mha_cross_context(h, M, mha)
        m_list.append(M)
        u_list.append(u)
        # 每步 forward_cross 都会替换 mha 的缓存；逐份留存快照，反向按 t
        # 装回（backward_cross 成功后保留缓存而不推进）。
        cross_caches.append(mha._cache)
        memory.append([float(v) for v in h])

    dWhy = [[0.0] * H for _ in range(V)]
    dby = [0.0] * V
    du_list = [None] * T

    for t in range(T):
        u = u_list[t]
        y = ids[t + 1]

        # logit：仅以 u 替代 h，其余与 train-lstm 相同。
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

        # 按 t 升序累加 dWhy += g⊗u、dby += g（组内顺序同 train-lstm）。
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

    # 交叉注意力残差与记忆的反向：dhs 置零，按 t 降序累加；四组投影梯度
    # 各自 0.0 起按 t 降序、行列序累加。
    dhs = [[0.0] * H for _ in range(T)]
    dWq = [[0.0] * H for _ in range(H)]
    dWk = [[0.0] * H for _ in range(H)]
    dWv = [[0.0] * H for _ in range(H)]
    dWo = [[0.0] * H for _ in range(H)]
    for t in range(T - 1, -1, -1):
        du = du_list[t]
        # 装回该步前向留存的快照（同 backward_cross_padded 的用法），
        # backward_cross 成功后保留缓存而不推进，故每步均须显式装回。
        mha._cache = cross_caches[t]
        dqx, dkvx, step_dWq, step_dWk, step_dWv, step_dWo = \
            mha.backward_cross([du])

        # 残差直连 du 与查询路径 dqx[0] 之和按 i 升序加至 dhs[t]。
        dqx0 = dqx[0]
        dh_row = dhs[t]
        for i in range(H):
            dh_row[i] += du[i] + dqx0[i]
            if not math.isfinite(dh_row[i]):
                raise ValueError("dhs accumulated to a non-finite value")

        # dkvx 按 M_t 从旧到新映射：base 为其首行在完整记忆
        # [h0, h_0, ..., h_{t-1}] 中的下标；0 即 h0（梯度丢弃），
        # q>=1 对应 h_{q-1}。
        base = t + 1 - len(m_list[t])
        for p, dm_row in enumerate(dkvx):
            q = base + p
            if q == 0:
                continue
            target = dhs[q - 1]
            for i in range(H):
                target[i] += dm_row[i]
                if not math.isfinite(target[i]):
                    raise ValueError("dhs accumulated to a non-finite value")

        # 四组投影梯度按 t 降序、行（a）列（j）序累加。
        for a in range(H):
            sum_q, sum_k, sum_v, sum_o = dWq[a], dWk[a], dWv[a], dWo[a]
            step_q, step_k = step_dWq[a], step_dWk[a]
            step_v, step_o = step_dWv[a], step_dWo[a]
            for j in range(H):
                sum_q[j] += step_q[j]
                if not math.isfinite(sum_q[j]):
                    raise ValueError("dWq accumulated to a non-finite value")
                sum_k[j] += step_k[j]
                if not math.isfinite(sum_k[j]):
                    raise ValueError("dWk accumulated to a non-finite value")
                sum_v[j] += step_v[j]
                if not math.isfinite(sum_v[j]):
                    raise ValueError("dWv accumulated to a non-finite value")
                sum_o[j] += step_o[j]
                if not math.isfinite(sum_o[j]):
                    raise ValueError("dWo accumulated to a non-finite value")

    _dxs, _dh0, _dc0, dW, db = cell.backward_sequence(dhs, caches)

    # 依 dW、db、dWq、dWk、dWv、dWo、dWhy、dby 行序累加平方和求全局范数。
    sum_sq = 0.0
    for group in (dW, db, dWq, dWk, dWv, dWo, dWhy, dby):
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

    # 八组参数减 0.1 倍（裁剪后的）梯度；h0、c0、heads 不变。
    # 结果非有限即失败。
    def _updated(old, grad):
        value = float(old) - 0.1 * (grad * scale)
        if not math.isfinite(value):
            raise ValueError("updated parameter became non-finite")
        return value

    new_W = [[_updated(W[k][j], dW[k][j]) for j in range(V + H)]
             for k in range(4 * H)]
    new_b = [_updated(b[k], db[k]) for k in range(4 * H)]
    new_Wq = [[_updated(Wq[a][j], dWq[a][j]) for j in range(H)]
              for a in range(H)]
    new_Wk = [[_updated(Wk[a][j], dWk[a][j]) for j in range(H)]
              for a in range(H)]
    new_Wv = [[_updated(Wv[a][j], dWv[a][j]) for j in range(H)]
              for a in range(H)]
    new_Wo = [[_updated(Wo[a][j], dWo[a][j]) for j in range(H)]
              for a in range(H)]
    new_Why = [[_updated(Why[k][j], dWhy[k][j]) for j in range(H)]
               for k in range(V)]
    new_by = [_updated(by[k], dby[k]) for k in range(V)]

    obj = {
        "version": 4,
        "vocab": vocab,
        "W": new_W,
        "b": new_b,
        "Wq": new_Wq,
        "Wk": new_Wk,
        "Wv": new_Wv,
        "Wo": new_Wo,
        "heads": heads,
        "Why": new_Why,
        "by": new_by,
        "h0": [float(v) for v in h0],
        "c0": [float(v) for v in c0],
    }
    text = json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False) + "\n"
    with open(out_path, "wb") as f:
        f.write(text.encode("utf-8"))


def _train_gru_attn_tbptt(model_path, corpus_path, out_path, window_text,
                          k_text):
    """带注意力上下文与截断 BPTT 对 GRU 模型做一次全语料 SGD 并写入 OUT。

    MODEL、CORPUS、OUT 完全沿用 train-gru（perplexity-gru version 3 的七
    键、F、严格 UTF-8、形状、词表及语料至少 2 码点契约）；WINDOW 的词法
    与任意位数安全截取完全沿用 perplexity-gru-attn。K 整串匹配
    [1-9][0-9]*（任意位数均合法）；令 T 为预测步数，按十进制位数及同长
    字典序与 T 比较求 Ke=min(K,T)，全程不先把超长 K 转为 int，否则抛
    ValueError。

    前向完全沿用 perplexity-gru-attn：置 h=h0、memory=[h0]，t 升序以当前
    字符的 V 长 one-hot 调用装入 W、b 的 GRUCell.forward 更新 h 并缓存每步
    cache；M_t 取 memory 末尾至多 WINDOW 项（顺序从旧到新），u_t 逐项等于
    h 加 attention([h], M_t, M_t, None) 的首行上下文，随后向 memory 追加 h
    的 float 副本。

    令 g_t = p_t-onehot(y_t)，按 train-gru 的 t/k/j 次序以 u_t 累加
    dWhy、dby，并对每个 j 自 0.0 按 k 升序求 du_t = Whyᵀg_t。置 dhs 为全
    零，按 t 降序调用 attention_context_backward(h_t, M_t, du_t)：dn 按 i
    升序加至 dhs[t]；dmemory 第 p 行令 q=t+1-len(M_t)+p，q=0 时梯度丢
    弃，否则 s=q-1，仅当 (T-1-s)//Ke 与 (T-1-t)//Ke 为同一截断块时，才按
    p、i 升序把该行加至 dhs[s]（跨块的记忆梯度一律丢弃）。任一累加非有
    限抛 ValueError。随后调用
    GRUCell.backward_sequence(dhs, caches, None, Ke) 取得 dW、db；梯度组
    序 dW、db、dWhy、dby，5.0 全局裁剪、0.1 更新、h0 不变与 OUT 写出均
    沿用 train-gru。成功时 stdout 为空并返回 0。
    """
    if not _WINDOW_RE.match(window_text):
        raise ValueError("WINDOW must match [1-9][0-9]*")
    if not _WINDOW_RE.match(k_text):
        raise ValueError("K must match [1-9][0-9]*")

    vocab, W, b, Why, by, h0 = _load_perplexity_gru_model(model_path)
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

    # Ke=min(K,T)：K 为无界文本，先按位数、同长按字典序与 str(T) 比较，
    # 仅当 K<T（其位数必不超过受内存约束的 str(T)）才转 int。
    t_limit = str(T)
    if len(k_text) > len(t_limit) or (
            len(k_text) == len(t_limit) and k_text >= t_limit):
        Ke = T
    else:
        Ke = int(k_text)

    cell = GRUCell(V, H)
    cell.W = [list(row) for row in W]
    cell.b = list(b)

    # 前向完全沿用 perplexity-gru-attn：t 升序以 h0 为初态逐步推进 h 并
    # 缓存每步 cache；M_t 为 memory 末尾至多 WINDOW 项，u_t = h + 首行
    # 上下文，随后向 memory 追加 h 的 float 副本。
    hs = []
    caches = []
    m_list = []
    u_list = []
    memory = [h0]
    h = list(h0)
    for t in range(T):
        x = [0.0] * V
        x[ids[t]] = 1.0
        h, cache = cell.forward(x, h)
        hs.append(h)
        caches.append(cache)
        M = _window_tail(memory, window_text)
        u = _attn_context(h, M)
        m_list.append(M)
        u_list.append(u)
        memory.append([float(v) for v in h])

    dWhy = [[0.0] * H for _ in range(V)]
    dby = [0.0] * V
    du_list = [None] * T

    for t in range(T):
        u = u_list[t]
        y = ids[t + 1]

        # logit：仅以 u 替代 h，其余与 train-gru 相同。
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

        # 按 t 升序累加 dWhy += g⊗u、dby += g（组内顺序同 train-gru）。
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

        # dmemory 第 p 行对应完整记忆 [h0, h_0, ..., h_{t-1}] 的第 q 行：
        # q=0 即 h0（梯度丢弃）；q>=1 对应 h_s（s=q-1）。末端对齐的截断
        # 块号为 (T-1)//Ke；仅同一块内的记忆梯度才加至 dhs[s]，跨块丢弃。
        base = t + 1 - len(m_list[t])
        block_t = (T - 1 - t) // Ke
        for p, dm_row in enumerate(dmemory):
            q = base + p
            if q == 0:
                continue
            s = q - 1
            if (T - 1 - s) // Ke != block_t:
                continue
            target = dhs[s]
            for i in range(H):
                target[i] += dm_row[i]
                if not math.isfinite(target[i]):
                    raise ValueError("dhs accumulated to a non-finite value")

    _dxs, _dh0, dW, db = cell.backward_sequence(dhs, caches, None, Ke)

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

    # 四组参数减 0.1 倍（裁剪后的）梯度；h0 不变。结果非有限即失败。
    def _updated(old, grad):
        value = float(old) - 0.1 * (grad * scale)
        if not math.isfinite(value):
            raise ValueError("updated parameter became non-finite")
        return value

    new_W = [[_updated(W[k][j], dW[k][j]) for j in range(V + H)]
             for k in range(3 * H)]
    new_b = [_updated(b[k], db[k]) for k in range(3 * H)]
    new_Why = [[_updated(Why[k][j], dWhy[k][j]) for j in range(H)]
               for k in range(V)]
    new_by = [_updated(by[k], dby[k]) for k in range(V)]

    obj = {
        "version": 3,
        "vocab": vocab,
        "W": new_W,
        "b": new_b,
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

    python seqmodel.py perplexity-gru MODEL CORPUS：MODEL 为 version 3 的
    GRU 模型（键 version、vocab、W、b、Why、by、h0，W 形状 3H×(V+H)、
    b 形状 3H），以 GRUCell.forward(x, h) 逐步推进隐状态，logit 与
    log-sum-exp 规则同 perplexity；输出契约与 perplexity 相同，不写任何
    文件。

    python seqmodel.py perplexity-gru-attn MODEL CORPUS WINDOW：MODEL 沿用
    perplexity-gru 的 version 3 七键、F、形状与严格 UTF-8 契约；WINDOW
    沿用 perplexity-lstm-attn 的词法及任意位数安全截取。置 h=h0、
    memory=[h0]，t 升序以当前字符 one-hot 调用装入 W、b 的
    GRUCell.forward 更新 h，再以 memory 末尾至多 WINDOW 项（从旧到新）
    为键/值调用 attention，用 h 与首行上下文之和作为 logit 隐状态，随后向
    memory 追加 h 的 float 副本；输出契约与 perplexity 相同，不写文件。

    python seqmodel.py train-lstm MODEL CORPUS OUT：对 version 2 的 LSTM
    模型做一次全语料 SGD 并写出新模型。t 升序以 (h0, c0) 为初态调用装入
    W、b 的 LSTMCell.forward 并缓存；输出层 g、dWhy、dby、dhs 沿用 train
    的公式与累加顺序（隐状态换为 LSTM 的 h），再以 backward_sequence 取得
    dW、db；梯度组序为 dW、db、dWhy、dby，全局范数、5.0 裁剪与 0.1 更新
    规则同 train，仅更新 W、b、Why、by，h0、c0 不变。成功时 stdout 为空
    并返回 0。

    python seqmodel.py train-gru MODEL CORPUS OUT：对 version 3 的 GRU
    模型做一次全语料 SGD 并写出新模型。t 升序以 h0 为初态调用装入 W、b
    的 GRUCell.forward 并缓存；输出层 softmax、g=p-onehot(y)、dWhy、dby、
    dhs 沿用 train-lstm 的公式与 t/k/j 累加顺序（隐状态换为 GRU 的 h），
    再以 GRUCell.backward_sequence(dhs, caches) 取得 dW、db；梯度组序为
    dW、db、dWhy、dby，全局范数、5.0 裁剪与 0.1 更新规则同 train-lstm，
    仅更新 W、b、Why、by，h0 不变。OUT 顶层键恰为 version、vocab、W、b、
    Why、by、h0（version 为 int 3）。成功时 stdout 为空并返回 0。

    python seqmodel.py sample MODEL START SEED TEMPERATURE LENGTH：成功时
    stdout 恰为 LENGTH 个采样码点的 UTF-8 编码再加一个 LF（LENGTH 为 0 时
    仅 LF），不写任何文件，返回 0。

    python seqmodel.py sample-lstm MODEL START SEED TEMPERATURE LENGTH：
    MODEL 为 version 2 的 LSTM 模型（键 version、vocab、W、b、Why、by、
    h0、c0），置 h=h0、c=c0、x=START 索引，每步以 x 的 V 长 one-hot 调用
    装入 W、b 的 LSTMCell.forward 推进 h、c，logit、温度缩放、稳定
    softmax 与抽样规则同 sample；输出契约与 sample 相同，不写任何文件。

    python seqmodel.py sample-gru MODEL START SEED TEMPERATURE LENGTH：
    MODEL 沿用 perplexity-gru 的 version 3 七键顺序、F、形状及严格 UTF-8
    契约（键 version、vocab、W、b、Why、by、h0，W 形状 3H×(V+H)、
    b 形状 3H），置 h=list(h0)、x=START 索引，每步以 x 的 V 长 one-hot
    调用装入 W、b 的 GRUCell.forward 取首项更新 h，logit、温度缩放、稳定
    softmax 与抽样规则同 sample；输出契约与 sample 相同，不写任何文件。

    python seqmodel.py sample-gru-attn MODEL START SEED TEMPERATURE LENGTH
    WINDOW：MODEL、START、SEED、TEMPERATURE、LENGTH 沿用 sample-gru；
    WINDOW 沿用 perplexity-gru-attn 的词法及任意位数安全截取。置
    h=h0、x=START 索引、memory=[h0]，每步以 x 的 one-hot 调用装入 W、b
    的 GRUCell.forward 更新 h，再以 memory 末尾至多 WINDOW 项（从旧到新）
    为键/值调用 attention，用 h 与首行上下文之和作为 logit 隐状态，按
    perplexity-gru-attn 的顺序计算 Why/by 仿射；温度缩放、稳定 softmax、
    单次随机源与词表升序抽样沿用 sample-gru；生成字符作为下一 x，再向
    memory 追加 h 的 float 副本；输出契约与 sample 相同，不写任何文件。

    python seqmodel.py sample-anneal MODEL START SEED START_T END_T LENGTH：
    以线性退火温度采样，第 t 步温度为
    START_T+(END_T-START_T)*t/(LENGTH-1)（LENGTH 为 1 时仅用 START_T，
    为 0 时不计算温度）；输出契约与 sample 相同。

    python seqmodel.py sample-gru-attn-anneal MODEL START SEED START_T
    END_T LENGTH WINDOW：MODEL、START、SEED、LENGTH、WINDOW 的校验及
    GRU 状态、注意力、logit、稳定 softmax 与抽样顺序均沿用
    sample-gru-attn；START_T、END_T 各经 float() 解析，须有限且严格
    大于 0；整次仅初始化一次随机源。LENGTH 为 0 时不计算温度，为 1 时
    仅用 START_T，否则第 t 步温度严格按
    START_T+(END_T-START_T)*t/(LENGTH-1) 求值且每步须有限、大于 0；
    每步以该温度替换固定温度，沿用同一 h、x、memory，生成后更新 x 并
    向 memory 追加 h 的 float 副本；输出契约与 sample 相同，不写任何
    文件。

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

    python seqmodel.py perplexity-lstm-attn-trace MODEL CORPUS WINDOW：
    MODEL、CORPUS、WINDOW 的校验与逐步计算均沿用 perplexity-lstm-attn。
    令 T=语料码点数-1；按 t 升序计算 loss=m+log(d)-z_y，L 从 0.0 依序
    累加 loss。stdout 恰为一个 JSON 对象加 LF：顶层键序恰为
    version,items,total_nll,perplexity；version 为 int 1；items 为 T 长
    列表，第 t 项键序恰为 t,target,nll，值依次为 int t、下一单码点
    str、format(loss,'.17g') 字符串；total_nll、perplexity 依次为
    format(L,'.17g')、format(exp(L/T),'.17g') 字符串。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'，不写文件。

    python seqmodel.py eval-windows MODEL CORPUS WINDOWS：MODEL、CORPUS
    及单窗口计算均沿用 perplexity-lstm-attn。WINDOWS 为逗号分隔的非空
    窗口列表，各项整串匹配 [1-9][0-9]*，空项非法，超长项按既有规则
    安全截取，重复项按序保留。每项均以 h=h0、c=c0、memory=[h0] 重置；
    设 T=语料码点数-1，L 为按 t 升序所得总负对数似然。stdout 依次
    输出 JSON 行，键序 window,steps,total_logprob,perplexity，值依次
    为原窗口 str、T（int）、format(-L,'.17g')、
    format(exp(L/T),'.17g')。每行恰由 json.dumps(obj,
    ensure_ascii=True,separators=(',',':'),allow_nan=False)+'\\n'
    生成的 UTF-8 字节组成，末行保留 LF；失败时不输出任何部分行，
    不写文件。

    python seqmodel.py window-sensitivity-lstm-attn MODEL CORPUS SHORT
    LONG：MODEL、CORPUS、状态推进、窗口截取与单步负对数似然均沿用
    perplexity-lstm-attn。SHORT、LONG 各须整串匹配 [1-9][0-9]*
    （任意位数均合法，不转 int）且数学值 SHORT<LONG（按十进制位数及
    同长字典序比较），否则失败。短、长两窗均从 h=h0、c=c0、
    memory=[h0] 独立重置。令 T=语料码点数-1，第 t 步短、长窗 NLL
    依次为 s、l，delta=s-l，total_delta 从 0.0 按 t 升序累加 delta；
    任一计算非有限即失败。stdout 先写 T 个 JSON 项行，键序
    t,target,short_nll,long_nll,delta，值依次为 int t、下一单码点
    str、format(s,'.17g')、format(l,'.17g')、format(delta,'.17g')；
    再写唯一汇总行，键序 steps,total_delta，值依次为 int T 与
    format(total_delta,'.17g')。每行恰由 json.dumps(obj,
    ensure_ascii=True,separators=(',',':'),allow_nan=False)+'\\n'
    生成的 UTF-8 字节组成，末行保留 LF；失败时不输出任何部分行，
    不写文件。

    python seqmodel.py compare-lstm-attn MODEL_A MODEL_B CORPUS WINDOW：
    MODEL_A、MODEL_B 各自沿用 perplexity-lstm-attn 的 version 2 八键、
    F、形状与严格 UTF-8 契约，两者 vocab 须逐项同序相等，H 可不同；
    CORPUS 沿用其严格 UTF-8、词表及至少 2 码点契约；WINDOW 沿用其
    [1-9][0-9]* 词法及任意位数安全截取。两轨各从自身 h0、c0、
    memory=[h0] 重置，状态推进、注意力、logit 与稳定 log-sum-exp 均沿
    用 perplexity-lstm-attn。令 T=语料码点数-1，第 t 步两轨 NLL 依次为
    a、b，d=b-a，total_delta 从 0.0 按 t 升序累加 d；任一计算非有限即
    失败。stdout 先写 T 个 JSON 项行，键序
    t,target,a_nll,b_nll,delta，值依次为 int t、下一单码点 str、
    format(a,'.17g')、format(b,'.17g')、format(d,'.17g')；再写唯一汇总
    行，键序 steps,total_delta，值依次为 int T 与
    format(total_delta,'.17g')。每行恰由 json.dumps(obj,
    ensure_ascii=True,separators=(',',':'),allow_nan=False)+'\\n'
    生成的 UTF-8 字节组成，末行保留 LF；失败时不输出任何部分行，
    不写文件。

    python seqmodel.py compare-suite MODEL_A MODEL_B LIST：MODEL_A、
    MODEL_B 的契约及每项的两轨独立重置与逐步 NLL 计算均沿用
    compare-lstm-attn。LIST 为严格 UTF-8 的 JSON 非空数组，每项恰为
    [corpus,window] 两个 str：corpus 为非空相对路径并按 LIST 父目录
    解析，window 整串匹配 [1-9][0-9]*（任意位数安全处理），重复项按
    序保留。每项令 D=0.0 按 t 升序累加 b_nll-a_nll，D 正/负/零时
    winner 为 A/B/tie；G=0.0 按项序累加 D，非有限即失败。stdout 恰为
    单个 JSON 对象加 LF，键序 items,summary：items 按清单序，每项为
    [corpus,window,steps,format(D,'.17g'),winner]（steps 为 int）；
    summary 为 [groups,a_wins,b_wins,ties,format(G,'.17g'),winner]
    （前四项为 int，winner 按 G 同规则）。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'；失败时不输出任何部分行，不写文件。

    python seqmodel.py rank-suite MODELS LIST：MODELS 为严格 UTF-8 的
    JSON 数组且至少 2 项，每项须为非空相对路径 str（按 MODELS 父目录
    解析），重复项按序保留。LIST 的结构、语料路径、WINDOW 及重复项
    契约沿用 compare-suite。模型读取与逐项计算沿用 compare-lstm-attn：
    所有模型 vocab 须同序相等，H 可不同；每个模型在每项均从自身 h0、
    c0、memory=[h0] 重置。每项 NLL 总和 L 从 0.0 按 t 升序累加；模型
    总分 S 从 0.0 按 LIST 序累加 L，任一计算非有限即失败。按
    (S,MODELS 原下标) 升序排名，S 以 float 精确比较。stdout 恰为单个
    JSON 对象加 LF，顶层唯一键 ranking；其值为排名后的数组，每项恰为
    [path,total_nll]，依次为 MODELS 原文 str 与
    format(S,'.17g') 字符串。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n'；失败时不输出任何部分行，不写文件。

    python seqmodel.py rank-suite-details MODELS LIST：MODELS、LIST 及模
    型/语料/WINDOW 的读取、校验、相对路径和重复保序均沿用 rank-suite
    （至少 2 个模型，vocab 同序，H 可不同），既有接口不变。按模型原下
    标 mi、清单原下标 gi 升序遍历，每项从该模型的 h0、c0、
    memory=[h0] 重置并沿用其逐步 NLL。L 从 0.0 按 t 升序累加；S 从
    0.0 按 gi 升序累加未格式化 L，非有限抛 ValueError。stdout 恰为键
    序 items,ranking 的单个 JSON 对象加 LF。items 按 (mi,gi) 展平，每
    项恰为 [model,corpus,window,steps,item_nll]，前三项是 MODELS/LIST
    原文 str，steps 是语料码点数减 1 的 int，item_nll 是
    format(L,'.17g')；ranking 按 (S,mi) 升序，每项恰为
    [model,format(S,'.17g')]，须等于同参 rank-suite 的 ranking。序列化
    与失败协议沿用 rank-suite；失败不部分输出、不写文件。

    python seqmodel.py rank-window-sensitivity MODELS LIST BASE：MODELS、
    LIST 及模型/语料/WINDOW 的读取、校验、相对路径和重复保序均沿用
    rank-suite-details；BASE 整串匹配 [1-9][0-9]*，任意位数安全处理。
    按模型原下标 mi、清单原下标 gi 升序遍历，每项分别以该项 window 与
    BASE 为窗口，从该模型 h0、c0、memory=[h0] 独立重置并沿用其逐步
    NLL，总量 L、B 各从 0.0 按 t 升序累加。令 delta=L-B；每个模型的
    敏感度 A 从 0.0 按 gi 升序累加 abs(delta)，任一计算非有限即失败。
    stdout 恰为键序 items,ranking 的单个 JSON 对象加 LF。items 按
    (mi,gi) 展平，每项恰为
    [model,corpus,window,steps,item_nll,base_nll,delta]，前三项是
    MODELS/LIST 原文 str，steps 是语料码点数减 1 的 int，后三项依次
    为 format(L,'.17g')、format(B,'.17g')、format(delta,'.17g')；
    ranking 按 (-A,mi) 升序（A 以 float 精确比较，下标决胜保序），每
    项恰为 [model,format(A,'.17g')]。序列化与失败协议沿用
    rank-suite-details；失败不部分输出、不写文件。

    python seqmodel.py rank-window-stability MODELS LIST BASES：MODELS、
    LIST 及模型/语料/WINDOW 的读取、校验、相对路径和重复保序均沿用
    rank-window-sensitivity，既有接口不变。BASES 为严格 UTF-8 的 JSON
    非空数组，每项 type 恰为 str 且整串匹配 [1-9][0-9]*（任意位数安全
    处理），重复项按序保留，否则失败。对每个 base（按 BASES 原序），
    按 mi、gi 升序严格沿用 rank-window-sensitivity 计算敏感度 A：每项
    以原 window 与该 base 独立重置，A 从 0.0 按 gi 累加未格式化的
    abs(L-B)，非有限即失败。各 base 按 (-A,mi) 升序排名，名次 r 从 1
    起。对每个模型令 Q=max(r)-min(r)，R 从 0 按 base 序累加 r。stdout
    恰为键序 bases,ranking 的单个 JSON 对象加 LF：bases 按 BASES 原
    序，每项恰为 [base,rows]，rows 按名次升序、每项恰为
    [model,format(A,'.17g'),r]；ranking 按 (Q,R,mi) 升序，每项恰为
    [model,Q,R]，Q、R 为 int。序列化、尾 LF、失败原子性及不写文件均
    沿用 rank-window-sensitivity。

    python seqmodel.py rank-window-stability-details MODELS LIST BASES：
    MODELS、LIST、BASES 及模型/语料/WINDOW 的读取、校验、相对路径解
    析、重复保序、状态重置、逐步 NLL 与成败协议均沿用
    rank-window-stability，既有接口不变。按原下标 (bi,mi,gi) 升序遍
    历，每项以原 window 与第 bi 个 base 独立重置求总 NLL L、B；令
    d=L-B、c=abs(d)，A[bi,mi] 从 0.0 按 gi 累加未格式化的 c，任一结
    果非有限即失败。各 bi 按 (-A,mi) 升序排名，名次 r 从 1 起；每模
    型 Q=max(r)-min(r)，R 从 0.0 按 bi 累加 r，最终按 (Q,R,mi) 升
    序。stdout 恰为键序 items,ranking 的单个 JSON 对象加 LF：items
    按 (bi,mi,gi) 展平，每项恰为
    [base,model,corpus,window,steps,L,B,d,c]，前四项为输入原文 str，
    steps 为 int，L、B、d、c 均为 format(x,'.17g')；ranking 按最终名
    次，每项恰为 [model,stats,Q,R]，stats 按 BASES 原序、每项恰为
    [base,format(A,'.17g'),r]，Q、R 为 int。序列化、尾 LF、失败原子
    性及不写文件均沿用 rank-window-stability。

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

    python seqmodel.py score-lstm-attn MODEL START TEXT TEMPERATURE
    WINDOW：确定性评分指定候选 TEXT。MODEL、START、TEMPERATURE、WINDOW
    与 CLI 协议沿用 sample-lstm-attn；TEXT 为可空 Unicode 字符串且每个
    码点须在 vocab，否则失败。无 SEED、无随机源，不写文件。置 h=h0、
    c=c0、x=START 索引、memory=[h0]、total=0.0，按 t 升序遍历 TEXT：
    每步以 x 的 one-hot 推进 h、c，经窗口注意力得到 u，按原顺序计算
    a[k]=z[k]/TEMPERATURE、m=max(a)、e[k]=exp(a[k]-m)，d 自 0.0 依 k
    升序累加 e[k]；不抽样，令 y 为 TEXT[t] 索引，
    lp=a[y]-m-math.log(d)，total 按 t 升序累加 lp，置 x=y，memory 追加
    h 的 float 副本。任一计算非有限即失败。成功时 stdout 恰为
    format(total,'.17g')+'\\n' 的 ASCII 字节（空 TEXT 为 "0\\n"），
    stderr 为空并返回 0。

    python seqmodel.py score-lstm-attn-batch MODEL START CANDIDATES
    TEMPERATURE WINDOW：确定性评分 CANDIDATES 文件中的多个候选串。
    MODEL、START、TEMPERATURE、WINDOW 与 CLI 协议沿用 score-lstm-attn；
    CANDIDATES 为严格 UTF-8 的 JSON 文件，顶层须为非空数组，元素为可空
    str 且每个码点须在 vocab，否则失败。无 SEED、无随机源，不写文件。
    每个候选均独立重置 h=h0、c=c0、x=START 索引、memory=[h0]、
    total=0.0，候选间不共享状态；逐步 one-hot 推进、窗口注意力、
    a[k]=z[k]/TEMPERATURE、m=max(a)、d 自 0.0 依 k 升序累加
    exp(a[k]-m)、lp=a[y]-m-math.log(d) 及 total 累加规则均沿用
    score-lstm-attn，任一计算非有限即失败。成功时 stdout 恰为单个 JSON
    对象加 LF，键序 items,best：items 按 CANDIDATES 原序，每项恰为
    [text,steps,format(total,'.17g')]，steps 为码点数 int；best 为最高
    total 的 int 下标，平分取较小下标。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n' 的 ASCII 字节，stderr 为空并返回 0。

    python seqmodel.py score-lstm-attn-temps MODEL START CANDIDATES TEMPS
    WINDOW：确定性评分 CANDIDATES 文件中多个候选串在多个温度下的得分。
    除 TEMPS、遍历及输出外，MODEL、START、CANDIDATES、WINDOW 与 CLI 协议
    沿用 score-lstm-attn-batch；TEMPS 为严格 UTF-8 的 JSON 非空数组，元素
    type 恰为 str，其 float() 值须有限且严格大于 0，重复项按序保留，非
    法即按既有协议失败。按 (ti,ci) 升序遍历温度 ti 与候选 ci，每组均从
    h=h0、c=c0、x=START 索引、memory=[h0]、total=0.0 独立重置，组间不共
    享状态，逐步 one-hot 推进、窗口注意力、温度缩放、稳定 softmax 与
    total 累加均沿用 score-lstm-attn-batch，任一计算非有限即失败。成功
    时 stdout 恰为单个 JSON 对象加 LF，键序 temperatures,items,best：
    temperatures 为 TEMPS 原序的 format(temp,'.17g') 字符串列表；items 按
    (ti,ci) 展平，每项恰为 [ti,ci,text,steps,format(total,'.17g')]，
    ti、ci、steps 为 int，text 为候选原文 str；best 为最高 total 的
    [ti,ci]，平分时取较小 ti 再取较小 ci。序列化、尾 LF、成败字节协议
    均沿用 score-lstm-attn-batch，不写文件。

    python seqmodel.py sample-lstm-attn-anneal MODEL START SEED START_T
    END_T LENGTH WINDOW：以线性退火温度、带注意力上下文从 version 2 的
    LSTM 模型采样。除温度外，MODEL、START、SEED、LENGTH、WINDOW、LSTM
    状态、注意力记忆、logit、稳定 softmax 与抽样规则均沿用
    sample-lstm-attn；温度退火规则同 sample-anneal（LENGTH 为 1 时仅用
    START_T，为 0 时不计算温度，否则第 t 步为
    START_T+(END_T-START_T)*t/(LENGTH-1) 且每步须有限并大于 0）；整次
    仅初始化一次随机源；输出契约与 sample 相同，不写文件。

    python seqmodel.py sample-gru-attn-anneal MODEL START SEED START_T
    END_T LENGTH WINDOW：以线性退火温度、带注意力上下文从 version 3 的
    GRU 模型采样。MODEL、START、SEED、LENGTH、WINDOW 的校验，以及 GRU
    状态、注意力、logit、稳定 softmax 与抽样顺序均沿用
    sample-gru-attn；START_T、END_T 各经 float() 解析，须有限且严格
    大于 0；整次仅初始化一次 random.Random(int(SEED))。LENGTH 为 0 时
    不计算温度，为 1 时仅用 START_T，否则 t 升序，第 t 步温度严格按
    START_T+(END_T-START_T)*t/(LENGTH-1) 求值且每步须有限并大于 0；
    每步以该温度替换固定温度，沿用同一 h、x、memory，生成后更新 x 并
    向 memory 追加 h 的 float 副本；任一中间量非有限即失败。输出契约
    与 sample 相同，不写文件。

    python seqmodel.py sample-gru-attn-top-p MODEL START SEED START_T
    END_T TOP_P LENGTH WINDOW：除 TOP_P 及核选样外，参数校验、LENGTH 的
    0/1 语义、GRU 状态、注意力记忆、线性温度、Why/by logit、稳定
    softmax 与有限性失败契约均沿用 sample-gru-attn-anneal。TOP_P 经
    float() 解析，须有限且 0<TOP_P<=1，否则失败。每步先按原顺序算
    e_k=exp(a_k-max(a))，d 自 0.0 依 k 升序累加，再将索引按 (-e_k,k)
    升序排列，依此自 0.0 累加 e，截取首个使累计值 >=TOP_P*d 的最短前
    缀，s 为其按该序自 0.0 累加所得之和；整次仅初始化一次
    random.Random(int(SEED))，每步令 u=random()*s，再按前缀顺序自 0.0
    累加 e，选首个累计值严格大于 u 的索引，无则选前缀末项，其字符作
    为下一输入；输出契约与 sample 相同，不写文件。

    python seqmodel.py sample-gru-attn-top-k MODEL START SEED START_T
    END_T TOP_K LENGTH WINDOW：除 TOP_K 及候选选样外，参数校验、LENGTH
    的 0/1 语义、GRU 状态、注意力记忆、线性温度、Why/by logit、稳定
    softmax、随机源生命周期与有限性失败契约均沿用 sample-gru-attn-top-p
    （整次仅构造一次 random.Random(int(SEED))）。TOP_K 整串匹配
    [1-9][0-9]* 且数学值不超过 V=len(vocab)：先按十进制位数、同长字典
    序与 V 比较，越界即失败，仅通过后才转 int，超长文本不触发转换异
    常。每步先按原顺序算 e_k=exp(a_k-max(a))，d 自 0.0 依 k 升序累
    加，再将索引按 (-e_k,k) 升序排列，候选恰为前 K 项，s 自 0.0 按候
    选序累加 e；每步令 u=random()*s，再按候选序自 0.0 累加 e，选首个
    累计值严格大于 u 的索引，无则取候选末项，其字符作为下一输入；输出
    契约与 sample 相同，不写文件。

    python seqmodel.py sample-gru-attn-top-k-top-p MODEL START SEED
    START_T END_T TOP_K TOP_P LENGTH WINDOW：除 TOP_P 及 top-k 之后的
    top-p 前缀筛选外，参数校验、LENGTH 的 0/1 语义、GRU 状态、注意力记
    忆、线性温度、Why/by logit、稳定 softmax、随机源生命周期与有限性失
    败契约均沿用 sample-gru-attn-top-k。TOP_K 仍匹配 [1-9][0-9]* 且不
    超过 V，以位数及同长字典序比较后才转 int；TOP_P 经 float() 解析，
    须有限且 0<TOP_P<=1，否则失败。每步沿原顺序求 e，索引按 (-e_k,k)
    升序取前 K 项；sK 从 0.0 依序累加，令 target=TOP_P*sK，再从 0.0
    累加 e，保留首次使累计值 >=target 的最短前缀，s 为其累计和。每步
    令 u=random()*s，从 0.0 按前缀序累加 e，选首个累计值严格大于 u 的
    索引，无则取末项；字符、x、h、memory 按原入口更新。任一新增乘法
    或累加非有限即失败。输出契约与 sample 相同，不写文件。

    python seqmodel.py sample-lstm-attn-top-p MODEL START SEED START_T
    END_T TOP_P LENGTH WINDOW：除 TOP_P 及核选样外，参数校验、LENGTH 的
    0/1 语义、线性温度、LSTM 状态、注意力记忆、Why/by logit、稳定
    softmax 与有限性失败契约均沿用 sample-lstm-attn-anneal。TOP_P 经
    float() 解析，须有限且 0<TOP_P<=1。每步先按原顺序算 e_k=exp(a_k-
    max(a))，d 自 0.0 依 k 升序累加，再将索引按 (-e_k,k) 升序排列，依
    此自 0.0 累加 e，截取首个使累计值 >=TOP_P*d 的最短前缀，s 为其按
    该序自 0.0 累加所得之和；整次仅初始化一次 random.Random(int(SEED))，
    每步令 u=random()*s，再按前缀顺序自 0.0 累加 e，选首个累计值严格
    大于 u 的索引，无则选前缀末项，其字符作为下一输入；输出契约与
    sample 相同，不写文件。

    python seqmodel.py sample-lstm-attn-top-p-scored MODEL START SEED
    START_T END_T TOP_P LENGTH WINDOW：全部校验、状态推进、温度退火、
    top-p 前缀、随机选中及错误协议均沿用 sample-lstm-attn-top-p；同参须
    消费相同随机序列并生成与原入口一致的 text。每步选中索引 k 后，以既
    有 a、m 和前缀质量 s 计算 lp=a[k]-m-log(s)；total 从 0.0 按 t 升序
    累加 lp，任一结果非有限即失败。stdout 恰为单个 JSON 对象加 LF，键序
    text,logprobs,total_logprob；text 为生成字符串，logprobs 为 LENGTH
    长字符串列表，第 t 项为 format(lp,'.17g')，total_logprob 为
    format(total,'.17g')。序列化恰用 json.dumps(obj,ensure_ascii=True,
    separators=(',',':'),allow_nan=False)+'\\n' 的 UTF-8 字节。返回码、
    stderr 及不写文件行为均沿用原入口。

    python seqmodel.py sample-lstm-attn-top-p-batch MODEL START SEEDS
    START_T END_T TOP_P LENGTH WINDOW：SEEDS 为严格 UTF-8 的 JSON 非空
    数组文件，每个元素 type 恰为 str 且整串匹配 0|-?[1-9][0-9]*，重复项
    按原序保留；文件读取、UTF-8/JSON 解析与元素非法均按既有错误协议失
    败。其余校验、解码、逐步得分与错误协议均沿用
    sample-lstm-attn-top-p-scored。按 SEEDS 原序逐项独立采样，每项逐值
    等同于以该 seed 单独调用原入口，项间不共享随机源或任何状态。令
    d_i=Σ_j H(text_i,text_j)，H 为两串逐码点不等位置数，j 按原序累
    加；medoid 按 (d_i,-total_i,i) 升序取首项下标，total 使用未格式化
    float。stdout 恰为单个 JSON 对象加 LF，键序 runs,medoid；
    runs[i] 恰为 [SEEDS[i],text_i,logs_i,format(total_i,'.17g'),d_i]，
    logs_i 为逐步 format(lp,'.17g') 字符串列表，medoid 为所选 int 下
    标。序列化恰用 json.dumps(obj,ensure_ascii=True,
    separators=(',',':'),allow_nan=False)+'\\n' 的 UTF-8 字节。失败时
    stdout 为空且不写文件。

    python seqmodel.py sample-lstm-attn-top-k MODEL START SEED START_T
    END_T TOP_K LENGTH WINDOW：除 TOP_K 及候选截取外，参数校验、LENGTH
    的 0/1 语义、线性温度、LSTM 状态、注意力记忆、Why/by logit、稳定
    softmax 与有限性失败契约均沿用 sample-lstm-attn-top-p。TOP_K 整串
    匹配 [1-9][0-9]* 且数学值 K 不超过 V=len(vocab)：先按十进制位数及
    同长度字典序与 V 比较，越界即失败，仅通过后转 int，任意位数文本不
    得触发整数转换异常。每步先按原顺序算 e_k=exp(a_k-max(a))，索引按
    (-e_k,k) 升序，候选恰为前 K 项，s 从 0.0 按候选序累加 e；整次仅初
    始化一次 random.Random(int(SEED))，每步令 u=random()*s，再按候选
    序自 0.0 累加 e，选首个累计值严格大于 u 的索引，无则取候选末项，
    其字符作为下一输入；输出契约与 sample 相同，不写文件。

    python seqmodel.py sample-lstm-attn-top-k-scored MODEL START SEED
    START_T END_T TOP_K LENGTH WINDOW：全部校验、状态推进、温度退火、
    top-k 候选、随机消费及错误协议均沿用 sample-lstm-attn-top-k；同参须
    消费相同随机序列并生成与原入口一致的 text。每步候选质量 s 按
    (-e[k],k) 升序前 K 项从 0.0 累加；选中索引 k 后，以既有 a、m 和候
    选质量 s 计算 lp=a[k]-m-log(s)；total 从 0.0 按 t 升序累加 lp，任
    一结果非有限即失败。stdout 恰为单个 JSON 对象加 LF，键序
    text,logprobs,total_logprob；text 为生成字符串（不含 LF），
    logprobs 为 LENGTH 长字符串列表、第 t 项为 format(lp,'.17g')，
    total_logprob 为 format(total,'.17g')；LENGTH 为 0 时三值依次为
    ""、[]、"0"。序列化恰用 json.dumps(obj,ensure_ascii=True,
    separators=(',',':'),allow_nan=False)+'\\n' 的 UTF-8 字节。返回
    码、stderr 及不写文件行为均沿用原入口。

    python seqmodel.py sample-lstm-attn-top-k-top-p MODEL START SEED
    START_T END_T TOP_K TOP_P LENGTH WINDOW：除 TOP_K、TOP_P 及候选截取
    外，参数校验、LENGTH 的 0/1 语义、线性温度、LSTM 状态、注意力记忆、
    Why/by logit、稳定 softmax 与有限性失败契约均沿用
    sample-lstm-attn-top-k。TOP_K 的词法、与 V 的十进制比较及延迟 int
    转换同 sample-lstm-attn-top-k；TOP_P 经 float() 解析，须有限且
    0<TOP_P<=1。每步先按原顺序算 e_k=exp(a_k-max(a))，索引按 (-e_k,k)
    升序并取前 K 项，sK 自 0.0 按该序累加 e，令 target=TOP_P*sK，再从
    0.0 按该序累加 e，保留首个使累计值 >=target 的最短前缀，s 为此前缀
    累计和；整次仅初始化一次 random.Random(int(SEED))，每步令
    u=random()*s，再按前缀序自 0.0 累加 e，选首个累计值严格大于 u 的索
    引，无则取前缀末项，其字符作为下一输入；任一新增乘法或累加非有限
    即失败；输出契约与 sample 相同，不写文件。

    python seqmodel.py sample-lstm-attn-top-k-top-p-scored MODEL START
    SEED START_T END_T TOP_K TOP_P LENGTH WINDOW：全部校验、状态推进、
    温度退火、top-k 后 top-p 截取、随机消费及错误协议均沿用
    sample-lstm-attn-top-k-top-p；同参须消费相同随机序列并生成与原入
    口一致的 text。每步选中索引 k 后，以既有 a、m 和最终前缀质量 s 计
    算 lp=a[k]-m-log(s)；total 从 0.0 按 t 升序累加 lp，任一结果非有
    限即失败。stdout 恰为单个 JSON 对象加 LF，键序
    text,logprobs,total_logprob；text 为生成字符串（不含 LF），
    logprobs 为 LENGTH 长字符串列表、第 t 项为 format(lp,'.17g')，
    total_logprob 为 format(total,'.17g')；LENGTH 为 0 时三值依次为
    ""、[]、"0"。序列化恰用 json.dumps(obj,ensure_ascii=True,
    separators=(',',':'),allow_nan=False)+'\\n' 的 UTF-8 字节。返回
    码、stderr 及不写文件行为均沿用原入口。

    python seqmodel.py sample-lstm-attn-top-k-top-p-batch MODEL START
    SEEDS START_T END_T TOP_K TOP_P LENGTH WINDOW：SEEDS 为严格 UTF-8 的
    JSON 非空数组文件，每个元素 type 恰为 str 且整串匹配
    0|-?[1-9][0-9]*，重复项按原序保留；文件读取、UTF-8/JSON 解析与元素
    非法均按既有错误协议失败。其余校验、温度退火、top-k 后 top-p 截取、
    随机消费、逐步得分与错误协议均沿用
    sample-lstm-attn-top-k-top-p-scored。按 SEEDS 原序逐项独立采样，每
    项逐值等同于以该 seed 单独调用原入口，项间不共享随机源或任何状态。
    令 d_i=Σ_j H(text_i,text_j)，H 为两串逐码点不等位置数（较短串长度
    之外的位置均计为不等），j 按原序累加；medoid 按 (d_i,-total_i,i) 升
    序取首项下标，total 使用未格式化 float。stdout 恰为单个 JSON 对象加
    LF，键序 runs,medoid；runs[i] 恰为
    [SEEDS[i],text_i,logs_i,format(total_i,'.17g'),d_i]，logs_i 为逐步
    format(lp,'.17g') 字符串列表，medoid 为所选 int 下标。序列化恰用
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n' 的 UTF-8 字节。失败时 stdout 为空且不写文件。

    python seqmodel.py sample-lstm-attn-consensus MODEL START SEEDS
    START_T END_T TOP_K TOP_P LENGTH WINDOW：除输出与逐位统计外，全部校验
    与逐项采样严格沿用 sample-lstm-attn-top-k-top-p-batch；同参 runs、
    medoid 逐值相同，原入口不变。令 N 为种子数。每个位置 t 按 vocab 索
    引 k 升序计数 n，仅输出 n>0 的 [vocab[k],n]；winner 取 n 最大者，并
    列取最小 k。令 p=n/N，e 从 0.0 按 k 升序累加 -p*math.log(p)（跳过
    p==0 项），E 从 0.0 按 t 升序累加 e，任一中间值非有限即失败。stdout
    为键序 runs,medoid,positions,consensus,mean_entropy 的紧凑 JSON 加
    LF；positions 按 t 升序，每项恰为
    [t,counts,winner,format(e,'.17g')]，consensus 连接各位置 winner，
    mean_entropy 为 format(E/LENGTH,'.17g')。LENGTH=0 时 positions、
    consensus、mean_entropy 依次为 []、""、"0"。序列化、返回码、stderr、
    失败原子性及不写文件均沿用原 batch。

    python seqmodel.py beam-lstm-attn MODEL START START_T END_T BEAM LENGTH
    WINDOW：以线性退火温度、带注意力上下文从 version 2 的 LSTM 模型做确定
    性束搜索。除 BEAM 与确定性选束外，参数校验、LENGTH 的 0/1 语义、线性
    温度、LSTM 状态、注意力记忆、Why/by logit、稳定 softmax 与有限性失败
    契约均沿用 sample-lstm-attn-anneal；本命令无 SEED、无随机源且不写文
    件。BEAM 整串匹配 [1-9][0-9]*（任意位数合法），每轮仅当其数学值小于
    候选数时转 int，否则保留全部候选。初始束为 (0.0, "", h0, c0, START
    索引, [h0])；第 t 轮逐束推进 h、c 并求缩放 logit a、m=max(a)、
    e[k]=exp(a[k]-m)，d 自 0.0 按 k 升序累加，每个 k 生成子束：分数加
    a[k]-m-log(d)，文本追加 vocab[k]，置 x=k，memory 追加 h 的 float 副
    本；全部子束按 (-分数, 生成索引元组) 升序，保留前 min(BEAM, 候选数)
    项，长度归一化因子固定为 1。LENGTH 为 0 时仅输出 LF，否则输出最终首
    束文本加 LF。

    python seqmodel.py beam-lstm-attn-topk-topp MODEL START START_T END_T
    TOP_K TOP_P BEAM LENGTH WINDOW：以线性退火温度、带注意力上下文从
    version 2 的 LSTM 模型做 top-k 截断再 top-p 截取的确定性束搜索。除
    TOP_K、TOP_P 及候选截取外，参数校验、LENGTH 的 0/1 语义、线性温度、
    LSTM 状态、注意力记忆、Why/by logit、稳定 softmax 与有限性失败契约
    均沿用 beam-lstm-attn；本命令无 SEED、无随机源且不写文件。TOP_K 整
    串匹配 [1-9][0-9]* 且数学值 K 不超过 V=len(vocab)：先按十进制位数及
    同长度字典序与 V 比较，越界即失败，仅通过后转 int。TOP_P 经 float()
    解析，须有限且 0<TOP_P<=1。BEAM 整串匹配 [1-9][0-9]*，每轮仅当其数
    学值小于候选数时转 int，否则保留全部候选。第 t 轮逐束按既有顺序求
    a、m=max(a)、e[k]=exp(a[k]-m)，索引按 (-e[k],k) 升序取前 K 项，sK
    自 0.0 按该序累加 e，令 target=TOP_P*sK，再从 0.0 按该序累加 e，取
    累计值首次 >=target 的最短前缀，s 为其累计和；仅为前缀中每个 k 生成
    子束，分数加 a[k]-m-log(s)，文本、x、h、c、memory 更新沿用
    beam-lstm-attn；上述乘加、log 与分数非有限均失败。全部子束按
    (-分数, 生成索引元组) 升序，保留前 min(BEAM, 候选数) 项。LENGTH 为
    0 时仅输出 LF，否则输出最终首束文本加 LF。

    python seqmodel.py beam-lstm-attn-topk-topp-nbest MODEL START START_T
    END_T TOP_K TOP_P BEAM N LENGTH WINDOW：除 N 及输出外，全部参数校验
    与逐轮搜索行为严格沿用 beam-lstm-attn-topk-topp（候选裁剪、累计分数
    与索引元组决胜一致），无 SEED、无随机源且不写文件。N 整串匹配
    [1-9][0-9]*（任意位数合法）；最终束形成后，以十进制位数及同长字典
    序比较 N 与最终束数，仅当 N 较小时才转 int 并取前 N 束，否则取全部，
    超长 N 文本不得触发整数转换异常。按最终束既有顺序，每个候选独占一
    个 JSON 行，其值仅为生成文本，恰用
    json.dumps(text,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n' 序列化；各行直接拼接，末行保留 LF。LENGTH 为
    0 时最终仅初始束，故输出一个空字符串 JSON 行。

    python seqmodel.py beam-lstm-attn-topk-topp-nbest-scored MODEL START
    START_T END_T TOP_K TOP_P BEAM N LENGTH WINDOW：除输出外，全部参数校
    验、逐轮搜索、候选裁剪、累计分数、索引元组决胜及 N 的超长十进制处理
    均严格沿用 beam-lstm-attn-topk-topp-nbest，无 SEED、无随机源且不写
    文件。按最终束既有顺序取前 min(N, 束数) 项，每项独占一个 JSON 对象
    行，键序恰为 text、score：值分别为生成字符串与累计分数的
    format(score, '.17g') 字符串（负零写作 "-0"）。每行恰由
    json.dumps(obj,ensure_ascii=True,separators=(',',':'),
    allow_nan=False)+'\\n' 生成，各行直接拼接且末行保留 LF，不输出额外
    空白。LENGTH 为 0 时唯一一行恰为 {"text":"","score":"0"}\\n。

    python seqmodel.py train-attn MODEL CORPUS OUT WINDOW：前向严格复用
    perplexity-attn 的 n_t、M_t、u_t 与 logit 顺序；反向令
    g_t = p_t-onehot(y_t)，按 t 升序累加 dWhy、dby（以 u_t 为隐状态），
    按 t 降序经 attention_context_backward 把梯度映射回 dhs（h0 梯度丢弃），
    再以同一 RNN 前向缓存调用 VanillaRNN.backward，裁剪、更新与 train 相同，
    h0 不变。

    python seqmodel.py train-lstm-attn MODEL CORPUS OUT WINDOW：前向严格复用
    perplexity-lstm-attn 的 h、c、memory、M_t、u_t 与 logit 顺序并缓存每步
    cache；反向令 g_t = p_t-onehot(y_t)，按 t 升序累加 dWhy、dby（以 u_t 为
    隐状态）并求 du_t = Whyᵀg_t，按 t 降序经 attention_context_backward 把
    梯度映射回 dhs（h0 梯度丢弃），再以 backward_sequence(dhs, caches) 取得
    dW、db；梯度组序、5.0 裁剪、0.1 更新、h0/c0 不变与 OUT 写出均沿用
    train-lstm。成功时 stdout 为空并返回 0。

    python seqmodel.py train-lstm-mha MODEL CORPUS OUT WINDOW：MODEL、CORPUS
    沿用 perplexity-lstm-mha 的 version 4 十三键契约；前向严格复用
    perplexity-lstm-mha 的 h、c、memory、M_t、四组投影多头交叉注意力与
    u_t 顺序并缓存每步 cache。反向令 g_t = p_t-onehot(y_t)，按 t 升序
    累加 dWhy、dby（以 u_t 为隐状态）并求 du_t = Whyᵀg_t；dhs 置零，
    t 降序调用 mha.backward_cross([du_t])：du_t 加 dqx[0] 加至 dhs[t]，
    dkvx 按 M_t 顺序映射回完整记忆（h0 项丢弃，其余加至对应 dhs），
    dWq、dWk、dWv、dWo 按 t 降序、行列序累加；再以
    backward_sequence(dhs, caches) 取得 dW、db。梯度依次为 dW、db、dWq、
    dWk、dWv、dWo、dWhy、dby，5.0 全局裁剪、0.1 更新，h0、c0、heads
    不变；OUT 十三键与键序沿用 perplexity-lstm-mha（version 为 int 4）。
    成功时 stdout 为空并返回 0。

    python seqmodel.py train-gru-attn-tbptt MODEL CORPUS OUT WINDOW K：
    MODEL、CORPUS、OUT、0.1 更新、5.0 裁剪及成败协议沿用 train-gru；
    WINDOW 沿用 perplexity-gru-attn 的词法及任意位数安全截取。K 整串匹配
    [1-9][0-9]*（任意位数均合法）；令 T 为预测步数，按十进制位数及同长
    字典序比较 K 与 T 求 Ke=min(K,T)，不先转换超长 K。前向完全沿用
    perplexity-gru-attn（逐步 h、M_t、u_t 与 GRU cache 均缓存）。令
    g=p-onehot(y)，按 train-gru 次序以 u_t 累加 dWhy、dby 并求
    du_t=Whyᵀg_t；dhs 置零，t 降序调用
    attention_context_backward(h_t,M_t,du_t)：dn 加至 dhs[t]；dmemory
    第 p 行令 q=t+1-len(M_t)+p，q=0 丢弃，否则 s=q-1，仅当
    (T-1-s)//Ke==(T-1-t)//Ke 才按 p、i 升序加至 dhs[s]。再调用
    GRUCell.backward_sequence(dhs,caches,None,Ke) 取 dW、db；梯度组序
    dW、db、dWhy、dby，h0 不变。成功时 stdout 为空并返回 0。

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
        elif len(argv) == 4 and argv[1] == "perplexity-gru":
            output = _perplexity_gru(argv[2], argv[3])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 5 and argv[1] == "perplexity-gru-attn":
            output = _perplexity_gru_attn(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 5 and argv[1] == "train-lstm":
            _train_lstm(argv[2], argv[3], argv[4])
        elif len(argv) == 5 and argv[1] == "train-gru":
            _train_gru(argv[2], argv[3], argv[4])
        elif len(argv) == 7 and argv[1] == "sample":
            output = _sample(argv[2], argv[3], argv[4], argv[5], argv[6])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 7 and argv[1] == "sample-lstm":
            output = _sample_lstm(argv[2], argv[3], argv[4], argv[5],
                                  argv[6])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 7 and argv[1] == "sample-gru":
            output = _sample_gru(argv[2], argv[3], argv[4], argv[5],
                                 argv[6])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 8 and argv[1] == "sample-gru-attn":
            output = _sample_gru_attn(argv[2], argv[3], argv[4], argv[5],
                                      argv[6], argv[7])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 8 and argv[1] == "sample-anneal":
            output = _sample_anneal(argv[2], argv[3], argv[4], argv[5],
                                    argv[6], argv[7])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 8 and argv[1] == "sample-lstm-anneal":
            output = _sample_lstm_anneal(argv[2], argv[3], argv[4], argv[5],
                                         argv[6], argv[7])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 9 and argv[1] == "sample-gru-attn-anneal":
            output = _sample_gru_attn_anneal(argv[2], argv[3], argv[4],
                                             argv[5], argv[6], argv[7],
                                             argv[8])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 10 and argv[1] == "sample-gru-attn-top-p":
            output = _sample_gru_attn_topp(argv[2], argv[3], argv[4],
                                           argv[5], argv[6], argv[7],
                                           argv[8], argv[9])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 10 and argv[1] == "sample-gru-attn-top-k":
            output = _sample_gru_attn_topk(argv[2], argv[3], argv[4],
                                           argv[5], argv[6], argv[7],
                                           argv[8], argv[9])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 11 and argv[1] == "sample-gru-attn-top-k-top-p":
            output = _sample_gru_attn_topk_topp(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                argv[8], argv[9], argv[10])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 5 and argv[1] == "perplexity-attn":
            output = _perplexity_attn(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 5 and argv[1] == "perplexity-lstm-attn":
            output = _perplexity_lstm_attn(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 5 and argv[1] == "perplexity-lstm-mha":
            output = _perplexity_lstm_mha(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 5 and argv[1] == "perplexity-lstm-attn-trace":
            output = _perplexity_lstm_attn_trace(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 5 and argv[1] == "eval-windows":
            output = _eval_windows(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 6 and argv[1] == "window-sensitivity-lstm-attn":
            output = _window_sensitivity_lstm_attn(
                argv[2], argv[3], argv[4], argv[5])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 6 and argv[1] == "compare-lstm-attn":
            output = _compare_lstm_attn(
                argv[2], argv[3], argv[4], argv[5])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 5 and argv[1] == "compare-suite":
            output = _compare_suite(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 4 and argv[1] == "rank-suite":
            output = _rank_suite(argv[2], argv[3])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 4 and argv[1] == "rank-suite-details":
            output = _rank_suite_details(argv[2], argv[3])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 5 and argv[1] == "rank-window-sensitivity":
            output = _rank_window_sensitivity(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 5 and argv[1] == "rank-window-stability":
            output = _rank_window_stability(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif (len(argv) == 5
              and argv[1] == "rank-window-stability-details"):
            output = _rank_window_stability_details(argv[2], argv[3],
                                                    argv[4])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 5 and argv[1] == "lstm-attn-weights":
            output = _lstm_attn_weights(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 5 and argv[1] == "lstm-attn-entropy":
            output = _lstm_attn_entropy(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 5 and argv[1] == "lstm-attn-mean-lag":
            output = _lstm_attn_mean_lag(argv[2], argv[3], argv[4])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 6 and argv[1] == "lstm-attn-reach":
            output = _lstm_attn_reach(argv[2], argv[3], argv[4], argv[5])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 6 and argv[1] == "lstm-attn-reach-profile":
            output = _lstm_attn_reach_profile(
                argv[2], argv[3], argv[4], argv[5])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 6 and argv[1] == "train-attn":
            _train_attn(argv[2], argv[3], argv[4], argv[5])
        elif len(argv) == 6 and argv[1] == "train-lstm-attn":
            _train_lstm_attn(argv[2], argv[3], argv[4], argv[5])
        elif len(argv) == 6 and argv[1] == "train-lstm-mha":
            _train_lstm_mha(argv[2], argv[3], argv[4], argv[5])
        elif len(argv) == 7 and argv[1] == "train-gru-attn-tbptt":
            _train_gru_attn_tbptt(argv[2], argv[3], argv[4], argv[5],
                                  argv[6])
        elif len(argv) == 8 and argv[1] == "sample-attn":
            output = _sample_attn(argv[2], argv[3], argv[4], argv[5],
                                  argv[6], argv[7])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 8 and argv[1] == "sample-lstm-attn":
            output = _sample_lstm_attn(argv[2], argv[3], argv[4], argv[5],
                                       argv[6], argv[7])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 7 and argv[1] == "score-lstm-attn":
            output = _score_lstm_attn(argv[2], argv[3], argv[4], argv[5],
                                      argv[6])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 7 and argv[1] == "score-lstm-attn-batch":
            output = _score_lstm_attn_batch(argv[2], argv[3], argv[4],
                                            argv[5], argv[6])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 7 and argv[1] == "score-lstm-attn-temps":
            output = _score_lstm_attn_temps(argv[2], argv[3], argv[4],
                                            argv[5], argv[6])
            sys.stdout.buffer.write(output.encode("ascii"))
        elif len(argv) == 9 and argv[1] == "sample-lstm-attn-anneal":
            output = _sample_lstm_attn_anneal(argv[2], argv[3], argv[4],
                                              argv[5], argv[6], argv[7],
                                              argv[8])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 10 and argv[1] == "sample-lstm-attn-top-p":
            output = _sample_lstm_attn_topp(argv[2], argv[3], argv[4],
                                           argv[5], argv[6], argv[7],
                                           argv[8], argv[9])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif (len(argv) == 10
              and argv[1] == "sample-lstm-attn-top-p-scored"):
            output = _sample_lstm_attn_topp_scored(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                argv[8], argv[9])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif (len(argv) == 10
              and argv[1] == "sample-lstm-attn-top-p-batch"):
            output = _sample_lstm_attn_topp_batch(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                argv[8], argv[9])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 10 and argv[1] == "sample-lstm-attn-top-k":
            output = _sample_lstm_attn_topk(argv[2], argv[3], argv[4],
                                            argv[5], argv[6], argv[7],
                                            argv[8], argv[9])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif (len(argv) == 10
              and argv[1] == "sample-lstm-attn-top-k-scored"):
            output = _sample_lstm_attn_topk_scored(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                argv[8], argv[9])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 11 and argv[1] == "sample-lstm-attn-top-k-top-p":
            output = _sample_lstm_attn_topk_topp(argv[2], argv[3], argv[4],
                                                 argv[5], argv[6], argv[7],
                                                 argv[8], argv[9], argv[10])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif (len(argv) == 11
              and argv[1] == "sample-lstm-attn-top-k-top-p-scored"):
            output = _sample_lstm_attn_topk_topp_scored(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                argv[8], argv[9], argv[10])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif (len(argv) == 11
              and argv[1] == "sample-lstm-attn-top-k-top-p-batch"):
            output = _sample_lstm_attn_topk_topp_batch(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                argv[8], argv[9], argv[10])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif (len(argv) == 11
              and argv[1] == "sample-lstm-attn-consensus"):
            output = _sample_lstm_attn_consensus(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                argv[8], argv[9], argv[10])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 9 and argv[1] == "beam-lstm-attn":
            output = _beam_lstm_attn(argv[2], argv[3], argv[4], argv[5],
                                     argv[6], argv[7], argv[8])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 11 and argv[1] == "beam-lstm-attn-topk-topp":
            output = _beam_lstm_attn_topk_topp(argv[2], argv[3], argv[4],
                                               argv[5], argv[6], argv[7],
                                               argv[8], argv[9], argv[10])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif len(argv) == 12 and argv[1] == "beam-lstm-attn-topk-topp-nbest":
            output = _beam_lstm_attn_topk_topp_nbest(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                argv[8], argv[9], argv[10], argv[11])
            sys.stdout.buffer.write(output.encode("utf-8"))
        elif (len(argv) == 12
              and argv[1] == "beam-lstm-attn-topk-topp-nbest-scored"):
            output = _beam_lstm_attn_topk_topp_nbest_scored(
                argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                argv[8], argv[9], argv[10], argv[11])
            sys.stdout.buffer.write(output.encode("utf-8"))
        else:
            raise ValueError(
                "usage: seqmodel.py perplexity MODEL CORPUS | "
                "seqmodel.py train MODEL CORPUS OUT | "
                "seqmodel.py sample MODEL START SEED TEMPERATURE LENGTH | "
                "seqmodel.py sample-anneal MODEL START SEED START_T END_T "
                "LENGTH | "
                "seqmodel.py sample-lstm-attn-top-p MODEL START SEED "
                "START_T END_T TOP_P LENGTH WINDOW | "
                "seqmodel.py sample-lstm-attn-top-k MODEL START SEED "
                "START_T END_T TOP_K LENGTH WINDOW | "
                "seqmodel.py sample-lstm-attn-top-k-top-p MODEL START SEED "
                "START_T END_T TOP_K TOP_P LENGTH WINDOW")
    except Exception:
        sys.stderr.buffer.write(b"error\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
