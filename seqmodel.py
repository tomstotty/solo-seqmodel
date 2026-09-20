"""seqmodel: 从零实现的序列建模库（仅 Python 标准库，离线）。

本模块提供确定性的 VanillaRNN、LSTMCell 与缩放点积 attention：

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

    scale = math.sqrt(float(D))

    w = []
    c = []
    for i in range(Tq):
        qi = q[i]

        # s[i][j] 从 0.0 按 d 升序累加点积，再除 sqrt(D)。
        s_row = [0.0] * Tk
        for j in range(Tk):
            kj = k[j]
            acc = 0.0
            for d in range(D):
                try:
                    acc += qi[d] * kj[d]
                except OverflowError:
                    # 两个合法 F 大整数先乘后加时可能在 int→float
                    # 转换处溢出，按契约改抛 ValueError。
                    raise ValueError(
                        "score dot product overflowed")
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

        # c[i][a] 从 0.0 按 j 升序累加 w[i][j]*v[j][a]。
        c_row = [0.0] * Dv
        for a in range(Dv):
            acc = 0.0
            for j in range(Tk):
                wv = w_row[j]
                if wv != 0.0:
                    try:
                        acc += wv * v[j][a]
                    except OverflowError:
                        # 合法 F 大整数在与 float 权值相乘时可能于
                        # int→float 转换处溢出，改抛 ValueError。
                        raise ValueError("context overflowed")
                    if not math.isfinite(acc):
                        raise ValueError(
                            "context accumulated to a non-finite value")
            c_row[a] = acc
        c.append(c_row)

    return c, w


def attention_backward(q, k, v, dc, mask=None):
    """缩放点积注意力的反向传播，返回 (dq, dk, dv)，不修改或复用任何输入。

    q、k、v、mask 完全沿用 attention 的契约；dc 须为 Tq×Dv 的 F 列表
    矩阵，否则抛 ValueError。先按与 attention 完全相同的前向语义重算
    权重 w，再令
        dw[i][j] = Σ_a (dc[i][a]*v[j][a])（mask False 位为 0.0）
        r[i]      = Σ_j (w[i][j]*dw[i][j])
        ds[i][j]  = w[i][j]*(dw[i][j]-r[i])/sqrt(D)（False 位为 0.0）
        dq[i][d]  = Σ_j (ds[i][j]*k[j][d])
        dk[j][d]  = Σ_i (ds[i][j]*q[i][d])
        dv[j][a]  = Σ_i (w[i][j]*dc[i][a])
    每个 Σ 均自 0.0 按其下标升序累加。返回形状依次为 Tq×D、Tk×D、
    Tk×Dv，元素均为 float，逐层新建列表。任一中间量或输出溢出或成为
    非有限值均抛 ValueError；实参数量错误沿用 Python 自带的 TypeError。
    相同输入结果确定。
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

    # dc：Tq×Dv 的 F 列表矩阵。
    if type(dc) is not list or len(dc) != Tq:
        raise ValueError("dc must be a list of shape %d×%d" % (Tq, Dv))
    for row in dc:
        if type(row) is not list or len(row) != Dv:
            raise ValueError("dc must be a list of shape %d×%d" % (Tq, Dv))
        for x in row:
            if not _is_f(x):
                raise ValueError("dc entries must be finite numbers, got %r"
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

    scale = math.sqrt(float(D))

    # 按与 attention 相同的前向语义重算 w。
    w = []
    for i in range(Tq):
        qi = q[i]

        # s[i][j] 从 0.0 按 d 升序累加点积，再除 sqrt(D)。
        s_row = [0.0] * Tk
        for j in range(Tk):
            kj = k[j]
            acc = 0.0
            for d in range(D):
                try:
                    acc += qi[d] * kj[d]
                except OverflowError:
                    raise ValueError("score dot product overflowed")
                if not math.isfinite(acc):
                    raise ValueError(
                        "score dot product accumulated to a non-finite value")
            sval = acc / scale
            if not math.isfinite(sval):
                raise ValueError("score became non-finite after scaling")
            s_row[j] = sval

        # 行最大值只在 True 位取。
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

    # 输出逐层新建，元素均为 float。
    dq = [[0.0] * D for _ in range(Tq)]
    dk = [[0.0] * D for _ in range(Tk)]
    dv = [[0.0] * Dv for _ in range(Tk)]

    # i 升序：逐行算 dw、r、ds、dq；dk 与 dv 跨 i 升序累加。
    for i in range(Tq):
        w_row = w[i]
        active_row = active[i]
        dc_row = dc[i]

        # dw[i][j] = Σ_a dc[i][a]*v[j][a]，False 位为 0.0。
        dw_row = [0.0] * Tk
        for j in range(Tk):
            if not active_row[j]:
                continue
            vj = v[j]
            acc = 0.0
            for a in range(Dv):
                try:
                    acc += dc_row[a] * vj[a]
                except OverflowError:
                    raise ValueError("dw overflowed")
                if not math.isfinite(acc):
                    raise ValueError("dw accumulated to a non-finite value")
            dw_row[j] = acc

        # r[i] = Σ_j w[i][j]*dw[i][j]，按 j 升序自 0.0 累加。
        r = 0.0
        for j in range(Tk):
            if w_row[j] != 0.0:
                try:
                    r += w_row[j] * dw_row[j]
                except OverflowError:
                    raise ValueError("r overflowed")
                if not math.isfinite(r):
                    raise ValueError("r accumulated to a non-finite value")

        # ds[i][j] = w[i][j]*(dw[i][j]-r[i])/sqrt(D)，False 位为 0.0。
        ds_row = [0.0] * Tk
        for j in range(Tk):
            if not active_row[j]:
                continue
            diff = dw_row[j] - r
            if not math.isfinite(diff):
                raise ValueError("dw-r became non-finite")
            try:
                dsv = w_row[j] * diff / scale
            except OverflowError:
                raise ValueError("ds overflowed")
            if not math.isfinite(dsv):
                raise ValueError("ds became non-finite")
            ds_row[j] = dsv

        # dq[i][d] = Σ_j ds[i][j]*k[j][d]，按 j 升序自 0.0 累加。
        dq_row = dq[i]
        for d in range(D):
            acc = 0.0
            for j in range(Tk):
                if ds_row[j] != 0.0:
                    try:
                        acc += ds_row[j] * k[j][d]
                    except OverflowError:
                        raise ValueError("dq overflowed")
                    if not math.isfinite(acc):
                        raise ValueError("dq accumulated to a non-finite value")
            dq_row[d] = acc

        # dk[j][d] = Σ_i ds[i][j]*q[i][d]：i 升序累加到同一输出单元。
        qi = q[i]
        for j in range(Tk):
            dsij = ds_row[j]
            if dsij != 0.0:
                dk_row = dk[j]
                for d in range(D):
                    try:
                        dk_row[d] += dsij * qi[d]
                    except OverflowError:
                        raise ValueError("dk overflowed")
                    if not math.isfinite(dk_row[d]):
                        raise ValueError(
                            "dk accumulated to a non-finite value")

            # dv[j][a] = Σ_i w[i][j]*dc[i][a]：i 升序累加，与 ds
            # 是否为零无关（False 位 w 为 0.0，自然跳过）。
            wij = w_row[j]
            if wij != 0.0:
                dv_row = dv[j]
                for a in range(Dv):
                    try:
                        dv_row[a] += wij * dc_row[a]
                    except OverflowError:
                        raise ValueError("dv overflowed")
                    if not math.isfinite(dv_row[a]):
                        raise ValueError(
                            "dv accumulated to a non-finite value")

    # 输出终检：所有元素必须为有限 float。
    for row in dq:
        for x in row:
            if type(x) is not float or not math.isfinite(x):
                raise ValueError("dq became non-finite")
    for row in dk:
        for x in row:
            if type(x) is not float or not math.isfinite(x):
                raise ValueError("dk became non-finite")
    for row in dv:
        for x in row:
            if type(x) is not float or not math.isfinite(x):
                raise ValueError("dv became non-finite")

    return dq, dk, dv


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
