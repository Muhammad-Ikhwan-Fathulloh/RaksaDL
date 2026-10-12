# raksadl/purepy.py — Inferensi int8 murni Python (tanpa microlite/numpy) untuk simulator (Wokwi) dan CPU kecil.
# Menjalankan graf portabel hasil raksadl.convert.export_graph, memakai aritmetika fixed-point seperti kernel
# referensi TFLite. Lambat dibanding microlite; dipakai untuk simulasi, uji, dan model sangat kecil.
import math

SUPPORTED = ("CONV_2D", "FULLY_CONNECTED", "MEAN", "SOFTMAX", "LOGISTIC", "RELU",
             "RESHAPE", "EXPAND_DIMS", "SQUEEZE")


def _qmul(real):
    if real == 0:
        return 0, 0
    m, e = math.frexp(real)
    q = int(round(m * 2147483648.0))
    if q == 2147483648:
        q //= 2
        e += 1
    return q, e


def _rq(x, q, e):
    """MultiplyByQuantizedMultiplier (SaturatingRoundingDoublingHighMul + RoundingDivideByPOT)."""
    right = 0
    if e > 0:
        x = x << e
    else:
        right = -e
    ab = x * q
    n = ab + (1073741824 if ab >= 0 else -1073741823)
    r = (n >> 31) if n >= 0 else -((-n) >> 31)
    if right:
        mask = (1 << right) - 1
        rem = r & mask
        thr = (mask >> 1) + (1 if r < 0 else 0)
        r = (r >> right) + (1 if rem > thr else 0)
    return r


def _act_range(act, so, zo):
    lo, hi = -128, 127
    if act == 1:                    # RELU
        lo = max(lo, zo)
    elif act == 3:                  # RELU6
        lo = max(lo, zo)
        hi = min(hi, zo + int(round(6.0 / so)))
    elif act == 2:                  # RELU_N1_TO_1
        lo = max(lo, zo + int(round(-1.0 / so)))
        hi = min(hi, zo + int(round(1.0 / so)))
    return lo, hi


def _quant(v, so, zo):
    q = int(math.floor(v / so + 0.5)) + zo
    return -128 if q < -128 else 127 if q > 127 else q


class PurePyBackend:
    def __init__(self, graph):
        self._t = graph["tensors"]
        self._in, self._out = graph["input"], graph["output"]
        self._plan = [self._prepare(op) for op in graph["ops"]]

    def _prepare(self, op):
        T, k, ins, outs, o = self._t, op["op"], op["in"], op["out"], op["opt"]
        x, y = T[ins[0]], T[outs[0]]
        p = {"k": k, "src": ins[0], "dst": outs[0], "opt": o, "xs": x["s"], "ys": y["s"],
             "zi": x["zp"][0] if x["zp"] else 0, "zo": y["zp"][0] if y["zp"] else 0,
             "si": x["sc"][0] if x["sc"] else 1.0, "so": y["sc"][0] if y["sc"] else 1.0}
        if k in ("CONV_2D", "FULLY_CONNECTED"):
            w = T[ins[1]]
            b = T[ins[2]] if len(ins) > 2 and ins[2] >= 0 else None
            p["w"], p["ws"], p["b"] = w["d"], w["s"], (b["d"] if b else None)
            p["m"] = [_qmul(p["si"] * s / p["so"]) for s in w["sc"]]
            p["lo"], p["hi"] = _act_range(o.get("act", 0), p["so"], p["zo"])
        elif k == "MEAN":
            p["axes"] = T[ins[1]]["d"]
        return p

    def run(self, q):
        v = {self._in: list(q)}
        for p in self._plan:
            k, x = p["k"], v[p["src"]]
            if k == "CONV_2D":
                y = self._conv(p, x)
            elif k == "FULLY_CONNECTED":
                y = self._fc(p, x)
            elif k == "MEAN":
                y = self._mean(p, x)
            elif k == "SOFTMAX" or k == "LOGISTIC" or k == "RELU":
                y = self._act(p, x)
            else:                   # RESHAPE / EXPAND_DIMS / SQUEEZE: data tetap, hanya bentuk berubah
                y = x
            v[p["dst"]] = y
        return v[self._out]

    def _conv(self, p, x):
        xs, ys, ws, o = p["xs"], p["ys"], p["ws"], p["opt"]
        H, W, C = xs[1], xs[2], xs[3]
        OH, OW, OC = ys[1], ys[2], ys[3]
        KH, KW = ws[1], ws[2]
        sh, sw = o["sh"], o["sw"]
        ph = pw = 0
        if o["pad"] == 0:           # SAME
            ph = max(0, (OH - 1) * sh + KH - H) // 2
            pw = max(0, (OW - 1) * sw + KW - W) // 2
        w, b, m, zi, zo, lo, hi = p["w"], p["b"], p["m"], p["zi"], p["zo"], p["lo"], p["hi"]
        y = []
        for oy in range(OH):
            for ox in range(OW):
                for oc in range(OC):
                    acc = b[oc] if b else 0
                    wb = oc * KH * KW * C
                    for ky in range(KH):
                        iy = oy * sh + ky - ph
                        if iy < 0 or iy >= H:
                            continue
                        for kx in range(KW):
                            ix = ox * sw + kx - pw
                            if ix < 0 or ix >= W:
                                continue
                            xi = (iy * W + ix) * C
                            wi = wb + (ky * KW + kx) * C
                            for c in range(C):
                                acc += (x[xi + c] - zi) * w[wi + c]
                    mq, me = m[oc] if len(m) > 1 else m[0]
                    r = _rq(acc, mq, me) + zo
                    y.append(lo if r < lo else hi if r > hi else r)
        return y

    def _fc(self, p, x):
        n = len(x)
        no = p["ys"][-1]
        w, b, m, zi, zo, lo, hi = p["w"], p["b"], p["m"], p["zi"], p["zo"], p["lo"], p["hi"]
        y = []
        for oc in range(no):
            acc = b[oc] if b else 0
            wb = oc * n
            for i in range(n):
                acc += (x[i] - zi) * w[wb + i]
            mq, me = m[oc] if len(m) > 1 else m[0]
            r = _rq(acc, mq, me) + zo
            y.append(lo if r < lo else hi if r > hi else r)
        return y

    def _mean(self, p, x):
        xs, rank = p["xs"], len(p["xs"])
        axes = [a if a >= 0 else a + rank for a in p["axes"]]
        cnt = 1
        for d in axes:
            cnt *= xs[d]
        st = [1] * rank
        for d in range(rank - 2, -1, -1):
            st[d] = st[d + 1] * xs[d + 1]
        ost, nout = [0] * rank, 1
        for d in range(rank - 1, -1, -1):
            if d not in axes:
                ost[d] = nout
                nout *= xs[d]
        sums = [0] * nout
        for i in range(len(x)):
            j, r = 0, i
            for d in range(rank):
                idx = r // st[d]
                r -= idx * st[d]
                j += idx * ost[d]
            sums[j] += x[i]
        k = p["si"] / p["so"]
        return [_quant((s / cnt - p["zi"]) * p["si"], p["so"], p["zo"]) for s in sums]

    def _act(self, p, x):
        si, zi, so, zo, k = p["si"], p["zi"], p["so"], p["zo"], p["k"]
        r = [(v - zi) * si for v in x]
        if k == "SOFTMAX":
            beta = p["opt"].get("beta", 1.0)
            mx = max(r)
            e = [math.exp((v - mx) * beta) for v in r]
            s = sum(e)
            r = [v / s for v in e]
        elif k == "LOGISTIC":
            r = [1.0 / (1.0 + math.exp(-v)) for v in r]
        else:
            r = [v if v > 0 else 0.0 for v in r]
        return [_quant(v, so, zo) for v in r]
