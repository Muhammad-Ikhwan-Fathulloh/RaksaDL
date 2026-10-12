# raksadl.py — DIHASILKAN OTOMATIS oleh tools/bundle_wokwi.py dari paket raksadl. Jangan diedit.

# ---- pipeline.py ----
# raksadl/pipeline.py — utilitas ringan untuk data sensor ke model: jendela geser dan pasca-proses.
import math


class Window:
    """Buffer melingkar (size sampel x channels). push() mengembalikan True saat jendela baru siap (tiap `hop` sampel)."""

    def __init__(self, size, channels, hop=None):
        self.size, self.channels = size, channels
        self.hop = hop or size
        self._buf = [0.0] * (size * channels)
        self._head = 0
        self._n = 0
        self._since = 0

    def push(self, sample):
        if len(sample) != self.channels:
            raise ValueError("jumlah channel salah")
        base = self._head * self.channels
        for c in range(self.channels):
            self._buf[base + c] = sample[c]
        self._head = (self._head + 1) % self.size
        if self._n < self.size:
            self._n += 1
            if self._n == self.size:
                self._since = 0
                return True
            return False
        self._since += 1
        if self._since >= self.hop:
            self._since = 0
            return True
        return False

    @property
    def ready(self):
        return self._n >= self.size

    def flatten(self):
        """Urut dari sampel tertua ke terbaru, bentuk (size*channels) berurutan per waktu."""
        if not self.ready:
            raise ValueError("jendela belum penuh")
        s = self._head * self.channels
        return self._buf[s:] + self._buf[:s]

    def clear(self):
        self._head = self._n = self._since = 0


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def argmax(v):
    best = 0
    for i in range(1, len(v)):
        if v[i] > v[best]:
            best = i
    return best


def top_k(v, k=3):
    idx = sorted(range(len(v)), key=lambda i: v[i], reverse=True)[:k]
    return [(i, v[i]) for i in idx]

# ---- purepy.py ----
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

# ---- backends.py ----
# raksadl/backends.py — satu antarmuka, dua mesin: microlite (perangkat) dan LiteRT/TensorFlow (laptop).
import gc

try:
    import microlite
    ON_DEVICE = True
except ImportError:
    microlite = None
    ON_DEVICE = False


class MicroBackend:
    """microlite.interpreter dari firmware tensorflow-micropython-examples.
    Callback dan nama method mengikuti contoh hello_world; sesuaikan di sini jika build kamu berbeda."""

    def __init__(self, blob, arena_size, n_in, n_out):
        gc.collect()
        self._blob = blob              # harus tetap hidup selama interpreter dipakai
        self._arena = bytearray(arena_size)
        self._n_in, self._n_out = n_in, n_out
        self._x, self._y = None, []
        self._it = microlite.interpreter(blob, self._arena, self._on_in, self._on_out)

    def _on_in(self, it):
        t = it.getInputTensor(0)
        for i in range(self._n_in):
            t.setValue(i, self._x[i])

    def _on_out(self, it):
        t = it.getOutputTensor(0)
        self._y = [t.getValue(i) for i in range(self._n_out)]

    def run(self, q):
        self._x = q
        self._it.invoke()
        return self._y


class LiteRTBackend:
    """Backend CPython untuk pengembangan, CI, dan verifikasi."""

    def __init__(self, blob):
        cls = None
        for name in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
            try:
                cls = __import__(name, None, None, ["Interpreter"]).Interpreter
                break
            except Exception:
                pass
        if cls is None:
            try:
                import tensorflow as tf
                cls = tf.lite.Interpreter
            except Exception:
                raise ImportError("Pasang ai-edge-litert (disarankan), tflite-runtime, atau tensorflow")
        self._it = cls(model_content=blob)
        self._it.allocate_tensors()
        self._i = self._it.get_input_details()[0]
        self._o = self._it.get_output_details()[0]

    def run(self, q):
        import numpy as np
        a = np.array(q, dtype=self._i["dtype"]).reshape(self._i["shape"])
        self._it.set_tensor(self._i["index"], a)
        self._it.invoke()
        return self._it.get_tensor(self._o["index"]).reshape(-1).tolist()


def make_backend(blob, arena_size, n_in, n_out, graph=None, backend="auto"):
    """backend: auto | microlite | litert | purepy. auto = microlite di perangkat, LiteRT di laptop,
    dan purepy bila tidak ada runtime lain tetapi graf tersedia (mis. simulator Wokwi)."""
    if backend == "purepy":
        if graph is None:
            raise ValueError("backend purepy butuh graf (graph_path)")
        return PurePyBackend(graph)
    if backend == "microlite" or (backend == "auto" and ON_DEVICE):
        return MicroBackend(blob, arena_size, n_in, n_out)
    if backend == "litert":
        return LiteRTBackend(blob)
    try:
        return LiteRTBackend(blob)
    except Exception:
        if graph is not None:
            return PurePyBackend(graph)
        raise

# ---- model.py ----
# raksadl/model.py — Model TFLite (int8 atau float32) dengan metadata, validasi input, dan verifikasi parity.
import gc

try:
    import ujson as json
except ImportError:
    import json
try:
    import uhashlib as _hashlib
except ImportError:
    import hashlib as _hashlib
try:
    import ubinascii as _binascii
except ImportError:
    import binascii as _binascii



def _prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def sha256_hex(data):
    h = _hashlib.sha256()
    h.update(data)
    return _binascii.hexlify(h.digest()).decode()


def sha256_file(path):
    h = _hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(512)
            if not c:
                break
            h.update(c)
    return _binascii.hexlify(h.digest()).decode()


class Model:
    """
    Satu input dan satu output tensor (v0.1). Input mentah (satuan sensor) -> normalisasi per channel
    (dari metadata) -> kuantisasi -> inferensi -> dekuantisasi.
    """

    def __init__(self, blob, meta, arena_size=16 * 1024, verify=True, backend="auto", graph=None):
        if isinstance(blob, str):
            with open(blob, "rb") as f:
                blob = f.read()
        if blob is not None and verify and meta.get("sha256") and sha256_hex(blob) != meta["sha256"]:
            raise ValueError("Model rusak atau diubah: sha256 tidak cocok dengan metadata")
        self.meta = meta
        self.labels = meta.get("labels")
        self._in, self._out = meta["input"], meta["output"]
        self.n_in, self.n_out = _prod(self._in["shape"]), _prod(self._out["shape"])
        self._norm = meta.get("norm")
        self._be = make_backend(blob, arena_size, self.n_in, self.n_out, graph, backend)

    @classmethod
    def load(cls, tflite_path, meta_path, graph_path=None, **kw):
        """tflite_path boleh None bila backend="purepy" (mis. Wokwi) dan graph_path diberikan."""
        with open(meta_path) as f:
            meta = json.load(f)
        graph = None
        if graph_path:
            if kw.get("verify", True) and meta.get("graph_sha256") and sha256_file(graph_path) != meta["graph_sha256"]:
                raise ValueError("Graf rusak atau diubah: sha256 tidak cocok dengan metadata")
            with open(graph_path) as f:
                graph = json.load(f)
        return cls(tflite_path, meta, graph=graph, **kw)

    # --- pra-proses ---
    def normalize(self, x):
        n = self._norm
        if not n:
            return list(x)
        mean, std = n["mean"], n["std"]
        k = len(mean)
        return [(x[i] - mean[i % k]) / std[i % k] for i in range(len(x))]

    def _quant(self, v):
        t = self._in
        if t["dtype"] != "int8":
            return v
        q = int(round(v / t["scale"])) + t["zero"]
        return -128 if q < -128 else 127 if q > 127 else q

    def _dequant(self, raw):
        t = self._out
        if t["dtype"] != "int8":
            return list(raw)
        return [(r - t["zero"]) * t["scale"] for r in raw]

    # --- inferensi ---
    def run(self, x, normalized=False):
        if len(x) != self.n_in:
            raise ValueError("panjang input %d, model butuh %d" % (len(x), self.n_in))
        for v in x:
            if v != v:
                raise ValueError("input mengandung NaN")
        z = x if normalized else self.normalize(x)
        return self._dequant(self._be.run([self._quant(v) for v in z]))

    def classify(self, x, normalized=False):
        """Return (indeks, skor, label). Mengasumsikan keluaran model adalah probabilitas (softmax/sigmoid)."""
        y = self.run(x, normalized)
        i = argmax(y) if len(y) > 1 else int(y[0] >= 0.5)
        score = y[i] if len(y) > 1 else (y[0] if i else 1.0 - y[0])
        return i, score, (self.labels[i] if self.labels else None)

    def top_k(self, x, k=3, normalized=False):
        return top_k(self.run(x, normalized), k)

    # --- verifikasi ---
    def selftest(self, tol=None):
        """Jalankan vektor uji dari metadata dan bandingkan dengan keluaran acuan. Return (n_menyimpang, n)."""
        v = self.meta["vectors"]
        if tol is None:
            tol = 1 if self._in["dtype"] == "int8" else 1e-4
        bad = 0
        for xin, yexp in zip(v["input"], v["output"]):
            got = self._be.run(list(xin))
            if max(abs(a - b) for a, b in zip(got, yexp)) > tol:
                bad += 1
        return bad, len(v["input"])

__version__ = "0.1.0"
