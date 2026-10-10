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

from .backends import make_backend
from .pipeline import argmax, top_k


def _prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def sha256_hex(data):
    h = _hashlib.sha256()
    h.update(data)
    return _binascii.hexlify(h.digest()).decode()


class Model:
    """
    Satu input dan satu output tensor (v0.1). Input mentah (satuan sensor) -> normalisasi per channel
    (dari metadata) -> kuantisasi -> inferensi -> dekuantisasi.
    """

    def __init__(self, blob, meta, arena_size=16 * 1024, verify=True):
        if isinstance(blob, str):
            with open(blob, "rb") as f:
                blob = f.read()
        if verify and meta.get("sha256") and sha256_hex(blob) != meta["sha256"]:
            raise ValueError("Model rusak atau diubah: sha256 tidak cocok dengan metadata")
        self.meta = meta
        self.labels = meta.get("labels")
        self._in, self._out = meta["input"], meta["output"]
        self.n_in, self.n_out = _prod(self._in["shape"]), _prod(self._out["shape"])
        self._norm = meta.get("norm")
        self._be = make_backend(blob, arena_size, self.n_in, self.n_out)

    @classmethod
    def load(cls, tflite_path, meta_path, **kw):
        with open(meta_path) as f:
            meta = json.load(f)
        return cls(tflite_path, meta, **kw)

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
