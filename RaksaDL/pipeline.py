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
