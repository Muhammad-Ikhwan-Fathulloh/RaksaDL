"""Data sintetis akselerometer 3-sumbu: 'normal' vs 'fault' (komponen getaran frekuensi tinggi). Hanya untuk demo/tes."""
import math, random

T, C = 32, 3
LABELS = ["normal", "fault"]
MEAN, STD = [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]   # satuan g, per channel


def make_window(fault, rng):
    ph = rng.uniform(0, 6.28)
    w = []
    for t in range(T):
        row = []
        for c in range(C):
            v = 0.4 * math.sin(2 * math.pi * t / 16 + ph + c) + (1.0 if c == 2 else 0.0)
            v += rng.gauss(0, 0.05)
            if fault:
                v += 0.6 * math.sin(2 * math.pi * t / 3.0 + ph) + rng.gauss(0, 0.15)
            row.append(v)
        w.append(row)
    return w


def flat(w):
    return [v for row in w for v in row]


def make_set(n, seed):
    rng = random.Random(seed)
    xs, ys = [], []
    for k in range(n):
        f = k % 2
        xs.append(flat(make_window(f, rng)))
        ys.append(f)
    return xs, ys
