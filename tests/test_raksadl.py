import json, pathlib, sys
import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "examples"))
import synth_vibration as S
import raksadl
from raksadl import Window
from raksadl.convert import convert, ParityError


def _norm(x):
    a = np.array(x, dtype="float32").reshape(-1, S.T, S.C)
    return (a - np.array(S.MEAN, "float32")) / np.array(S.STD, "float32")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tf.keras.utils.set_random_seed(0)
    xs, ys = S.make_set(800, 1)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((S.T, S.C)),
        tf.keras.layers.Conv1D(8, 5, strides=2, activation="relu"),
        tf.keras.layers.Conv1D(16, 3, strides=2, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2, activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy")
    m.fit(_norm(xs), np.array(ys), epochs=20, verbose=0)
    out = tmp_path_factory.mktemp("m")
    xv, _ = S.make_set(200, 2)
    meta = convert(m, _norm(xs), out, "vib", x_val=_norm(xv), labels=S.LABELS,
                   norm={"mean": S.MEAN, "std": S.STD})
    return m, out, meta, xs


def load(out):
    return raksadl.Model.load(str(out / "vib.tflite"), str(out / "vib.json"))


def test_parity_report_dan_ops(built):
    _, _, meta, _ = built
    assert meta["report"]["label_agreement"] >= 0.98
    assert len(meta["ops"]) > 0 and len(meta["sha256"]) == 64


def test_selftest_lulus(built):
    assert load(built[1]).selftest() == (0, 16)


def test_klasifikasi_normal_vs_fault(built):
    model = load(built[1])
    xt, yt = S.make_set(60, 99)
    hit = sum(model.classify(x)[0] == y for x, y in zip(xt, yt))
    assert hit / len(yt) >= 0.9
    assert model.classify(xt[1])[2] in S.LABELS


def test_hash_menolak_model_rusak(built):
    out = built[1]
    blob = bytearray((out / "vib.tflite").read_bytes()); blob[len(blob) // 2] ^= 0xFF
    meta = json.loads((out / "vib.json").read_text())
    with pytest.raises(ValueError, match="sha256"):
        raksadl.Model(bytes(blob), meta)


def test_validasi_input(built):
    model = load(built[1])
    with pytest.raises(ValueError, match="NaN"):
        model.run([float("nan")] * (S.T * S.C))
    with pytest.raises(ValueError, match="panjang"):
        model.run([0.0] * 5)


def test_selftest_mendeteksi_penyimpangan(built):
    model = load(built[1])
    model.meta["vectors"]["output"] = [[v + 30 for v in r] for r in model.meta["vectors"]["output"]]
    assert model.selftest()[0] > 0


def test_normalisasi_per_channel(built):
    model = load(built[1])
    z = model.normalize([0.0, 0.0, 1.0] * 2)
    assert all(abs(v) < 1e-9 for v in z)


def test_convert_menolak_bila_parity_gagal(built, tmp_path):
    m, _, _, xs = built
    with pytest.raises(ParityError):
        convert(m, _norm(xs), tmp_path, "x", max_diff=-1.0)
    assert not (tmp_path / "x.tflite").exists()


def test_window_urutan_dan_hop():
    w = Window(size=3, channels=2, hop=2)
    flags = [w.push([i, i * 10]) for i in range(1, 7)]
    assert flags == [False, False, True, False, True, False]
    assert w.flatten() == [4, 40, 5, 50, 6, 60]
