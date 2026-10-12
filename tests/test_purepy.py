import importlib.util, json, pathlib, sys
import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")
from test_raksadl import built, _norm          # fixture model 1D-CNN (dilatih di tes)
import synth_vibration as S
import raksadl
from raksadl.convert import export_graph, UnsupportedOp

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def both(out):
    t, m, g = str(out / "vib.tflite"), str(out / "vib.json"), str(out / "vib.graph.json")
    return (raksadl.Model.load(t, m, backend="litert"),
            raksadl.Model.load(None, m, graph_path=g, backend="purepy"))


def test_purepy_selftest_lulus(built):
    _, pp = both(built[1])
    assert pp.selftest() == (0, 16)


def test_purepy_sama_dengan_tflite(built):
    ref, pp = both(built[1])
    xt, _ = S.make_set(60, 123)
    worst = 0
    for x in xt:
        q = [ref._quant(v) for v in ref.normalize(x)]
        a, b = ref._be.run(q), pp._be.run(q)
        worst = max(worst, max(abs(i - j) for i, j in zip(a, b)))
    assert worst <= 1


def test_graf_rusak_ditolak(built):
    out = built[1]
    g = out / "vib.graph.json"
    orig = g.read_text()
    try:
        g.write_text(orig.replace("1", "2", 1))
        with pytest.raises(ValueError, match="Graf rusak"):
            raksadl.Model.load(None, str(out / "vib.json"), graph_path=str(g), backend="purepy")
    finally:
        g.write_text(orig)


def test_op_tidak_didukung_ditolak():
    m = tf.keras.Sequential([tf.keras.layers.Input((8, 3)), tf.keras.layers.Conv1D(4, 3),
                             tf.keras.layers.MaxPooling1D(2), tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(2, activation="softmax")])
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = lambda: ([np.random.rand(1, 8, 3).astype("float32")] for _ in range(10))
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = c.inference_output_type = tf.int8
    with pytest.raises(UnsupportedOp):
        export_graph(c.convert())


def test_bundle_wokwi_satu_file(built, tmp_path):
    from bundle_wokwi import bundle
    f = bundle(tmp_path / "raksadl_bundle.py")
    spec = importlib.util.spec_from_file_location("raksadl_bundle", f)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = built[1]
    m = mod.Model.load(None, str(out / "vib.json"), graph_path=str(out / "vib.graph.json"), backend="purepy")
    assert m.selftest() == (0, 16)
    xt, yt = S.make_set(40, 7)
    assert sum(m.classify(x)[0] == y for x, y in zip(xt, yt)) / 40 >= 0.9


def test_main_wokwi_smoke(built, monkeypatch):
    """Jalankan wokwi/main.py dengan stub machine/time: logika SIM + tombol + LED berjalan."""
    import time, types
    state = {"btn": 1}

    class Pin:
        OUT, IN, PULL_UP = 1, 0, 2
        def __init__(self, n, mode=None, pull=None): self.n, self.v = n, 0
        def value(self, v=None):
            if v is None: return state["btn"] if self.n == 15 else self.v
            self.v = v
        def on(self): self.v = 1

    stub = types.ModuleType("machine"); stub.Pin = Pin; stub.I2C = object
    monkeypatch.setitem(sys.modules, "machine", stub)
    monkeypatch.setattr(time, "sleep_ms", lambda ms: None, raising=False)
    monkeypatch.setattr(time, "ticks_ms", lambda: int(time.time() * 1000), raising=False)
    monkeypatch.setattr(time, "ticks_us", lambda: int(time.time() * 1e6), raising=False)
    monkeypatch.setattr(time, "ticks_diff", lambda a, b: a - b, raising=False)
    out = built[1]
    for f in ("vib.json", "vib.graph.json"):
        (out / f.replace("vib", "vibration")).write_text((out / f).read_text())
    monkeypatch.chdir(out)
    monkeypatch.syspath_prepend(str(ROOT / "wokwi"))
    bundle_src = ROOT / "wokwi" / "raksadl.py"
    spec = importlib.util.spec_from_file_location("wk_main", ROOT / "wokwi" / "main.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.main(max_windows=3)
    state["btn"] = 0
    mod.main(max_windows=3)
