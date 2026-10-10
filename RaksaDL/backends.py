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


def make_backend(blob, arena_size, n_in, n_out):
    if ON_DEVICE:
        return MicroBackend(blob, arena_size, n_in, n_out)
    return LiteRTBackend(blob)
