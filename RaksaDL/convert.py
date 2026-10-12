# raksadl/convert.py — HANYA CPython (laptop/Colab/CI). Keras -> TFLite + metadata + laporan parity.
# Tidak diinstal ke perangkat. Butuh: tensorflow, numpy.
import argparse, hashlib, json, pathlib
import numpy as np


class ParityError(RuntimeError):
    pass


class UnsupportedOp(ValueError):
    pass


def export_graph(blob):
    """TFLite flatbuffer -> graf portabel (dict) untuk raksadl.purepy. Hanya model int8 dengan op yang didukung."""
    from tensorflow.lite.python import schema_py_generated as S
    from .purepy import SUPPORTED
    mt = S.ModelT.InitFromPackedBuf(blob, 0)
    sg = mt.subgraphs[0]
    names = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
    kinds, npk = {0: "float32", 2: "int32", 9: "int8"}, {0: np.float32, 2: np.int32, 9: np.int8}
    tensors = []
    for t in sg.tensors:
        if t.type not in kinds:
            raise UnsupportedOp("tipe tensor %s tidak didukung" % t.type)
        q = t.quantization
        has = q is not None and q.scale is not None and len(q.scale) > 0
        raw = mt.buffers[t.buffer].data
        tensors.append({
            "s": [int(d) for d in t.shape], "t": kinds[t.type],
            "sc": [float(v) for v in q.scale] if has else [],
            "zp": [int(v) for v in q.zeroPoint] if has else [],
            "d": np.frombuffer(raw.tobytes(), dtype=npk[t.type]).tolist() if raw is not None and len(raw) else None})
    ops = []
    for op in sg.operators:
        oc = mt.operatorCodes[op.opcodeIndex]
        name = names[max(oc.builtinCode, oc.deprecatedBuiltinCode)]
        if name not in SUPPORTED:
            raise UnsupportedOp("op %s belum didukung backend purepy" % name)
        b, opt = op.builtinOptions, {}
        if name == "CONV_2D":
            if getattr(b, "dilationHFactor", 1) != 1 or getattr(b, "dilationWFactor", 1) != 1:
                raise UnsupportedOp("dilasi belum didukung")
            opt = {"pad": int(b.padding), "sh": int(b.strideH), "sw": int(b.strideW),
                   "act": int(b.fusedActivationFunction)}
        elif name == "FULLY_CONNECTED":
            opt = {"act": int(b.fusedActivationFunction)}
        elif name == "SOFTMAX":
            opt = {"beta": float(b.beta)}
        ops.append({"op": name, "in": [int(i) for i in op.inputs], "out": [int(i) for i in op.outputs], "opt": opt})
    return {"input": int(sg.inputs[0]), "output": int(sg.outputs[0]), "tensors": tensors, "ops": ops}


def _interp(blob):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_content=blob)
    it.allocate_tensors()
    return it


def _ops(it):
    try:  # API privat TFLite; berguna untuk mencocokkan dengan op yang ada di firmware
        return sorted({d["op_name"] for d in it._get_ops_details()} - {"DELEGATE"})
    except Exception:
        return []


def convert(model, x_repr, out_dir, name, x_val=None, labels=None, norm=None, quant="int8",
            min_agreement=0.98, max_diff=0.15, n_vectors=16, check=True):
    """x_repr/x_val: array float32 yang sudah dinormalisasi persis seperti yang dilihat model (tanpa dimensi batch
    di luar sampel). norm={'mean':[..],'std':[..]} dicatat agar perangkat menormalisasi data mentah yang sama."""
    import tensorflow as tf
    x_repr = np.asarray(x_repr, dtype="float32")
    x_val = x_repr if x_val is None else np.asarray(x_val, dtype="float32")

    c = tf.lite.TFLiteConverter.from_keras_model(model)
    if quant == "int8":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = lambda: ([s[None, ...]] for s in x_repr[:200])
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = c.inference_output_type = tf.int8
    blob = c.convert()

    it = _interp(blob)
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    (si, zi), (so, zo) = i["quantization"], o["quantization"]
    is_q = quant == "int8"

    def run(sample):
        a = sample[None, ...]
        if is_q:
            a = np.clip(np.round(a / si + zi), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], a)
        it.invoke()
        raw = it.get_tensor(o["index"]).reshape(-1)
        deq = (raw.astype("float32") - zo) * so if is_q else raw.astype("float32")
        return a.reshape(-1), raw, deq

    ref = model.predict(x_val, verbose=0).reshape(len(x_val), -1)
    outs = [run(s) for s in x_val]
    got = np.array([r[2] for r in outs])
    if ref.shape[1] > 1:
        agree = float(np.mean(ref.argmax(1) == got.argmax(1)))
    else:
        agree = float(np.mean((ref[:, 0] >= .5) == (got[:, 0] >= .5)))
    report = {"label_agreement": agree, "max_abs_diff": float(np.abs(ref - got).max()),
              "n_val": int(len(x_val)), "model_bytes": len(blob), "quant": quant}
    if check and (agree < min_agreement or report["max_abs_diff"] > max_diff):
        raise ParityError("Parity gagal, model tidak ditulis: %s" % json.dumps(report))

    graph = None
    if is_q:
        try:
            graph = export_graph(blob)
            from .purepy import PurePyBackend
            pp = PurePyBackend(graph)
            report["purepy_max_raw_diff"] = max(
                int(max(abs(a - b) for a, b in zip(pp.run([int(v) for v in o[0]]), [int(v) for v in o[1]])))
                for o in outs[:100])
        except UnsupportedOp as e:
            report["purepy"] = "tidak didukung: %s" % e

    k = min(n_vectors, len(x_val))
    meta = {
        "format": 1, "name": name, "labels": labels, "norm": norm, "ops": _ops(it),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "input": {"shape": [int(d) for d in i["shape"]], "dtype": "int8" if is_q else "float32",
                  "scale": float(si), "zero": int(zi)},
        "output": {"shape": [int(d) for d in o["shape"]], "dtype": "int8" if is_q else "float32",
                   "scale": float(so), "zero": int(zo)},
        "vectors": {"input": [outs[j][0].tolist() for j in range(k)],
                    "output": [outs[j][1].tolist() for j in range(k)]},
        "report": report,
    }
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / (name + ".tflite")).write_bytes(blob)
    meta["graph"], meta["graph_sha256"] = None, None
    if graph is not None:
        gtxt = json.dumps(graph, separators=(",", ":"))
        (out / (name + ".graph.json")).write_text(gtxt)
        meta["graph"] = name + ".graph.json"
        meta["graph_sha256"] = hashlib.sha256(gtxt.encode()).hexdigest()
    (out / (name + ".json")).write_text(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser(description="Konversi model Keras ke TFLite untuk raksadl")
    ap.add_argument("model"); ap.add_argument("repr_npy"); ap.add_argument("--val")
    ap.add_argument("--name", default="model"); ap.add_argument("--out", default="models")
    ap.add_argument("--labels"); ap.add_argument("--float32", action="store_true")
    a = ap.parse_args()
    import tensorflow as tf
    m = convert(tf.keras.models.load_model(a.model), np.load(a.repr_npy), a.out, a.name,
                x_val=np.load(a.val) if a.val else None,
                labels=a.labels.split(",") if a.labels else None,
                quant="float32" if a.float32 else "int8")
    print(json.dumps(m["report"], indent=1), "\nops:", m["ops"])


if __name__ == "__main__":
    main()
