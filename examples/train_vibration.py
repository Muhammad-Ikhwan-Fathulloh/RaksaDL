"""CPython: latih 1D-CNN kecil, konversi ke int8, tulis models/vibration.{tflite,json}.
pip install tensorflow numpy  &&  python examples/train_vibration.py"""
import pathlib, sys
import numpy as np
import tensorflow as tf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from raksadl.convert import convert
import synth_vibration as S

tf.keras.utils.set_random_seed(0)
xs, ys = S.make_set(1200, 1)
xv, yv = S.make_set(300, 2)


def norm(x):
    a = np.array(x, dtype="float32").reshape(-1, S.T, S.C)
    return (a - np.array(S.MEAN, "float32")) / np.array(S.STD, "float32")


model = tf.keras.Sequential([
    tf.keras.layers.Input((S.T, S.C)),
    tf.keras.layers.Conv1D(8, 5, strides=2, activation="relu"),
    tf.keras.layers.Conv1D(16, 3, strides=2, activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(2, activation="softmax"),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(norm(xs), np.array(ys), epochs=25, batch_size=32, verbose=0)
print("akurasi float (val):", model.evaluate(norm(xv), np.array(yv), verbose=0)[1])

meta = convert(model, norm(xs), pathlib.Path(__file__).parent / "models", "vibration",
               x_val=norm(xv), labels=S.LABELS, norm={"mean": S.MEAN, "std": S.STD})
print("laporan:", meta["report"], "\nops:", meta["ops"])
