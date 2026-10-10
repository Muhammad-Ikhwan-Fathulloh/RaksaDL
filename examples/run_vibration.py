# Jalan di ESP32 (firmware microlite) maupun laptop. Di ESP32 salin models/vibration.* ke flash.
import raksadl
from raksadl import Window

BASE = "examples/models/"
model = raksadl.Model.load(BASE + "vibration.tflite", BASE + "vibration.json")

bad, n = model.selftest()
print("PARITY", "PASS" if bad == 0 else "FAIL", "(%d/%d menyimpang)" % (bad, n))
if bad:
    raise SystemExit("Model tidak lolos parity")

win = Window(size=32, channels=3, hop=16)
# Ganti sumber ini dengan pembacaan akselerometer (mis. MPU6050) tiap ~10 ms.
import math
for t in range(96):
    s = [0.4 * math.sin(t / 2.5), 0.3 * math.sin(t / 2.5 + 1), 1.0 + 0.2 * math.sin(t / 2.5 + 2)]
    if win.push(s):
        i, score, label = model.classify(win.flatten())
        print("t=%d -> %s (%.2f)" % (t, label, score))
