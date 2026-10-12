# main.py — RaksaDL di Wokwi (ESP32 + MicroPython). Klasifikasi getaran 3-sumbu dengan CNN 1D int8.
# Mode SIM=True: sampel akselerometer sintetis (MPU6050 di Wokwi bernilai statis); tombol = suntik getaran "fault".
# Mode SIM=False: baca MPU6050 asli lewat I2C (untuk perangkat nyata).
import math, random, time
from machine import Pin, I2C
from raksadl import Model, Window

SIM = True
FS_HZ = 50              # laju sampel
RUN_SELFTEST = True

led_ok = Pin(2, Pin.OUT)       # hijau: normal
led_bad = Pin(4, Pin.OUT)      # merah: fault
btn = Pin(15, Pin.IN, Pin.PULL_UP)   # ditekan = suntik getaran fault (hanya mode SIM)


def gauss(sd):
    u1 = max(random.random(), 1e-9)
    return sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * random.random())


class SimAccel:
    """Generator sama dengan data latih (sinus + derau; fault = tambahan komponen frekuensi tinggi)."""

    def __init__(self):
        self.t = 0
        self.ph = random.random() * 6.28

    def read(self, fault):
        t, s = self.t, []
        self.t += 1
        for c in range(3):
            v = 0.4 * math.sin(2 * math.pi * t / 16 + self.ph + c) + (1.0 if c == 2 else 0.0) + gauss(0.05)
            if fault:
                v += 0.6 * math.sin(2 * math.pi * t / 3.0 + self.ph) + gauss(0.15)
            s.append(v)
        return s


class MPU6050:
    def __init__(self, i2c, addr=0x68):
        self.i2c, self.addr = i2c, addr
        i2c.writeto_mem(addr, 0x6B, b"\x00")       # bangunkan sensor

    def read(self, fault=False):
        d = self.i2c.readfrom_mem(self.addr, 0x3B, 6)
        out = []
        for i in range(3):
            v = (d[2 * i] << 8) | d[2 * i + 1]
            out.append((v - 65536 if v & 0x8000 else v) / 16384.0)   # +-2 g
        return out


def main(max_windows=None):
    model = Model.load(None, "vibration.json", graph_path="vibration.graph.json", backend="purepy")
    if RUN_SELFTEST:
        bad, n = model.selftest()
        print("PARITY", "PASS" if bad == 0 else "FAIL", "(%d/%d menyimpang)" % (bad, n))
        if bad:
            led_bad.on()
            raise SystemExit("Model tidak lolos parity")

    src = SimAccel() if SIM else MPU6050(I2C(0, scl=Pin(22), sda=Pin(21), freq=400000))
    win = Window(size=32, channels=3, hop=16)
    period, done = 1000 // FS_HZ, 0
    print("Mulai. Tahan tombol (GPIO15) untuk menyuntik getaran fault." if SIM else "Mulai membaca MPU6050.")
    while True:
        t0 = time.ticks_ms()
        fault = SIM and btn.value() == 0
        if win.push(src.read(fault)):
            t1 = time.ticks_us()
            i, score, label = model.classify(win.flatten())
            us = time.ticks_diff(time.ticks_us(), t1)
            led_ok.value(1 if i == 0 else 0)
            led_bad.value(1 if i == 1 else 0)
            print("%s (%.2f) | inferensi %d ms%s" % (label, score, us // 1000, " | fault disuntik" if fault else ""))
            done += 1
            if max_windows and done >= max_windows:
                return
        wait = period - time.ticks_diff(time.ticks_ms(), t0)
        if wait > 0:
            time.sleep_ms(wait)


if __name__ == "__main__":
    main()
