# RaksaDL

Deep learning (TensorFlow Lite) untuk MicroPython dan CPython, dengan fokus pada **verifikasi**: setiap model dikirim bersama metadata (parameter kuantisasi, SHA-256, daftar op, vektor uji) sehingga perangkat bisa membuktikan bahwa ia menjalankan model yang sama dengan hasil training.

Pasangan dari [Raksa](https://github.com/Muhammad-Ikhwan-Fathulloh/Raksa) (ML klasik + WebSocket). RaksaDL khusus jaringan saraf (CNN, MLP, dst.).

## Instalasi

### Via `pip` (CPython / Laptop Testing)

```bash
pip install raksa-dl
```

Atau development install dari repo lokal:

```bash
pip install -e .[dev]
```

### Via `mip` (MicroPython on Device)

#### 1. Menggunakan `mpremote` (Disarankan)

Hubungkan board ESP32/RP2040 (dengan **firmware microlite**) ke PC via USB, lalu jalankan:

```bash
mpremote mip install github:Muhammad-Ikhwan-Fathulloh/RaksaDL
```

#### 2. Langsung di REPL MicroPython

Pastikan board sudah terkoneksi Wi-Fi, lalu:

```python
import mip
mip.install("github:Muhammad-Ikhwan-Fathulloh/RaksaDL")
```

> ⚠️ **Prasyarat Perangkat**: Butuh firmware MicroPython dengan `microlite` (TensorFlow Lite Micro). Cek: `import microlite`. Op yang dipakai model harus tersedia di firmware; bandingkan dengan `meta["ops"]`.

---

## Alur Kerja End-to-End

```
# 1. LATIH & KONVERSI (Laptop / Colab)
python examples/train_vibration.py     # -> vibration.tflite + vibration.json
pytest tests/                          # unit test converter & parity

# 2. VERIFIKASI (Laptop)
python examples/run_vibration.py       # pastikan model lulus selftest()

# 3. DEPLOY (ESP32 dengan microlite)
mpremote mip install github:Muhammad-Ikhwan-Fathulloh/RaksaDL
mpremote cp models/vibration.tflite :
mpremote cp models/vibration.json :
mpremote run examples/run_vibration.py
```

---

## Contoh Kode Minimalis (< 15 baris)

```python
import math, raksadl
from raksadl import Window

# 1. Load model — SHA-256 & vektor uji dicek otomatis
model = raksadl.Model.load("vibration.tflite", "vibration.json")
bad, n = model.selftest()
print("PARITY:", "PASS ✓" if bad == 0 else "FAIL ✗", f"({bad}/{n} menyimpang)")

# 2. Pipeline deret waktu: jendela 32 x 3 channel (ax, ay, az), hop = 16
win = Window(size=32, channels=3, hop=16)

# 3. Push sampel (ganti dengan baca I2C/SPI sensor nyata)
for t in range(96):
    s = [0.4*math.sin(t/2.5), 0.3*math.sin(t/2.5+1), 1.0+0.2*math.sin(t/2.5+2)]
    if win.push(s):
        idx, skor, label = model.classify(win.flatten())
        print(f"t={t} → {label} ({skor:.2f})")
```

---

## Contoh Lengkap & Simulasi Wokwi

Contoh siap pakai tersedia di folder [`examples/`](examples).

### 🧪 Contoh Wokwi MicroPython (Simulator Online)

File-file ini bisa langsung di-upload ke [Wokwi](https://wokwi.com/) — buat project MicroPython lalu salin `diagram.json` dan `main.py`.

| File | Skenario | Komponen Wokwi |
|---|---|---|
| [`wokwi/wokwi_mpu6050_vibration.py`](examples/wokwi/wokwi_mpu6050_vibration.py) | Klasifikasi getaran dengan MPU6050 (3-axis accel) + Window sliding | ESP32, MPU6050 (I2C), LED |
| [`wokwi/wokwi_dht_anomaly.py`](examples/wokwi/wokwi_dht_anomaly.py) | Deteksi anomali suhu/kelembaban dengan Z-score rolling | ESP32, DHT22, OLED SSD1306 |
| [`wokwi/wokwi_light_classifier.py`](examples/wokwi/wokwi_light_classifier.py) | Klasifikasi intensitas cahaya + ambang adaptif dari simulasi model | ESP32, LDR (ADC), Potensiometer, LED RGB |
| [`wokwi/diagram_mpu6050.json`](examples/wokwi/diagram_mpu6050.json) | Rangkaian Wokwi: ESP32 + MPU6050 + LED indikator | (file Wokwi) |
| [`wokwi/diagram_dht_oled.json`](examples/wokwi/diagram_dht_oled.json) | Rangkaian Wokwi: ESP32 + DHT22 + SSD1306 | (file Wokwi) |

> 💡 **Catatan Wokwi**: Firmware Wokwi standar tidak menyertakan `microlite` (TFLite Micro). Contoh di atas menggunakan **wrapper simulasi** (`RaksaDLMock`) yang mereplikasi API `Model`, `Window`, `classify()` — sehingga logika pipeline, pengambilan data sensor, dan thresholding bisa diuji 100% di simulator sebelum flash ke perangkat nyata dengan firmware microlite.

### Contoh Hardware & Produksi (ESP32 / RP2040)

| File | Skenario | Sensor / Input |
|---|---|---|
| [`run_vibration.py`](examples/run_vibration.py) | Contoh utama: selftest → pipeline Window → klasifikasi getaran | Akselerometer (MPU6050/LSM6DS3) |
| [`train_vibration.py`](examples/train_vibration.py) | Training CNN 1D sederhana → quantisasi int8 → export metadata | TensorFlow/Keras (CPython only) |

---

## Referensi API

### Inti `raksadl.Model`

| Method/Class | Deskripsi |
|---|---|
| `Model.load(tflite_path, meta_path, **kw)` | Muat model + metadata. **SHA-256 diverifikasi otomatis** jika `verify=True` (default). |
| `Model(blob, meta, arena_size=16384, verify=True)` | Konstruktor langsung dari bytes blob. |
| `model.selftest(tol=None)` | Jalankan *golden vectors* dari metadata. Return `(n_menyimpang, n_total)`. |
| `model.run(x, normalized=False)` | Inferensi mentah, return array output. Input dicek NaN & panjangnya. |
| `model.classify(x, normalized=False)` | Return tuple `(indeks, skor, label)` — label dari `meta.labels` jika ada. |
| `model.top_k(x, k=3, normalized=False)` | Return list `[(idx, skor), ...]` top-k tertinggi. |
| `model.normalize(x)` | Pra-proses: standardisasi per-channel menggunakan `meta.norm.{mean,std}`. |

### Pipeline `raksadl.pipeline`

| Class/Function | Deskripsi |
|---|---|
| `Window(size, channels, hop=None)` | Buffer melingkar. `push(sample)` return `True` tiap `hop` sampel bila jendela penuh. |
| `window.flatten()` | Ubah buffer jadi flat array (tertua→terbaru) siap feed ke model. |
| `window.ready` / `window.clear()` | Properti status & reset buffer. |
| `softmax(v)` / `argmax(v)` / `top_k(v, k)` | Fungsi utilitas pasca-proses (pure Python, `@micropython.native`-friendly). |

### Converter `raksadl.convert` (hanya CPython)

| Fungsi | Deskripsi |
|---|---|
| `convert_saved_model(saved_model_dir, out_prefix, ...)` | Kuantisasi int8 + parity check → output `.tflite` + `.json`. |

**Metadata JSON** yang dibangkitkan converter berisi: `sha256`, `input/output` (shape, dtype, scale, zero_point), `norm` (mean/std per channel), `labels`, `ops` (daftar TFLite op), `vectors` (golden input/output untuk selftest).

---

## Batasan v0.1 Transparan

- Satu tensor input dan satu output tensor saja.
- Di perangkat nyata: butuh firmware **microlite** (`mocleiri/tensorflow-micropython-examples`). Nama method microlite terpusat di [`RaksaDL/backends.py`](RaksaDL/backends.py) jika perlu penyesuaian.
- Model dimuat penuh ke RAM; arena default 16 KB — naikkan lewat parameter `arena_size` bila model besar.
- Contoh Wokwi menggunakan mock inference (bukan TFLite sesungguhnya) karena keterbatasan firmware simulator.

---

## Ucapan Terima Kasih

Rangkaian kuantisasi, operator, dan pola verifikasi RaksaDL terinspirasi dari [Raksa](https://github.com/Muhammad-Ikhwan-Fathulloh/Raksa) dan pustaka [NocML C++](https://github.com/Nocturnailed-Community/NocML) untuk Arduino. Special credits kepada Noc Lab untuk fondasi literasi TinyML hardware developer di Indonesia.

---

## Lisensi

Dikembangkan oleh **Noc Lab** untuk pengembangan kapabilitas TinyML pada perangkat terbatas. Lisensi: MIT License.
