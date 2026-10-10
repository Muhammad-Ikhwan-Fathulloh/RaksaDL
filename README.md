# RaksaDL

Deep learning (TensorFlow Lite) untuk MicroPython dan CPython, dengan fokus pada **verifikasi**: setiap model dikirim bersama metadata (parameter kuantisasi, SHA-256, daftar op, vektor uji) sehingga perangkat bisa membuktikan bahwa ia menjalankan model yang sama dengan hasil training.

Pasangan dari [Raksa](https://github.com/Muhammad-Ikhwan-Fathulloh/Raksa) (ML klasik + WebSocket). RaksaDL khusus jaringan saraf (CNN, MLP, dst.).

## Alur
    # laptop/Colab
    pip install -e .[dev]
    python examples/train_vibration.py     # latih -> konversi int8 -> parity -> models/vibration.{tflite,json}
    pytest tests/
    python examples/run_vibration.py       # jalan di laptop

    # ESP32 (firmware microlite)
    mpremote mip install github:Muhammad-Ikhwan-Fathulloh/RaksaDL
    # salin models/vibration.* ke flash, lalu jalankan examples/run_vibration.py

## API singkat
    model = raksadl.Model.load("m.tflite", "m.json")   # sha256 diperiksa saat load
    model.selftest()                                  # (n_menyimpang, n) vs vektor uji
    model.run(x)  /  model.classify(x)  /  model.top_k(x, 3)
    win = raksadl.Window(size, channels, hop)           # jendela geser untuk deret waktu

`raksadl.convert` (hanya CPython) menolak menulis model bila parity float vs int8 di bawah ambang, dan mencetak daftar op TFLite.

## Batasan v0.1 (jujur)
- Satu tensor input dan satu output.
- Di perangkat butuh firmware **microlite** (mocleiri/tensorflow-micropython-examples). Op yang dipakai model harus terdaftar di firmware; bandingkan dengan `meta["ops"]`.
- Backend `MicroBackend` belum diuji di perangkat nyata; nama method microlite ada di satu kelas (`backends.py`) bila perlu disesuaikan.
- Model dimuat penuh ke RAM; arena default 16 KB (naikkan lewat `arena_size`).
