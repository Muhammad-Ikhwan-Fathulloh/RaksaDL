# RaksaDL di Wokwi (ESP32 + MicroPython)

Menjalankan model CNN 1D int8 **yang sama** dengan hasil training, lewat backend `purepy` (Python murni, tanpa microlite).
Parity terhadap TFLite diperiksa: selisih output mentah 0 pada 100 jendela validasi, dan `selftest()` lolos (16/16).

## File
| File | Fungsi |
|---|---|
| `main.py` | Program utama (mode SIM atau MPU6050) |
| `raksadl.py` | Library RaksaDL dalam satu file (dihasilkan `tools/bundle_wokwi.py`) |
| `vibration.json` | Metadata model (kuantisasi, normalisasi, SHA-256, vektor uji) |
| `vibration.graph.json` | Graf model portabel (~5 KB) |
| `diagram.json` | Rangkaian: ESP32, MPU6050, LED hijau (GPIO2), LED merah (GPIO4), tombol (GPIO15) |

## Cara 1: Editor web Wokwi
1. Buka https://wokwi.com/projects/new/micropython-esp32.
2. Buat file `raksadl.py`, `vibration.json`, `vibration.graph.json`, lalu ganti isi `main.py` dan `diagram.json` dengan versi dari folder ini.
3. Jalankan. Serial monitor akan menampilkan `PARITY PASS`, lalu satu baris per jendela.
4. Tahan tombol kuning (GPIO15) untuk menyuntik getaran fault: label berubah ke `fault` dan LED merah menyala.

Jika `diagram.json` ditolak atau ada kabel merah, mulai dari template di atas dan sambungkan manual:
SDA→GPIO21, SCL→GPIO22, VCC→3V3, GND→GND; LED hijau→GPIO2, LED merah→GPIO4 (masing-masing lewat resistor 220 Ω); tombol→GPIO15 dan GND.
Nilai `env` pada board mengikuti template Wokwi saat ini; salin dari template jika berbeda.

## Cara 2: Wokwi untuk VS Code (file lebih mudah diunggah)
`python -m mpremote connect port:rfc2217://localhost:4000 fs cp main.py raksadl.py vibration.json vibration.graph.json :`
lalu `... run main.py`.

## Catatan jujur
- **Mode SIM:** MPU6050 di Wokwi bernilai statis, jadi sampel dibangkitkan sintetis dengan generator yang sama dengan data latih. Data sintetis ini bukan bukti performa di dunia nyata.
- **Kecepatan:** Python murni jauh lebih lambat dari microlite. Perhatikan nilai `inferensi ... ms` di serial. Jika lebih lama dari 320 ms (16 sampel pada 50 Hz), turunkan `FS_HZ` atau perbesar hop.
- **Perangkat nyata:** cukup ganti `Model.load(None, ..., backend="purepy")` menjadi `Model.load("vibration.tflite", "vibration.json")` pada firmware microlite, tanpa mengubah kode lain.
- **Belum diuji di Wokwi sungguhan:** semua ini sudah dijalankan di CPython dengan stub `machine`/`time`, bukan di simulator Wokwi. Kemungkinan perbedaan: heap MicroPython di simulator dan nama pin pada `diagram.json`.
- **Op yang didukung backend purepy:** CONV_2D, FULLY_CONNECTED, MEAN, SOFTMAX, LOGISTIC, RELU, RESHAPE, EXPAND_DIMS, SQUEEZE. Model dengan op lain (mis. MaxPool) ditolak saat konversi dengan pesan jelas.
