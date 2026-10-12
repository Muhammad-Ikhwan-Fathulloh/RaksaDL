"""Gabungkan paket raksadl (bagian perangkat) menjadi SATU file raksadl.py untuk ditempel ke Wokwi.
python tools/bundle_wokwi.py [keluaran]   (default: wokwi/raksadl.py)"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORDER = ["pipeline", "purepy", "backends", "model"]


def bundle(out_path):
    parts = ["# raksadl.py — DIHASILKAN OTOMATIS oleh tools/bundle_wokwi.py dari paket raksadl. Jangan diedit.\n"]
    for name in ORDER:
        src = (ROOT / "raksadl" / (name + ".py")).read_text()
        src = re.sub(r"^from \.\w* ?import .*\n", "", src, flags=re.M)   # buang impor relatif
        parts.append("\n# ---- %s.py ----\n%s" % (name, src))
    parts.append('\n__version__ = "0.1.0"\n')
    pathlib.Path(out_path).write_text("".join(parts))
    return out_path


if __name__ == "__main__":
    print("ditulis:", bundle(sys.argv[1] if len(sys.argv) > 1 else ROOT / "wokwi" / "raksadl.py"))
