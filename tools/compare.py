import ctypes
import json
import re
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACT = ROOT / "ref/hf/needle3.cact"
WHEEL = next((ROOT / "ref/hf/wheel").glob("cactus_needle-*-manylinux2014_x86_64.whl"), None)
VENDOR_LIB = ROOT / "ref/hf/wheel/libneedle3.so"
OURS = ROOT / "out/aot/needle"
TOOLS = ROOT / "tools/tools.json"
QUERY = "what's it like in Lagos right now?"


def status_kb(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1])
    return 0


def vendor(repeat):
    if not VENDOR_LIB.exists():
        with zipfile.ZipFile(WHEEL) as wheel:
            VENDOR_LIB.write_bytes(wheel.read("needle/libneedle3.so"))
    lib = ctypes.CDLL(str(VENDOR_LIB))
    lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    data = CACT.read_bytes()
    start = time.perf_counter()
    assert lib.needle_load(data, len(data)) >= 0
    loaded = time.perf_counter()
    assert lib.needle_init(b"", TOOLS.read_bytes(), None) >= 0
    primed = time.perf_counter()
    out = ctypes.create_string_buffer(65536)
    times, last = [], b""
    for _ in range(repeat):
        t0 = time.perf_counter()
        rc = lib.needle_complete(QUERY.encode(), None, 0, 256, out, len(out))
        times.append((time.perf_counter() - t0) * 1000)
        assert rc >= 0, out.value
        last = out.value
    del data
    return {"engine": "vendor libneedle3", "load_ms": (loaded - start) * 1000, "prime_ms": (primed - loaded) * 1000,
            "median_ms": statistics.median(times), "min_ms": min(times), "peak_rss_kb": status_kb("VmHWM"),
            "anon_kb": status_kb("RssAnon"), "output": last.decode()[:300]}


def ours(repeat, threads):
    env = {"NEEDLE_THREADS": str(threads), "PATH": "/usr/bin:/bin"}
    process = subprocess.Popen([str(OURS), "run", str(CACT), str(TOOLS), QUERY, str(repeat)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = anon = 0
    while process.poll() is None:
        try:
            fields = dict(line.split(":", 1) for line in Path(f"/proc/{process.pid}/status").read_text().splitlines() if ":" in line)
            peak = max(peak, int(fields["VmHWM"].split()[0]))
            anon = max(anon, int(fields["RssAnon"].split()[0]))
        except (OSError, KeyError):
            pass
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    totals = [float(m) for m in re.findall(r"total ([\d.]+)ms", stderr)]
    prime = float(re.search(r"tokens in ([\d.]+)ms", stderr).group(1))
    return {"engine": f"needle-cs ({threads} thread{'s' if threads > 1 else ''})", "prime_ms": prime,
            "median_ms": statistics.median(totals), "min_ms": min(totals), "peak_rss_kb": peak, "anon_kb": anon,
            "output": stdout.strip()[-300:]}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    if mode == "vendor-child":
        print(json.dumps(vendor(repeat)))
        return
    rows = [json.loads(subprocess.check_output([sys.executable, __file__, "vendor-child", str(repeat)], text=True))]
    for threads in (1, 4, 8):
        rows.append(ours(repeat, threads))
    print(f"{'engine':<26}{'prime ms':>10}{'median ms':>11}{'best ms':>9}{'peak RSS MB':>13}{'private MB':>12}")
    for row in rows:
        print(f"{row['engine']:<26}{row['prime_ms']:>10.1f}{row['median_ms']:>11.1f}{row['min_ms']:>9.1f}{row['peak_rss_kb'] / 1024:>13.1f}{row['anon_kb'] / 1024:>12.1f}")
    for row in rows:
        print(f"\n{row['engine']}:\n  {row['output']}")


if __name__ == "__main__":
    main()
