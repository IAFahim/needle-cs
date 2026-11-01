import ctypes, json, sys, wave, time
import numpy as np
lib = ctypes.CDLL(sys.argv[1])
f32 = ctypes.POINTER(ctypes.c_float)
lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]; lib.needle_load.restype = ctypes.c_int
lib.needle_transcribe.argtypes = [f32, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
lib.needle_transcribe.restype = ctypes.c_int
data = open(sys.argv[2], "rb").read()
assert lib.needle_load(data, len(data)) >= 0
buf = ctypes.create_string_buffer(1 << 20)
for path in sys.argv[3:]:
    w = wave.open(path); n = w.getnframes(); ch = w.getnchannels(); sr = w.getframerate()
    pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1: pcm = pcm.reshape(-1, ch).mean(1)
    if sr != 16000:
        idx = np.arange(0, len(pcm), sr / 16000.0); pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
    arr = pcm.ctypes.data_as(f32)
    for kw in (b"", b"Omeprazole") if "hotword" in path else (b"",):
        t = time.perf_counter()
        rc = lib.needle_transcribe(arr, len(pcm), b"", kw, 0, buf, len(buf))
        ms = (time.perf_counter() - t) * 1000
        r = json.loads(buf.value.decode())
        print(f"{path.split('/')[-1]} kw={kw.decode() or '-'} {ms:.0f} ms: {r.get('text','')[:330]}")
