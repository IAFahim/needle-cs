import ctypes, json, sys, wave
import numpy as np
lib = ctypes.CDLL(sys.argv[1]); f32 = ctypes.POINTER(ctypes.c_float)
lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]; lib.needle_load.restype = ctypes.c_int
lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]; lib.needle_init.restype = ctypes.c_int
lib.needle_complete.argtypes = [ctypes.c_char_p, f32, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]; lib.needle_complete.restype = ctypes.c_int
lib.needle_reset.argtypes = []
for p in (sys.argv[2], sys.argv[3]):
    d = open(p, "rb").read(); assert lib.needle_load(d, len(d)) >= 0
tools = open(sys.argv[4], "rb").read()
assert lib.needle_init(b"", tools, None) >= 0
buf = ctypes.create_string_buffer(1 << 16)
for path in sys.argv[5:]:
    w = wave.open(path); ch = w.getnchannels(); sr = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1: pcm = pcm.reshape(-1, ch).mean(1)
    if sr != 16000:
        idx = np.arange(0, len(pcm), sr / 16000.0); pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
    pcm = np.ascontiguousarray(pcm); lib.needle_reset()
    rc = lib.needle_complete(None, pcm.ctypes.data_as(f32), len(pcm), 512, buf, len(buf))
    print("rc", rc, buf.value.decode()[:600])
    r = json.loads(buf.value.decode())
    print(path.split("/")[-1], "->", [(c["name"], c["arguments"]) for c in r.get("function_calls", [])], "conf", r.get("confidence"), "| heard:", r.get("audio_text"))
