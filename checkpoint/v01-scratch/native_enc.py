import ctypes, sys, wave
import numpy as np
lib = ctypes.CDLL(sys.argv[1])
f32 = ctypes.POINTER(ctypes.c_float)
lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]; lib.needle_load.restype = ctypes.c_int
lib.needle_embed.argtypes = [ctypes.c_char_p, f32, ctypes.c_int, f32, ctypes.c_int]; lib.needle_embed.restype = ctypes.c_int
data = open(sys.argv[2], "rb").read(); assert lib.needle_load(data, len(data)) >= 0
w = wave.open(sys.argv[3])
pcm = np.ascontiguousarray(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0)
out = np.zeros(2_000_000, dtype=np.float32)
rc = lib.needle_embed(None, pcm.ctypes.data_as(f32), len(pcm), out.ctypes.data_as(f32), len(out))
print("rc", rc)
np.save(sys.argv[4], out[:rc].copy())
