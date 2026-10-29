import ctypes, json, sys, time, os
lib = ctypes.CDLL(sys.argv[1])
lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.needle_init.restype = ctypes.c_int
lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
lib.needle_complete.restype = ctypes.c_int
lib.needle_reset.argtypes = []
lib.needle_reset.restype = None
lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
lib.needle_load.restype = ctypes.c_int
data = open(sys.argv[2], "rb").read()
assert lib.needle_load(data, len(data)) >= 0
tools = open(sys.argv[3], "rb").read()
t = time.perf_counter()
prefix = lib.needle_init(b"", tools, None)
print("init", prefix, f"{(time.perf_counter()-t)*1000:.1f} ms")
buf = ctypes.create_string_buffer(65536)
queries = ["what's it like in Lagos right now?", "set a timer for 12 minutes and play some Miles Davis"]
for q in queries:
    times = []
    for i in range(int(sys.argv[4]) if len(sys.argv) > 4 else 10):
        lib.needle_reset()
        t = time.perf_counter()
        rc = lib.needle_complete(q.encode(), None, 0, 512, buf, len(buf))
        times.append((time.perf_counter()-t)*1000)
    r = json.loads(buf.value.decode())
    times.sort()
    print(f"{q[:40]:40s} best {times[0]:.2f} ms median {times[len(times)//2]:.2f} ms  calls={[(c['name'], c['arguments']) for c in r.get('function_calls',[])]}")
    print("   ", {k: r.get(k) for k in ('prefill_tokens','prefill_tps','decode_tokens','decode_tps','time_to_first_token_ms','total_time_ms','confidence')})
