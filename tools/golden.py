import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
CACT = ROOT / "ref/hf/needle3.cact"
SPM = ROOT / "ref/hf/tokenizer/tokenizer.model"
OUT = ROOT / "ref/golden"
LOCK = ROOT / "tools/ref.lock"

HEADER = "<48If"
RECORD = "<BBHIIIIQQII"
FP16, FP32, CQ, RAW = 1, 2, 3, 4
DECODE_BUDGET = 40
GROUP = 128
M32 = 0xFFFFFFFF
PRIME = 0x01000193
SEED = 0x9E3779B9

TOOLS = [
    {"name": "get_weather", "description": "Get the current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}},
    {"name": "set_timer", "description": "Set a countdown timer.",
     "parameters": {"type": "object", "properties": {"minutes": {"type": "integer", "description": "Length in minutes"}}, "required": ["minutes"]}},
    {"name": "play_music", "description": "Play a song or artist.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Song or artist"}}, "required": ["query"]}},
]
QUERIES = ["what's it like in Lagos right now?", "set a timer for 12 minutes and play some Miles Davis"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_of(query, tools):
    return ("<|im_start|>user\n<tools>" + json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
            + "</tools>\n" + query + "<|im_end|>\n<|im_start|>assistant\n")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def a8(x):
    peak = np.abs(x).max(-1, keepdims=True)
    scale = np.where(peak > 0, peak / 127.0, 1.0)
    return np.clip(np.round(x / scale), -128, 127) * scale


def unit(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def zc(x, scale):
    return (1.0 + scale) * x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def shift(x, k):
    if k == 0:
        return x
    out = np.zeros_like(x)
    if k < x.shape[0]:
        out[k:] = x[:-k]
    return out


def walsh(n):
    h = np.array([[1.0]], np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def sinkhorn(logits, iters=20):
    k = logits
    for _ in range(iters):
        k = k - np.log(np.exp(k - k.max(-1, keepdims=True)).sum(-1, keepdims=True)) - k.max(-1, keepdims=True)
        k = k - np.log(np.exp(k - k.max(-2, keepdims=True)).sum(-2, keepdims=True)) - k.max(-2, keepdims=True)
    return np.exp(k)


class Cact:
    def __init__(self, path):
        self.raw = Path(path).read_bytes()
        h = struct.unpack_from(HEADER, self.raw, 0)
        self.h = h
        self.n, cb_n = h[1], h[2]
        off = struct.calcsize(HEADER)
        self.cb = np.frombuffer(self.raw, np.float32, cb_n, off)
        off += cb_n * 4
        size = struct.calcsize(RECORD)
        self.recs = [struct.unpack_from(RECORD, self.raw, off + i * size) for i in range(self.n)]
        self.hada = walsh(GROUP)
        self.layers, self.sites = h[10], list(h[32:32 + h[31]])

    def shape(self, i):
        r = self.recs[i]
        return tuple(r[3:3 + r[1]])

    def dense(self, i):
        r = self.recs[i]
        dt = {FP16: np.float16, FP32: np.float32}[r[0]]
        return np.frombuffer(self.raw, dt, int(np.prod(self.shape(i))), r[7]).reshape(self.shape(i)).astype(np.float32)

    def blob(self, i):
        r = self.recs[i]
        return self.raw[r[7]:r[7] + r[8]]

    def cq(self, i, rows=None):
        r = self.recs[i]
        out, cols = self.shape(i)
        bits = r[10]
        pad = (cols + GROUP - 1) // GROUP * GROUP
        row_bytes = pad * bits // 8
        packed = np.frombuffer(self.raw, np.uint8, out * row_bytes, r[7]).reshape(out, row_bytes)
        norms = np.frombuffer(self.raw, np.float16, out * pad // GROUP, r[7] + out * row_bytes).reshape(out, pad // GROUP)
        if rows is not None:
            packed, norms = packed[rows], norms[rows]
        n = packed.shape[0]
        planes = np.unpackbits(packed, axis=1, bitorder="little").reshape(n, pad, bits)
        idx = (planes.astype(np.uint16) << np.arange(bits, dtype=np.uint16)).sum(-1)
        table = {2: self.cb[0:4], 3: self.cb[4:12], 4: self.cb[12:28]}[bits]
        unit_rows = table[idx].reshape(n, pad // GROUP, GROUP) * norms.astype(np.float32)[:, :, None]
        return (unit_rows @ self.hada).reshape(n, pad)[:, :cols]


def engram_indices(tokens, orders, heads, slots):
    u = tokens.astype(np.uint32)
    columns = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            acc = np.full_like(u, np.uint32((SEED * (oi * heads + h + 1)) & M32))
            for j in range(order):
                acc = (acc ^ shift(u, j)) * np.uint32(PRIME)
            acc = acc ^ (acc >> np.uint32(15))
            columns.append((acc % np.uint32(slots)).astype(np.int64))
    return np.stack(columns, -1)


class Model:
    def __init__(self, cact):
        self.c = cact
        h = cact.h
        self.d, self.heads, self.kv_heads, self.layers = h[7], h[8], h[9], h[10]
        self.qk, self.vd, self.hada_n, self.lanes = h[11], h[12], h[14], h[15]
        self.window, self.gmask = h[16], h[17] | (h[18] << 32)
        self.slots, self.sub = h[20], h[21]
        self.orders = list(h[26 + 1:26 + 1 + h[26]])
        self.rope = h[48]
        self.tables = len(self.orders) * 3
        self.embedding = cact.cq(0)
        self.cache = {}

    def w(self, i):
        if i not in self.cache:
            self.cache[i] = self.c.cq(i)
        return self.cache[i]

    def kron(self, z, a, b):
        t = z.shape[0]
        return np.einsum("tij,ik,jl->tkl", z.reshape(t, a.shape[0], b.shape[0]), a, b).reshape(t, -1)

    def engram(self, tokens, site, x_dim):
        c = self.c
        base = 552 + 4 * site
        t = len(tokens)
        idx = engram_indices(tokens, self.orders, 3, self.slots)
        rows = (np.arange(self.tables)[None, :] * self.slots + idx).reshape(-1)
        fetched = c.cq(base, rows).reshape(t, self.tables, self.sub)
        ok = np.stack([(np.arange(t) >= o - 1) for o in self.orders for _ in range(3)], -1).astype(np.float32)
        e = a8((fetched * ok[:, :, None]).reshape(t, -1))
        k = e @ self.w(base + 1).T
        v = e @ self.w(base + 2).T
        taps = c.dense(base + 3)
        v = sum(taps[j] * shift(v, j * 3) for j in range(4))
        return k, v

    def mlp(self, x, b):
        c = self.c
        d1, d2, b2, d3, d4 = (c.dense(b + 14 + i) for i in range(5))
        w = [c.dense(b + 19 + i) for i in range(6)]
        cond_v, cond_u = c.dense(b + 25), c.dense(b + 26)
        p1 = c.dense(550).astype(np.int64)
        p2 = c.dense(551).astype(np.int64)
        logits = x @ cond_v
        e = np.exp(logits - logits.max(-1, keepdims=True))
        cond = 1.0 + (e / e.sum(-1, keepdims=True)) @ cond_u
        z = np.pad(x, ((0, 0), (0, self.hada_n - self.d)))
        z = self.kron(d1 * z, w[0], w[1])[:, p1]
        z = d2 * cond * z + b2
        z = z * sigmoid(z)
        z = self.kron(z, w[2], w[3])[:, p2]
        z = self.kron(d3 * z, w[4], w[5])
        return (d4 * z)[:, :self.d]

    def attention(self, xq, b, layer, positions):
        c = self.c
        t = xq.shape[0]
        q = xq @ self.w(b + 1).T
        k = xq @ self.w(b + 2).T
        v = xq @ self.w(b + 3).T
        qt, kt, vt = c.dense(b + 4), c.dense(b + 5), c.dense(b + 6)
        q = sum(qt[j] * shift(q, j) for j in range(3))
        k = sum(kt[j] * shift(k, j) for j in range(3))
        v = sum(vt[j] * shift(v, j) for j in range(3))
        q = zc(q.reshape(t, self.heads, self.qk), c.dense(b + 7))
        k = zc(k.reshape(t, self.kv_heads, self.qk), c.dense(b + 8))
        v = v.reshape(t, self.kv_heads, self.vd)
        half = self.qk // 2
        freqs = 1.0 / (self.rope ** (np.arange(0, self.qk, 2, dtype=np.float32) / self.qk))
        ang = positions[:, None].astype(np.float32) * freqs[None, :]
        cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]

        def rot(x):
            x1, x2 = x[..., :half], x[..., half:]
            return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

        q, k = a8(rot(q)), a8(rot(k))
        v = a8(v)
        group = self.heads // self.kv_heads
        out = np.zeros((t, self.heads, self.vd), np.float32)
        is_global = bool(self.gmask >> layer & 1)
        dist = positions[:, None] - positions[None, :]
        allowed = dist >= 0
        if not is_global:
            allowed &= dist < self.window
        for h in range(self.heads):
            s = q[:, h] @ k[:, h // group].T / np.sqrt(self.qk)
            s = np.where(allowed, s, -np.inf)
            p = np.exp(s - s.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            out[:, h] = p @ v[:, h // group]
        out = out.reshape(t, -1) * sigmoid(xq @ self.w(b + 9).T)
        return a8(out) @ self.w(b + 10).T

    def forward(self, tokens, dump=None):
        c = self.c
        t = len(tokens)
        positions = np.arange(t)
        x0 = self.embedding[tokens] * np.sqrt(self.d)
        stream = np.repeat(x0[:, None, :], self.lanes, 1).astype(np.float32)
        eye = np.eye(self.lanes, dtype=np.float32)
        engrams = {s: self.engram(tokens, i, self.d) for i, s in enumerate(c.sites)}
        mhc = {n: c.dense(541 + i) for i, n in enumerate(["a_pre", "a_post", "a_res", "b_pre", "b_post", "b_res"])}
        phi = [self.w(547), self.w(548), self.w(549)]
        if dump is not None:
            x0.astype(np.float32).tofile(dump / "x0.bin")
        for l in range(self.layers):
            b = 1 + 27 * l
            lane = eye[l % self.lanes]
            nx = a8(unit(stream.reshape(t, -1)))
            p_pre = phi[0][l * 4:(l + 1) * 4]
            p_post = phi[1][l * 4:(l + 1) * 4]
            p_res = phi[2][l * 16:(l + 1) * 16]
            hpre = sigmoid(mhc["a_pre"][l] * (nx @ p_pre.T) + mhc["b_pre"][l] + (8 * lane - 4))
            u = np.einsum("tn,tnc->tc", hpre, stream)
            x = u
            if l in engrams:
                ek, ev = engrams[l]
                alpha = sigmoid((unit(x) * unit(ek)).sum(-1) / np.sqrt(self.d))
                x = x + alpha[:, None] * ev
            skip = x
            xq = a8(zc(x, c.dense(b)))
            att = self.attention(xq, b, l, positions)
            x = skip + sigmoid(c.dense(b + 12)[0]) * zc(att, c.dense(b + 11))
            x = x + self.mlp(zc(x, c.dense(b + 13)), b)
            y = x - u
            hpost = 2 * sigmoid(mhc["a_post"][l] * (nx @ p_post.T) + mhc["b_post"][l] + (-4 * (1 - lane)))
            res = (nx @ p_res.T).reshape(t, 4, 4)
            hres = sinkhorn(mhc["a_res"][l] * res + mhc["b_res"][l])
            stream = (np.einsum("tij,tjc->tic", hres, stream) + hpost[:, :, None] * y[:, None, :]).astype(np.float32)
            if dump is not None:
                stream.tofile(dump / f"stream{l:02d}.bin")
        x = zc(stream.mean(1), c.dense(572))
        return a8(x[-1:]) @ self.embedding.T


def parse_tokenizer(blob):
    n, pad, eos, bos, unk, dummy, byte_fb, _ = struct.unpack_from("<IIIIIBBH", blob, 0)
    off = struct.calcsize("<IIIIIBBH")
    pieces, scores, types = [], [], []
    for _ in range(n):
        score, kind, ln = struct.unpack_from("<fBH", blob, off)
        off += struct.calcsize("<fBH")
        pieces.append(blob[off:off + ln].decode("utf-8"))
        scores.append(score)
        types.append(kind)
        off += ln
    return pieces, scores, types, dummy, byte_fb, unk


class RefTokenizer:
    def __init__(self, blob):
        self.pieces, self.scores, self.types, self.dummy, self.byte_fb, self.unk = parse_tokenizer(blob)
        self.p2id = {p: i for i, p in enumerate(self.pieces)}
        self.byte_id = {int(p[3:5], 16): i for i, (p, t) in enumerate(zip(self.pieces, self.types)) if t == 4}
        self.markers = sorted((p for p, t in zip(self.pieces, self.types) if t == 3), key=len, reverse=True)

    def bpe(self, seg):
        syms = list(seg)
        while len(syms) > 1:
            best, at = None, -1
            for j in range(len(syms) - 1):
                idx = self.p2id.get(syms[j] + syms[j + 1])
                if idx is not None and (best is None or self.scores[idx] > best):
                    best, at = self.scores[idx], j
            if at < 0:
                break
            syms[at:at + 2] = [syms[at] + syms[at + 1]]
        ids = []
        for s in syms:
            idx = self.p2id.get(s)
            if idx is not None:
                ids.append(idx)
            elif self.byte_fb:
                ids.extend(self.byte_id[x] for x in s.encode("utf-8"))
            else:
                ids.append(self.unk)
        return ids

    def encode(self, text):
        esc = text.replace(" ", "▁")
        if self.dummy:
            esc = "▁" + esc
        ids, buf, i = [], [], 0
        while i < len(esc):
            marker = next((m for m in self.markers if esc.startswith(m, i)), None)
            if marker is not None:
                ids += self.bpe("".join(buf))
                buf = []
                ids.append(self.p2id[marker])
                i += len(marker)
            else:
                buf.append(esc[i])
                i += 1
        return ids + self.bpe("".join(buf))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cact = Cact(CACT)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(SPM))
    ref = RefTokenizer(cact.blob(cact.n - 1))
    cases = []
    for query in QUERIES:
        prompt = prompt_of(query, TOOLS)
        sp_ids = [2] + sp.Encode(prompt, out_type=int)
        ref_ids = [2] + ref.encode(prompt)
        print("tokenizer agree:", sp_ids == ref_ids, len(sp_ids))
        cases.append({"query": query, "prompt": prompt, "sp_ids": sp_ids, "ref_ids": ref_ids})
    probes = ["hello world", "  two  spaces\ttab\nnewline ünïcödé 日本語 🙂", "<tools>[]</tools>\n12 minutes, \"quoted\" {json: [1,2]}"]
    probe_cases = [{"text": p, "sp_ids": sp.Encode(p, out_type=int), "ref_ids": ref.encode(p)} for p in probes]
    for p in probe_cases:
        print("probe agree:", p["sp_ids"] == p["ref_ids"], p["sp_ids"][:12])
    model = Model(cact)
    dump = OUT / "case0"
    dump.mkdir(exist_ok=True)
    ids = np.array(cases[0]["sp_ids"])
    logits = model.forward(ids, dump)[0]
    logits.astype(np.float32).tofile(dump / "logits.bin")
    ids.astype(np.int32).tofile(dump / "ids.bin")
    print("first token:", int(logits.argmax()), repr(sp.Decode([int(logits.argmax())])))
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else DECODE_BUDGET
    results = []
    for case in cases:
        seq = list(case["sp_ids"])
        generated = []
        for _ in range(budget):
            nxt = int(model.forward(np.array(seq))[0].argmax())
            if nxt == 1:
                break
            seq.append(nxt)
            generated.append(nxt)
        case["generated"] = generated
        case["text"] = sp.Decode(generated)
        print(case["query"], "->", case["text"])
        results.append(case)
    provenance = {"model_sha256": file_sha256(CACT), "ref_lock_sha256": file_sha256(LOCK)}
    (OUT / "golden.json").write_text(json.dumps({"budget": budget, **provenance, "cases": results, "probes": probe_cases}, ensure_ascii=False))


if __name__ == "__main__":
    main()
