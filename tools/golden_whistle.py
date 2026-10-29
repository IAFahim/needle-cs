import json
import sys
import wave
from pathlib import Path

import numpy as np

from golden import Cact, RefTokenizer, a8, shift, sigmoid, sinkhorn, unit, walsh, zc

ROOT = Path(__file__).resolve().parent.parent
CACT = ROOT / "ref/hf/whistle/whistle.cact"
AUDIO = ROOT / "ref/audio"
OUT = ROOT / "ref/golden/whistle"

SAMPLE_RATE = 16000
BOS, EOS = 2, 1
MAX_SAMPLES = 30 * SAMPLE_RATE
CLIPS = {
    "record": ("record.wav", None),
    "test": ("test.wav", None),
    "hotword": ("hotword.wav", None),
    "long25": ("test_long.wav", 25 * SAMPLE_RATE),
}
ABSOLUTE_FLOOR = 2.0 ** -24
RELATIVE_FLOOR = 1e-8


def read_wav(path, limit=None):
    with wave.open(str(path)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    assert width == 2
    pcm = np.frombuffer(raw, "<i2").reshape(-1, channels).mean(axis=1, dtype=np.float32) / np.float32(32768.0)
    if rate != SAMPLE_RATE:
        import soxr

        pcm = soxr.resample(pcm.astype(np.float32), rate, SAMPLE_RATE, quality="HQ")
    pcm = np.ascontiguousarray(pcm, np.float32)
    return pcm[:limit] if limit else pcm


class Cursor:
    def __init__(self):
        self.next = 0

    def take(self, count=1):
        first = self.next
        self.next += count
        return first if count == 1 else list(range(first, first + count))


class Whistle:
    def __init__(self, cact, quant=True, rotated=False, floor="relative"):
        self.floor = floor
        self.c = cact
        self.quant = quant
        self.rotated = rotated
        self.cache = {}
        h = cact.h
        self.dim, self.heads_total, self.kv_heads, self.layers = h[7], h[8], h[9], h[10]
        self.qk, self.vd, self.hada_n, self.lanes = h[11], h[12], h[14], h[15]
        self.rope = h[48]
        self.orders = list(h[27:27 + h[26]])
        self.engram_slots, self.engram_sub, self.engram_tables = h[20], h[21], h[22]
        self.engram_heads = self.engram_tables // len(self.orders)
        self.sites = list(h[32:32 + h[31]])
        self.cact_hada = cact.hada
        self.index_layout()

    def index_layout(self):
        cur = Cursor()
        d, q = cur.take, cur.take
        self.embedding_id = q()
        self.decoder = []
        for _ in range(self.layers):
            layer = {"norm_in": d(), "q": q(), "k": q(), "v": q(), "q_taps": d(), "k_taps": d(), "v_taps": d(),
                     "q_norm": d(), "k_norm": d(), "gate": q(), "out": q(), "post_norm": d(), "attn_gate": d(),
                     "pre_hada": d(), "mlp": cur.take(13)}
            self.decoder.append(layer)
        self.hyper = {n: d() for n in ["a_pre", "a_post", "a_res", "b_pre", "b_post", "b_res"]}
        self.hyper.update({n: q() for n in ["phi_pre", "phi_post", "phi_res"]})
        self.perm_a, self.perm_b = d(), d()
        self.engram = [{"tables": q(), "key": q(), "value": q(), "taps": d()} for _ in self.sites]
        self.final_norm = d()
        manifest = self.dense(d())
        self.enc_layers, self.conv_kernel, self.stem, self.mels = int(manifest[3]), int(manifest[5]), int(manifest[6]), int(manifest[8])
        self.fft, self.window, self.hop = int(manifest[10]), int(manifest[11]), int(manifest[12])
        self.cross = [{"norm": d(), "q": q(), "k": q(), "v": q(), "gate": q(), "out": q(), "q_norm": d(),
                       "k_norm": d(), "post_norm": d(), "mix": d()} for _ in range(self.layers)]
        self.position_mix = d()
        self.encoder = []
        for _ in range(self.enc_layers):
            layer = {n: d() for n in ["pre_mlp_norm0", "attn_norm", "post_attn_norm", "conv_norm", "conv_out_norm", "pre_mlp_norm"]}
            layer.update({n: q() for n in ["q", "k", "v", "gate", "out"]})
            layer.update({"q_norm": d(), "k_norm": d(), "attn_gate": d(), "pw1": q(), "pw2": q(), "depthwise": d(),
                          "mlp0": cur.take(13), "mlp": cur.take(13)})
            self.encoder.append(layer)
        self.enc_hyper = {n: d() for n in ["a_pre", "a_post", "a_res", "b_pre", "b_post", "b_res"]}
        self.enc_hyper.update({n: q() for n in ["phi_pre", "phi_post", "phi_res"]})
        self.enc_final_norm = d()
        self.stem_first, self.stem_dw1, self.stem_pw1, self.stem_dw2, self.stem_pw2, self.stem_out = d(), d(), q(), d(), q(), q()
        self.mel = d()

    def dense(self, i):
        if i not in self.cache:
            self.cache[i] = self.c.dense(i)
        return self.cache[i]

    def weight(self, i):
        if i not in self.cache:
            self.cache[i] = self.c.cq(i)
        return self.cache[i]

    def rotate_quantize(self, x):
        shape = x.shape
        groups = x.reshape(-1, shape[-1] // 128, 128).astype(np.float32) @ self.cact_hada
        peak = np.abs(groups).max(-1, keepdims=True)
        scale = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float32)
        quantized = np.clip(np.round(groups / scale), -128, 127) * scale
        return (quantized @ self.cact_hada).reshape(shape).astype(np.float32)

    def mm(self, x, i, rows=None):
        w = self.weight(i) if rows is None else self.weight(i)[rows]
        return (self.rotate_quantize(x) if self.rotated else x) @ w.T

    def act(self, x, site):
        return a8(x) if self.quant is True or (self.quant and site in self.quant) else x

    def mel_features(self, pcm):
        frames = len(pcm) // self.hop
        padded = np.pad(pcm.astype(np.float32), (self.window // 2, self.window // 2))
        index = np.arange(frames)[:, None] * self.hop + np.arange(self.window)[None, :]
        window = np.hanning(self.window).astype(np.float32)
        spectrum = np.fft.rfft(padded[index] * window, self.fft)
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
        energy = power @ self.dense(self.mel)
        floor = ABSOLUTE_FLOOR if self.floor == "absolute" else RELATIVE_FLOOR * energy.max()
        x = np.log(energy + np.float32(max(floor, 1e-30))).astype(np.float32)
        mean = x.mean(0, dtype=np.float64)
        std = np.sqrt(((x - mean) ** 2).sum(0, dtype=np.float64) / max(frames - 1, 1))
        return ((x - mean) / (std + 1e-5)).astype(np.float32)

    def depthwise(self, x, taps):
        t, f, _ = x.shape
        taps = taps.reshape(3, 3, -1)
        out_t, out_f = (t + 1) // 2, (f + 1) // 2
        padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        out = np.zeros((out_t, out_f, taps.shape[-1]), np.float32)
        for dt in range(3):
            for df in range(3):
                out += padded[dt:dt + 2 * out_t:2, df:df + 2 * out_f:2] * taps[dt, df]
        return out

    def stem_forward(self, features):
        x = self.depthwise(features[:, :, None], self.dense(self.stem_first))
        x = self.mm(self.depthwise(x * sigmoid(x), self.dense(self.stem_dw1)), self.stem_pw1)
        x = self.mm(self.depthwise(x * sigmoid(x), self.dense(self.stem_dw2)), self.stem_pw2)
        x = x * sigmoid(x)
        flat = np.swapaxes(x, 1, 2).reshape(x.shape[0], -1)
        return self.mm(self.act(flat, "stem"), self.stem_out)

    def mlp(self, x, ids):
        c = self.c
        d1, d2, b2, d3, d4 = (self.dense(i) for i in ids[:5])
        w = [self.dense(i) for i in ids[5:11]]
        cond_v, cond_u = self.dense(ids[11]), self.dense(ids[12])
        p1, p2 = self.dense(self.perm_a).astype(np.int64), self.dense(self.perm_b).astype(np.int64)
        logits = x @ cond_v
        e = np.exp(logits - logits.max(-1, keepdims=True))
        cond = 1.0 + (e / e.sum(-1, keepdims=True)) @ cond_u
        z = np.pad(x, ((0, 0), (0, self.hada_n - self.dim)))

        def kron(z, a, b):
            return np.einsum("tij,ik,jl->tkl", z.reshape(len(z), a.shape[0], b.shape[0]), a, b).reshape(len(z), -1)

        z = kron(d1 * z, w[0], w[1])[:, p1]
        z = d2 * cond * z + b2
        z = z * sigmoid(z)
        z = kron(z, w[2], w[3])[:, p2]
        z = kron(d3 * z, w[4], w[5])
        return (d4 * z)[:, :self.dim]

    def rotate(self, x, positions):
        half = self.qk // 2
        freqs = 1.0 / (self.rope ** (np.arange(0, self.qk, 2, dtype=np.float32) / self.qk))
        angle = positions[:, None].astype(np.float32) * freqs[None, :]
        cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def attend(self, q, k, v, allowed):
        group = q.shape[1] // k.shape[1]
        out = np.zeros((q.shape[0], q.shape[1], v.shape[-1]), np.float32)
        for h in range(q.shape[1]):
            s = q[:, h] @ k[:, h // group].T / np.sqrt(self.qk)
            s = np.where(allowed, s, -np.inf)
            p = np.exp(s - s.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            out[:, h] = p @ v[:, h // group]
        return out

    def hyper_mix(self, stream, hyper, l, lane):
        t = stream.shape[0]
        nx = self.act(unit(stream.reshape(t, -1)), "route")
        a_pre, a_post, a_res = (self.dense(hyper[n])[l] for n in ["a_pre", "a_post", "a_res"])
        b_pre, b_post, b_res = (self.dense(hyper[n])[l] for n in ["b_pre", "b_post", "b_res"])
        hpre = sigmoid(a_pre * self.mm(nx, hyper["phi_pre"], slice(l * 4, l * 4 + 4)) + b_pre + (8 * lane - 4))
        hpost = 2 * sigmoid(a_post * self.mm(nx, hyper["phi_post"], slice(l * 4, l * 4 + 4)) + b_post + (-4 * (1 - lane)))
        hres = sinkhorn(a_res * self.mm(nx, hyper["phi_res"], slice(l * 16, l * 16 + 16)).reshape(t, 4, 4) + b_res)
        return np.einsum("tn,tnc->tc", hpre, stream), hpost, hres

    def encoder_block(self, u, w):
        t = u.shape[0]
        x = u + 0.5 * self.mlp(zc(u, self.dense(w["pre_mlp_norm0"])), w["mlp0"])
        skip = x
        h = self.act(zc(x, self.dense(w["attn_norm"])), "attn_in")
        q = self.mm(h, w["q"]).reshape(t, self.heads_total, self.qk)
        k = self.mm(h, w["k"]).reshape(t, self.kv_heads, self.qk)
        v = self.mm(h, w["v"]).reshape(t, self.kv_heads, self.vd)
        positions = np.arange(t)
        q = self.rotate(zc(q, self.dense(w["q_norm"])), positions)
        k = self.rotate(zc(k, self.dense(w["k_norm"])), positions)
        q, k, v = self.act(q, "qkv"), self.act(k, "qkv"), self.act(v, "qkv")
        out = self.attend(q, k, v, np.ones((t, t), bool)).reshape(t, -1)
        out = out * sigmoid(self.mm(h, w["gate"]))
        out = self.mm(self.act(out, "attn_out"), w["out"])
        x = skip + sigmoid(self.dense(w["attn_gate"])[0]) * zc(out, self.dense(w["post_attn_norm"]))
        h = self.mm(self.act(zc(x, self.dense(w["conv_norm"])), "pw1_in"), w["pw1"])
        h = h[:, :self.dim] * sigmoid(h[:, self.dim:])
        taps = self.dense(w["depthwise"])
        pad = self.conv_kernel // 2
        padded = np.pad(h, ((pad, pad), (0, 0)))
        h = sum(taps[j] * padded[j:j + t] for j in range(self.conv_kernel))
        h = zc(h, self.dense(w["conv_out_norm"]))
        mode = getattr(self, "pw2_mode", "after")
        if mode == "before":
            h = self.act(h, "pw2_in")
        h = h * sigmoid(h)
        if mode == "after":
            h = self.act(h, "pw2_in")
        x = x + self.mm(h, w["pw2"])
        return x + 0.5 * self.mlp(zc(x, self.dense(w["pre_mlp_norm"])), w["mlp"])

    def encode(self, pcm):
        pcm = pcm[:MAX_SAMPLES]
        stages = {}
        features = self.mel_features(pcm)
        stages["mel"] = features
        stem = self.stem_forward(features).astype(np.float32)
        stages["stem"] = stem
        stream = np.repeat(stem[:, None, :], 4, 1).astype(np.float32)
        for l in range(self.enc_layers):
            u, hpost, hres = self.hyper_mix(stream, self.enc_hyper, l, np.eye(4, dtype=np.float32)[l % 4])
            y = self.encoder_block(u, self.encoder[l]) - u
            stream = (np.einsum("tij,tjc->tic", hres, stream) + hpost[:, :, None] * y[:, None, :]).astype(np.float32)
            stages[f"layer{l}"] = stream
        encoded = zc(stream.mean(1), self.dense(self.enc_final_norm)).astype(np.float32)
        stages["encoder"] = encoded
        return stages

    def memory(self, encoded):
        t = len(encoded)
        step = -np.log(10000.0) / (self.dim // 2 - 1)
        angle = np.arange(t)[:, None] * np.exp(step * np.arange(self.dim // 2))[None, :]
        position = np.concatenate([np.sin(angle), np.cos(angle)], -1)
        mix = sigmoid(self.dense(self.position_mix)[0])
        return self.act((encoded + mix * position).astype(np.float32), "memory")

    def cross_memory(self, mem):
        keys, values = [], []
        for c in self.cross:
            k = self.mm(mem, c["k"]).reshape(len(mem), self.heads_total, self.qk)
            v = self.mm(mem, c["v"]).reshape(len(mem), self.heads_total, self.vd)
            keys.append(self.act(zc(k, self.dense(c["k_norm"])), "qkv"))
            values.append(self.act(v, "qkv"))
        return keys, values

    def engram_features(self, tokens, site):
        e = self.engram[site]
        t = len(tokens)
        u = tokens.astype(np.uint32)
        columns = []
        for oi, order in enumerate(self.orders):
            for h in range(self.engram_heads):
                acc = np.full_like(u, np.uint32((0x9E3779B9 * (oi * self.engram_heads + h + 1)) & 0xFFFFFFFF))
                for j in range(order):
                    acc = (acc ^ shift(u, j)) * np.uint32(0x01000193)
                acc = acc ^ (acc >> np.uint32(15))
                columns.append((acc % np.uint32(self.engram_slots)).astype(np.int64))
        idx = np.stack(columns, -1)
        tables = len(self.orders) * self.engram_heads
        rows = (np.arange(tables)[None, :] * self.engram_slots + idx).reshape(-1)
        fetched = self.c.cq(e["tables"], rows).reshape(t, tables, self.engram_sub)
        ok = np.stack([(np.arange(t) >= o - 1) for o in self.orders for _ in range(self.engram_heads)], -1).astype(np.float32)
        feats = self.act((fetched * ok[:, :, None]).reshape(t, -1), "engram")
        k = self.mm(feats, e["key"])
        v = self.mm(feats, e["value"])
        taps = self.dense(e["taps"])
        v = sum(taps[j] * shift(v, j * max(self.orders)) for j in range(4))
        return k, v

    def decoder_block(self, u, l, tokens, cross_kv, engrams):
        w = self.decoder[l]
        t = u.shape[0]
        x = u
        if l in self.sites:
            ek, ev = engrams[self.sites.index(l)]
            alpha = sigmoid((unit(x) * unit(ek)).sum(-1) / np.sqrt(self.dim))
            x = x + alpha[:, None] * ev
        skip = x
        h = self.act(zc(x, self.dense(w["norm_in"])), "attn_in")
        q = self.mm(h, w["q"])
        k = self.mm(h, w["k"])
        v = self.mm(h, w["v"])
        q = sum(self.dense(w["q_taps"])[j] * shift(q, j) for j in range(3))
        k = sum(self.dense(w["k_taps"])[j] * shift(k, j) for j in range(3))
        v = sum(self.dense(w["v_taps"])[j] * shift(v, j) for j in range(3))
        positions = np.arange(t)
        q = self.rotate(zc(q.reshape(t, self.heads_total, self.qk), self.dense(w["q_norm"])), positions)
        k = self.rotate(zc(k.reshape(t, self.kv_heads, self.qk), self.dense(w["k_norm"])), positions)
        v = v.reshape(t, self.kv_heads, self.vd)
        out = self.attend(self.act(q, "qkv"), self.act(k, "qkv"), self.act(v, "qkv"), positions[:, None] >= positions[None, :]).reshape(t, -1)
        out = out * sigmoid(self.mm(h, w["gate"]))
        out = self.mm(self.act(out, "attn_out"), w["out"])
        x = skip + sigmoid(self.dense(w["attn_gate"])[0]) * zc(out, self.dense(w["post_norm"]))

        c = self.cross[l]
        skip = x
        h = self.act(zc(x, self.dense(c["norm"])), "attn_in")
        q = zc(self.mm(h, c["q"]).reshape(t, self.heads_total, self.qk), self.dense(c["q_norm"]))
        keys, values = cross_kv
        out = self.attend(self.act(q, "qkv"), keys[l], values[l], np.ones((t, len(keys[l])), bool)).reshape(t, -1)
        out = out * sigmoid(self.mm(h, c["gate"]))
        out = self.mm(self.act(out, "attn_out"), c["out"])
        x = skip + sigmoid(self.dense(c["mix"])[0]) * zc(out, self.dense(c["post_norm"]))

        skip = x
        x = skip + self.mlp(zc(x, self.dense(w["pre_hada"])), w["mlp"])
        return x

    def logits(self, tokens, cross_kv):
        tokens = np.asarray(tokens)
        t = len(tokens)
        embedding = self.weight(self.embedding_id)
        stream = np.repeat((embedding[tokens] * np.sqrt(self.dim))[:, None, :], 4, 1).astype(np.float32)
        engrams = [self.engram_features(tokens, s) for s in range(len(self.sites))]
        for l in range(self.layers):
            u, hpost, hres = self.hyper_mix(stream, self.hyper, l, np.eye(4, dtype=np.float32)[l % 4])
            y = self.decoder_block(u, l, tokens, cross_kv, engrams) - u
            stream = (np.einsum("tij,tjc->tic", hres, stream) + hpost[:, :, None] * y[:, None, :]).astype(np.float32)
        x = zc(stream.mean(1), self.dense(self.final_norm))
        return self.mm(self.act(x, "logits"), self.embedding_id).astype(np.float32)


def greedy(model, tokenizer, cross_kv, limit=96):
    vocab = len(tokenizer.pieces)
    text_vocab = vocab - 7
    tokens = [BOS]
    first = model.logits(tokens, cross_kv)[-1]
    tag = text_vocab + int(first[text_vocab:].argmax())
    tokens.append(tag)
    generated = []
    for _ in range(limit):
        scores = model.logits(tokens, cross_kv)[-1][:text_vocab].copy()
        scores[[0, 2, 3]] = -1e30
        best = int(scores.argmax())
        if best == EOS:
            break
        generated.append(best)
        tokens.append(best)
    return generated


def detokenize(tokenizer, ids):
    out = bytearray()
    for i in ids:
        piece = tokenizer.pieces[i]
        if tokenizer.types[i] == 4:
            out.append(int(piece[3:5], 16))
        else:
            out.extend(piece.replace("▁", " ").encode("utf-8"))
    return out.decode("utf-8", "replace").strip()


def dump_clip(model, tokenizer, name, pcm, with_text):
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    pcm.astype(np.float32).tofile(target / "pcm.f32")
    stages = model.encode(pcm)
    frames = len(stages["encoder"])
    for key, value in stages.items():
        value.astype(np.float32).tofile(target / f"{key}.f32")
    mem = model.memory(stages["encoder"])
    mem.astype(np.float32).tofile(target / "memory.f32")
    cross_kv = model.cross_memory(mem)
    cross_kv[0][0].astype(np.float32).tofile(target / "cross_key0.f32")
    cross_kv[1][-1].astype(np.float32).tofile(target / "cross_value_last.f32")
    text_vocab = len(tokenizer.pieces) - 7
    first = model.logits([BOS], cross_kv)[-1]
    tag = text_vocab + int(first[text_vocab:].argmax())
    second = model.logits([BOS, tag], cross_kv)[-1]
    first.tofile(target / "logits_bos.f32")
    second.tofile(target / "logits_tag.f32")
    meta = {"frames": frames, "samples": int(len(pcm)), "tag": tag, "stages": ["mel", "stem"] + [f"layer{l}" for l in range(model.enc_layers)] + ["encoder"]}
    if with_text:
        meta["text"] = detokenize(tokenizer, greedy(model, tokenizer, cross_kv))
        print(name, "->", meta["text"])
    (target / "meta.json").write_text(json.dumps(meta))
    print(name, "frames", frames)


def main():
    names = sys.argv[1:] or list(CLIPS)
    with_text = "--text" in names
    names = [n for n in names if n != "--text"] or list(CLIPS)
    cact = Cact(CACT)
    model = Whistle(cact, quant=True)
    tokenizer = RefTokenizer(cact.blob(cact.n - 1))
    for name in names:
        file, limit = CLIPS[name]
        dump_clip(model, tokenizer, name, read_wav(AUDIO / file, limit), with_text)


if __name__ == "__main__":
    main()
