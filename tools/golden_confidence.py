import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import Cact, ROOT, a8, sigmoid, unit

cact = Cact(ROOT / "ref/hf/needle3.cact")
dump = ROOT / "ref/golden/case0"
ids = np.fromfile(dump / "ids.bin", np.int32)
t = len(ids)
x0 = np.fromfile(dump / "x0.bin", np.float32).reshape(t, 768)
cells = [x0] + [np.fromfile(dump / f"stream{l:02d}.bin", np.float32).reshape(t, 4, 768).mean(1) for l in range(20)]
cells = np.stack(cells, 1)

probes = cact.cq(574).reshape(21, 4, 768)
gain = cact.dense(575)
query = cact.cq(576)
row_bias = cact.dense(577)
proj = cact.cq(578)
bias = cact.dense(579)

scores = np.einsum("tld,lkd->lkt", cells, probes) / np.sqrt(768)
weights = np.exp(scores - scores.max(-1, keepdims=True))
weights /= weights.sum(-1, keepdims=True)
r = np.einsum("lkt,tld->lkd", weights, cells)
r = unit(r) * gain[:, :, None]
u = np.einsum("lkd,qd->qlk", r, query) / np.sqrt(768) + row_bias
w = np.exp(u.reshape(4, -1) - u.reshape(4, -1).max(-1, keepdims=True))
w /= w.sum(-1, keepdims=True)
pooled = np.einsum("qm,md->qd", w, r.reshape(-1, 768)).reshape(-1)
logit = float(proj[0] @ pooled + bias[0])
print(logit, sigmoid(logit))
(dump / "confidence.txt").write_text(f"{logit}\n")
pooled.astype(np.float32).tofile(dump / "pooled.bin")
