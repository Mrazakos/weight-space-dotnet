The restart wiped the collected data (weights are untouched, 11 passes left). Rather than re-collect everything, I'll focus the remaining budget on one hypothesis that follows directly from how the outputs are used and can be measured **without labels**:

The pipeline averages probabilities over role-swapped views. If the network has a bias `b` toward one output slot (H or A) in its own frame, then in the original view the H−A margin is `m+b` and in the swapped view (mapped back) `m−b`. Averaging σ(m+b) and σ(m−b) attenuates weak evidence `m` by σ′(b)/σ′(0) — so an own-frame H/A bias directly weakens the averaged evidence in exactly the ambiguous phases where the errors are. `b` is measurable as half the difference between original-view and swapped-view margins. I'll re-collect 6 chunks (x, y only), rebuild the simulator, and measure it.

```python
import collections
def erf_np(x):
    s = np.sign(x); ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    yv = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return s * yv
def gelu(v): return v * 0.5 * (1.0 + erf_np(v / np.sqrt(2.0)))
def lsm(lg):
    z = lg - lg.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))
def sig(v): return 1.0 / (1.0 + np.exp(-v))
def run_gru(X, Wd, pre, reverse=False):
    B, T, I = X.shape
    Gr = X @ Wd[pre + ".Wir"].T + (Wd[pre + ".bir"] + Wd[pre + ".bhr"])
    Gz = X @ Wd[pre + ".Wiz"].T + (Wd[pre + ".biz"] + Wd[pre + ".bhz"])
    Gn = X @ Wd[pre + ".Win"].T + Wd[pre + ".bin"]
    bhn = Wd[pre + ".bhn"]
    WhrT, WhzT, WhnT = Wd[pre + ".Whr"].T.copy(), Wd[pre + ".Whz"].T.copy(), Wd[pre + ".Whn"].T.copy()
    s = np.zeros((B, 128), np.float32); out = np.empty((B, T, 128), np.float32)
    for t in (range(T - 1, -1, -1) if reverse else range(T)):
        r = sig(Gr[:, t] + s @ WhrT); z = sig(Gz[:, t] + s @ WhzT)
        n = np.tanh(Gn[:, t] + r * (s @ WhnT + bhn))
        s = (1 - z) * n + z * s
        out[:, t] = s
    return out
def front_end(X, Wd):
    a = gelu(X @ Wd["inp.W"].T + Wd["inp.b"]).astype(np.float32)
    B, T, C = a.shape
    ap = np.concatenate([np.zeros((B, 4, C), np.float32), a, np.zeros((B, 4, C), np.float32)], 1)
    c = np.zeros((B, T, 128), np.float32) + Wd["conv.b"]
    for k in range(9): c += ap[:, k:k + T] @ Wd["conv.W"][:, :, k].T
    return a + np.maximum(c, 0)
def back_end(h, Wd, want_m=False):
    g1 = np.concatenate([run_gru(h, Wd, "gru1f"), run_gru(h, Wd, "gru1b", True)], -1)
    qf = run_gru(g1, Wd, "gru2f"); qb = run_gru(g1, Wd, "gru2b", True)
    del g1
    m = gelu(qf @ Wd["head1.Wf"].T + qb @ Wd["head1.Wb"].T + h @ Wd["head1.Wh"].T + Wd["head1.b"]).astype(np.float32)
    return m if want_m else m @ Wd["head2.W"].T + Wd["head2.b"]
HB, AB = 13, 30
def views(x):
    def mir(v):
        v = v.copy()
        for i in (2, 8, HB + 9, AB + 9): v[:, i] = -v[:, i]
        return v
    def swp(v):
        o = v.copy()
        o[:, HB:HB + 17] = v[:, AB:AB + 17]; o[:, AB:AB + 17] = v[:, HB:HB + 17]
        for i in (1, 2, 7, 8, HB + 8, HB + 9, AB + 8, AB + 9, 47): o[:, i] = -o[:, i]
        return o
    return np.stack([x, mir(x), swp(x), mir(swp(x))], 0)
SL = slice(1000, 23000)
def pipe_views(lg, vs=(0, 1, 2, 3)):
    p = np.exp(np.stack([lsm(lg[v].astype(np.float64)) for v in vs], 0))
    for i, v in enumerate(vs):
        if v >= 2: p[i] = p[i][:, [1, 0, 2]]
    return np.log(np.clip(p.mean(0), 1e-6, 1.0))

D = {}
for name, half, start in [("dfl0", 0, 20000), ("dfl2", 2, 30000), ("dfl1", 1, 44000), ("dfl3", 3, 4000), ("sc8", 8, 20000), ("sc9", 9, 40000)]:
    o = forward(half, start=start, n=24000)
    D[name] = {"x": o["x"].copy(), "y": o["y"].copy()}
    del o
NAMES = list(D)
# m (penultimate) for all views, base weights: lets any head2 edit be evaluated for free
M0 = {}
for n in NAMES:
    X = views(D[n]["x"])
    M0[n] = back_end(front_end(X, BASE), BASE, want_m=True).astype(np.float16)
    del X
def logits_from_m(n, Wd=None):
    Wd = Wd or W
    return M0[n].astype(np.float32) @ Wd["head2.W"].T + Wd["head2.b"]
def summ(tag, Wd=None, vs=(0, 1, 2, 3), LG=None):
    per = collections.defaultdict(list)
    for n in NAMES:
        lg = LG[n] if LG is not None else logits_from_m(n, Wd)
        y = D[n]["y"][SL]; pr = decode(pipe_views(lg[:, SL], vs)); mt = metrics(y, pr)
        per[n[:2]].append((mt["spell_f1"], mt["macro_f1"], mt["n_pred_spells"], mt["n_gt_spells"], (pr == y).mean()))
    s = "%-26s" % tag
    for src in per:
        a = np.array(per[src]); s += " | %s sF1 %.4f mF1 %.4f n %3d/%3d acc %.4f" % (src, a[:, 0].mean(), a[:, 1].mean(), a[:, 2].sum(), a[:, 3].sum(), a[:, 4].mean())
    print(s)
summ("BASE 4 views", BASE)
print("passes left", passes_left())
```