The own-frame H/A bias hypothesis is refuted: `b` is ≈ ±0.1–0.3 logits with inconsistent sign across chunks, so there is no constant slot bias to remove. What does exist is a frame-specific role-swap disagreement (mean |b| ≈ 0.7–0.8 logits in ambiguous frames, vs 0.25–0.4 for the y-mirror) — a non-equivariance that the view averaging already handles and that no bias-type edit can remove.

One last mechanistic hypothesis for the under-splitting: the recurrent states are too sticky (update gates `z` hold too long), so short possessions never register. If true, lowering the update-gate bias should add possessions and raise spellF1, and raising it should do the opposite. I'll test this as a two-sided sensitivity measurement in the simulator, separately for each GRU layer.

```python
def full_logits(Wd):
    out = {}
    for n in NAMES:
        X = views(D[n]["x"])
        out[n] = back_end(front_end(X, Wd), Wd).astype(np.float32)
    return out
def variant(edits):
    Wd = {k: BASE[k] for k in BASE}
    for k, fn in edits.items():
        Wd[k] = fn(BASE[k].copy())
    return Wd
LGB = full_logits(BASE)
summ("BASE (resim)", LG=LGB)
for layer in ("gru2", "gru1"):
    for dlt in (-0.5, +0.5):
        Wd = variant({layer + "f.biz": (lambda b, d=dlt: b + d), layer + "b.biz": (lambda b, d=dlt: b + d)})
        summ("%s z-bias %+.1f" % (layer, dlt), LG=full_logits(Wd))
```