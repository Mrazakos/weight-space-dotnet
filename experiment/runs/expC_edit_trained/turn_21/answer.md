The gates behave exactly as the mechanism predicts (lower update-gate bias → more possessions, 144→154 on DFL; higher → fewer), but the score moves within noise and in opposite directions on DFL (+0.007) and SC (−0.015). So the stickiness is not mis-set; the trained value is at the trade-off. Two last free checks: (1) logit scale / D offset again, now under the real four-view probability averaging (where over-sharp logits turn the average into a vote); (2) a read of the weights for structural anomalies (dead or runaway units, home/away asymmetry at the input, unused features).

```python
print("four-view sensitivity to logit scale and D offset (free, from stored m)")
def variant_head(scale=1.0, dD=0.0):
    Wd = {"head2.W": BASE["head2.W"] * scale, "head2.b": BASE["head2.b"] * scale + np.array([0, 0, dD], np.float32)}
    return Wd
for sc_ in (0.5, 0.7, 1.0, 1.4, 2.0):
    summ("scale %.1f" % sc_, variant_head(scale=sc_))
for dD in (-1.0, -0.5, 0.5, 1.0):
    summ("D offset %+.1f" % dD, variant_head(dD=dD))

print("\nweight read")
Wi = BASE["inp.W"]
cn = np.linalg.norm(Wi, axis=0)
names_k = ["d1", "d2", "n2", "n5", "n10", "relv", "near_speed", "n_seen", "cx", "cy", "sx", "sy", "front", "back", "mean_speed", "slow", "top_speed"]
print("input column norms, global 0..12:", np.round(cn[:13], 2), " d1_diff:", round(float(cn[47]), 2))
print("block feature: HOME norm / AWAY norm")
print("  " + "  ".join("%s %.2f/%.2f" % (names_k[k], cn[13 + k], cn[30 + k]) for k in range(17)))
m_all = np.concatenate([M0[n][:, SL].astype(np.float32).reshape(-1, 128) for n in NAMES], 0)
mm = m_all.mean(0); mx = m_all.max(0); sd = m_all.std(0)
print("m units: n with std<0.02 (dead): %d ; max activation overall %.2f ; units with max>6: %d" % ((sd < 0.02).sum(), mx.max(), (mx > 6).sum()))
w2 = BASE["head2.W"]
contrib = np.abs(w2).sum(0) * sd
order = np.argsort(-contrib)[:8]
print("top m units by |w|*std:", [(int(u), round(float(contrib[u]), 2), np.round(w2[:, u], 2).tolist()) for u in order])
for pre in ("gru1f", "gru1b", "gru2f", "gru2b"):
    bz = BASE[pre + ".biz"] + BASE[pre + ".bhz"]
    print(pre, "update-gate bias: mean %+.2f min %+.2f max %+.2f ; ||Whz|| %.1f ||Wiz|| %.1f ||Whn|| %.1f" % (
        bz.mean(), bz.min(), bz.max(), np.linalg.norm(BASE[pre + ".Whz"]), np.linalg.norm(BASE[pre + ".Wiz"]), np.linalg.norm(BASE[pre + ".Whn"])))
print("any non-finite weights:", any(not np.isfinite(BASE[k]).all() for k in BASE))
```