Data and simulator rebuilt (5 passes left). Now the label-free measurement: the own-frame H/A bias `b = (margin in original views − margin in swapped views, both mapped to real teams) / 2`, overall and specifically in the ambiguous regime, plus the same check for D-vs-live and the y-mirror (which should show no bias by construction of the measurement).

```python
def bias_report():
    rows = []
    for n in NAMES:
        lg = logits_from_m(n, BASE)[:, SL].astype(np.float64)
        own = lg[:, :, 0] - lg[:, :, 1]                 # own-frame H-A margin, per view
        real = own.copy(); real[2:] = -real[2:]         # real-team H-A margin
        r_o = real[:2].mean(0); r_s = real[2:].mean(0)
        b = (r_o - r_s) / 2                             # own-frame bias toward the "H" slot
        mavg = (r_o + r_s) / 2
        dl = lg[:, :, 2] - np.maximum(lg[:, :, 0], lg[:, :, 1])   # D vs best live, per view
        livepred = dl.mean(0) < 0
        amb = livepred & (np.abs(mavg) < 2)
        strong = livepred & (np.abs(mavg) >= 4)
        seen = D[n]["x"][SL, 0] > 0.5
        rows.append((n, b[livepred].mean(), np.median(b[livepred]), b[amb].mean(), np.median(b[amb]), amb.mean(), b[strong].mean(),
                     b[amb & seen].mean(), b[amb & ~seen].mean(), np.abs(b[amb]).mean(),
                     (dl[:2].mean(0) - dl[2:].mean(0)).mean(), (real[0] - real[1])[livepred].mean(), np.abs(real[0] - real[1])[amb].mean()))
    print("chunk | b mean(live) med(live) | b mean(amb) med(amb) fracAmb | b(strong) | b amb&seen  amb&unseen | mean|b| amb || D-margin orig-swap | mirror diff mean, mean|.| amb")
    for r in rows:
        print("%-5s | %+.3f %+.3f | %+.3f %+.3f %.3f | %+.3f | %+.3f %+.3f | %.3f || %+.3f | %+.3f %.3f" % r)
bias_report()
print("head2.b", BASE["head2.b"])
```