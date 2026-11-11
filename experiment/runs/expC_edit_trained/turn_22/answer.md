Everything is flat around the trained values under the real four-view pipeline too (scale 1.0 is the DFL optimum; the SC blip at 0.7 is one or two possessions out of 75 and DFL moves the other way), and the weight read shows no structural anomaly (no dead units, home/away input columns balanced, gate biases unremarkable). I'll confirm the working weights are bit-identical to the original before concluding.

```python
same = {k: bool(np.array_equal(W[k], BASE[k])) for k in BASE}
print("all tensors identical to BASE:", all(same.values()), "| differing:", [k for k, v in same.items() if not v])
print("passes left:", passes_left())
```