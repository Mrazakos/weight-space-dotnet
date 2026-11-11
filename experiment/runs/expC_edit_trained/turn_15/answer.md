Timing is unbiased everywhere (median offsets ≈ 0, zero-crossing of the median margin at the true boundary), so there is no systematic lateness to correct — DFL boundaries are just noisy. The output also says the process was restarted; let me check what survived before planning further.

```python
g = globals()
print("passes left", passes_left())
print("have D:", "D" in g, "| have LG0:", "LG0" in g, "| have sim fns:", "full_logits" in g, "run_gru" in g)
print("weights unchanged:", all(np.array_equal(W[k], BASE[k]) for k in BASE))
```