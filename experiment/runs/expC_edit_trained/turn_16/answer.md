```python
print("passes left", passes_left())
for nm in ["D", "LG0", "full_logits", "run_gru"]:
    try:
        eval(nm); print(nm, "present")
    except NameError:
        print(nm, "MISSING")
print("weights unchanged:", all(np.array_equal(W[k], BASE[k]) for k in BASE))
```