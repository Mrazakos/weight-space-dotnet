```python
print("passes left", passes_left())
try:
    print("D present with", len(D), "chunks")
except NameError:
    print("D MISSING")
try:
    print("LG0 present", len(LG0))
except NameError:
    print("LG0 MISSING")
try:
    print("full_logits present", full_logits is not None)
except NameError:
    print("full_logits MISSING")
print("weights unchanged:", all(np.array_equal(W[k], BASE[k]) for k in BASE))
```