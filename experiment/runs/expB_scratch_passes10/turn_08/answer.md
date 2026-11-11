**Change for pass 9**

A single global experiment: all output logits are scaled by 1.3 (both `head2.W` and `head2.b`), leaving the relative evidence unchanged.

The reason is that the network still produces fewer possessions than the truth (1514 vs 1630), and its per-frame probabilities are soft compared with the decoder's switching penalties of 16 and 24. Short dead balls and short first possessions after a restart may therefore not accumulate enough evidence to be entered. Sharper logits make the decoder more willing to switch.

If this fragments possessions instead, I will go back to the pass 8 network.

```edits
head2.W[0:3, 0:128] *= 1.3
head2.b[0:3] *= 1.3
```