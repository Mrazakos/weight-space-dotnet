**Changes for pass 7**

- **Noise-robust control evidence:** on DFL the relative-velocity feature is noisy even for a still ball, so my control units rarely fire there. I'm adding a second evidence path per team that ignores `relv`: ball seen, own player within about 1 m, and no opponent close. It feeds the same latch difference units at reduced weight.
- **Slightly larger H/A margins:** the network still produces fewer possessions than the truth (1449 vs 1630), so the head weights go up by roughly a quarter. I'll revert this if it over-fragments.
- **"Nobody near the ball" as dead evidence:** in the third excerpt the ball was unseen with the nearest players 17–27 m from its interpolated position. A new unit fires when both teams' nearest distances are large; the no-players-tracked unit is re-weighted so that case keeps its previous net effect.

```edits
# a20 / a21: proximity-only control evidence (seen, within ~1 m, opponent not close)
inp.b[20] = -0.5
inp.W[20, 0] = 2.5
inp.W[20, 13] = -15
inp.W[20, 32] = -1.0
inp.W[20, 33] = -1.5
inp.b[21] = -0.5
inp.W[21, 0] = 2.5
inp.W[21, 30] = -15
inp.W[21, 15] = -1.0
inp.W[21, 16] = -1.5
conv.W[9, 20, 4] = 0.6
conv.W[9, 21, 4] = -0.6
conv.W[10, 21, 4] = 0.6
conv.W[10, 20, 4] = -0.6

# stronger possession margins
head1.Wf[0, 0] = 0.45
head1.Wb[0, 0] = 0.25
head1.Wf[0, 1] = 0.2
head2.W[0, 2] = -0.2
head2.W[1, 2] = 0.2
head2.W[0, 3] = 0.2
head2.W[1, 3] = -0.2

# a22: nobody near the ball -> dead evidence
inp.b[22] = -3
inp.W[22, 13] = 2
inp.W[22, 30] = 2
head1.Wh[1, 22] = 0.25
head1.Wh[1, 15] = -1.2
```