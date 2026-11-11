**Changes for pass 8**

One addition only, so its effect can be read cleanly: an "exited at the line" memory for dead balls.

- **Why:** in the third excerpt the ball reached the goal line and was then lost, and the straight-line interpolation drifted back into the pitch during the gap, so my line-proximity units switched off and the dead ball was missed.
- **Forward:** `gru1f[4]` records whether the ball was at or beyond a line at its last sighting, and holds that while the ball is unseen.
- **Backward:** `gru1b[4]` does the same for the next sighting, which catches a ball that reappears at the touchline for a throw-in.
- **Head:** each adds dead evidence (up to about +1.4) only on unseen frames.

The risk is a winger's ball lost near the touchline in live play being read as dead; the weight is kept moderate for that reason.

```edits
# gru1f[4] / gru1b[4]: "ball at/beyond a line at last (next) sighting", held while unseen
gru1f.Win[4, 5] = 2
gru1f.Win[4, 6] = 2
gru1f.bin[4] = -2
gru1f.biz[4] = -3
gru1f.Wiz[4, 17] = 3
gru1b.Win[4, 5] = 2
gru1b.Win[4, 6] = 2
gru1b.bin[4] = -2
gru1b.biz[4] = -3
gru1b.Wiz[4, 17] = 3

# copies through layer 2
gru2f.Win[4, 4] = 2
gru2f.biz[4] = -20
gru2b.Win[4, 132] = 2
gru2b.biz[4] = -20

# head: active only when unseen and memory says "at line"
head1.Wf[9, 4] = 2
head1.Wh[9, 17] = 1
head1.b[9] = -3
head1.Wb[10, 4] = 2
head1.Wh[10, 17] = 1
head1.b[10] = -3
head2.W[2, 9] = 0.8
head2.W[2, 10] = 0.8
```