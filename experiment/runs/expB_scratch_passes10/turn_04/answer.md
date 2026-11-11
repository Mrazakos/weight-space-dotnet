**Changes for pass 5**

The third excerpt (DFL) shows a ball that is stationary but jitters by about a metre, so per-frame speed reads as fast and my stationary-ball unit never fires there. I'm adding a noise-robust stationarity detector on a longer timescale:

- **Smoothed ball position:** `gru1f[2,3]` and `gru1b[2,3]` hold running averages of the seen ball x and y (time constant about 0.8 s), frozen while the ball is unseen. The forward one summarises where the ball has been, the backward one where it will be.
- **Displacement:** `gru2f[2,3]` take the clipped difference between the forward and backward averages, which measures ball movement over roughly 2 s. Across a gap it measures whether the ball reappears where it vanished.
- **Head:** four units form the absolute values. A small displacement adds up to +1.2 to the D logit; a large one subtracts up to about 3.6.

Nothing else changes this pass, so the effect can be read cleanly.

```edits
# linear copies of seen ball position
inp.b[18] = 6
inp.W[18, 1] = 1
inp.b[19] = 6
inp.W[19, 2] = 1

# EMA of ball x / y, forward and backward, frozen when unseen (h17 = 3 when unseen)
gru1f.Win[2, 18] = 0.5
gru1f.bin[2] = -3
gru1f.biz[2] = 3
gru1f.Wiz[2, 17] = 5
gru1f.Win[3, 19] = 0.5
gru1f.bin[3] = -3
gru1f.biz[3] = 3
gru1f.Wiz[3, 17] = 5
gru1b.Win[2, 18] = 0.5
gru1b.bin[2] = -3
gru1b.biz[2] = 3
gru1b.Wiz[2, 17] = 5
gru1b.Win[3, 19] = 0.5
gru1b.bin[3] = -3
gru1b.biz[3] = 3
gru1b.Wiz[3, 17] = 5

# displacement (forward EMA - backward EMA), clipped
gru2f.Win[2, 2] = 40
gru2f.Win[2, 130] = -40
gru2f.biz[2] = -20
gru2f.Win[3, 3] = 26
gru2f.Win[3, 131] = -26
gru2f.biz[3] = -20

# |displacement| in the head -> against dead; stationary -> for dead
head1.Wf[4, 2] = 8
head1.Wf[5, 2] = -8
head1.Wf[6, 3] = 8
head1.Wf[7, 3] = -8
head2.W[2, 4:8] = -0.3
head2.b[2] = -4.8
```