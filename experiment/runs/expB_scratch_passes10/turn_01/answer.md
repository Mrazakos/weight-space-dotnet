**Changes for pass 2**

- **Possession latch in contested play:** the opponent-proximity penalty was killing control evidence whenever a defender was close (first excerpt), so I'm halving it. The latch now reads the *difference* of the two control signals through two conv/ReLU units (`h9 = relu(a0 − a1)`, `h10 = relu(a1 − a0)`). When both teams are near the ball the state is held instead of being pulled to zero.
- **Stronger H/A margins:** the network produced too few possessions (1239 vs 1630) and P(H)/P(A) were close, so the head weights on the latches and nearest-team signal go up.
- **False dead ball when no players are tracked:** in the second excerpt `d1` was 4.0 and `slow` was 0 for both teams, which my "lazy" unit read as everyone standing still. I'm adding `n_seen` to that unit so missing tracks read as neutral.

```edits
# softer opponent penalties in control evidence
inp.W[0, 32] = -0.7
inp.W[0, 33] = -0.9
inp.W[1, 15] = -0.7
inp.W[1, 16] = -0.9

# lazy unit: neutral when players are not tracked
inp.W[4, 20] = 0.625
inp.W[4, 37] = 0.625
inp.b[4] = 4.75

# conv units 9, 10: rectified control difference (current frame only)
conv.W[9, 0, 4] = 1
conv.W[9, 1, 4] = -1
conv.W[10, 1, 4] = 1
conv.W[10, 0, 4] = -1

# latches read the difference units
gru1f.Win[0, 0] = 0
gru1f.Win[0, 1] = 0
gru1f.Wiz[0, 0] = 0
gru1f.Wiz[0, 1] = 0
gru1f.Win[0, 9] = 3
gru1f.Win[0, 10] = -3
gru1f.Wiz[0, 9] = -5.5
gru1f.Wiz[0, 10] = -5.5
gru1b.Win[0, 0] = 0
gru1b.Win[0, 1] = 0
gru1b.Wiz[0, 0] = 0
gru1b.Wiz[0, 1] = 0
gru1b.Win[0, 9] = 3
gru1b.Win[0, 10] = -3
gru1b.Wiz[0, 9] = -5.5
gru1b.Wiz[0, 10] = -5.5

# stronger possession margins
head1.Wf[0, 0] = 0.35
head1.Wb[0, 0] = 0.2
head1.Wf[0, 1] = 0.15
```