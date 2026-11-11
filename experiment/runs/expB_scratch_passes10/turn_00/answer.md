## Design (pass 1)

This first pass is a small, symmetric hand-built circuit; almost all weights stay 0. Thresholds for player speed and the "slow" share are my guesses and will need calibrating from the feedback.

**Input layer `a` (9 units):**
- `a0` / `a1`: controlled-possession evidence for HOME / AWAY. Each fires when the team's nearest player is close to the ball, moves with it (small `relv`, so it is automatically off when the ball is unseen), and no opponent is within 2–5 m.
- `a2`: linear copy of `d1_diff` (offset by 6 so GELU is linear).
- `a3`: ball seen and stationary (below about 0.8 m/s).
- `a4`: "lazy players", a linear copy of 2·(slow_H + slow_A) − 2·(mean_speed_H + mean_speed_A).
- `a5` / `a6`: ball near or beyond a touchline / goal line.
- `a7`: ball moving fast (above 2.5 m/s), which counts against dead.
- `a8`: long ball gap (since + until).

The conv layer is unused, so `h = a`.

**GRU layer 1:**
- `gru1f[0]` and `gru1b[0]` are possession latches. The candidate is tanh(3·(a0 − a1)); the keep gate sits near 1 and opens only when control evidence is present. The forward one holds the last controller, the backward one the next controller.
- `gru1f[1]` is an instantaneous clipped nearest-team signal, tanh(−5·d1_diff).

**GRU layer 2:** pass-through copies of those three states (keep gate forced to 0).

**Head:**
- `m0` = 6 + 0.25·forward latch + 0.15·backward latch + 0.1·nearest-team. The H logit is +(m0 − 6) and the A logit is −(m0 − 6), so a confident possession gives about 1 nat per frame and a flip needs roughly 1.5 s of evidence against the 16 penalty.
- `m1` = 6 + dead score. The dead score is −1 + 1.5·(lazy + 1.25) + 0.8·stationary + touchline + goal line − 0.7·fast ball + 0.3·long gap. The D logit is m1 − 6.

The circuit treats HOME and AWAY identically and uses only |x|, |y| quantities, so the four views agree.

```edits
reset
# ---- input layer ----
# a0: home control evidence
inp.b[0] = 3
inp.W[0, 13] = -7
inp.W[0, 18] = -5
inp.W[0, 32] = -1.5
inp.W[0, 33] = -1.5
# a1: away control evidence
inp.b[1] = 3
inp.W[1, 30] = -7
inp.W[1, 35] = -5
inp.W[1, 15] = -1.5
inp.W[1, 16] = -1.5
# a2: linear d1_diff (+6)
inp.b[2] = 6
inp.W[2, 47] = 1
# a3: seen & stationary ball
inp.W[3, 0] = 2.5
inp.W[3, 3] = -30
# a4: lazy players (linear, +6)
inp.b[4] = 6
inp.W[4, 28] = 2
inp.W[4, 45] = 2
inp.W[4, 27] = -2
inp.W[4, 44] = -2
# a5: near/beyond touchline
inp.b[5] = 1.5
inp.W[5, 12] = -10
# a6: near/beyond goal line
inp.b[6] = 1.5
inp.W[6, 11] = -10
# a7: fast ball
inp.b[7] = -1.5
inp.W[7, 3] = 6
# a8: long gap
inp.b[8] = -0.9
inp.W[8, 5] = 3
inp.W[8, 6] = 3

# ---- GRU1 forward: unit 0 latch, unit 1 nearest-team ----
gru1f.Win[0, 0] = 3
gru1f.Win[0, 1] = -3
gru1f.Wiz[0, 0] = -5.5
gru1f.Wiz[0, 1] = -5.5
gru1f.biz[0] = 6
gru1f.Win[1, 2] = -5
gru1f.bin[1] = 30
gru1f.biz[1] = -20
# ---- GRU1 backward: unit 0 latch ----
gru1b.Win[0, 0] = 3
gru1b.Win[0, 1] = -3
gru1b.Wiz[0, 0] = -5.5
gru1b.Wiz[0, 1] = -5.5
gru1b.biz[0] = 6

# ---- GRU2: pass-through ----
gru2f.Win[0, 0] = 2
gru2f.biz[0] = -20
gru2f.Win[1, 1] = 2
gru2f.biz[1] = -20
gru2b.Win[0, 128] = 2
gru2b.biz[0] = -20

# ---- head ----
head1.Wf[0, 0] = 0.25
head1.Wf[0, 1] = 0.1
head1.Wb[0, 0] = 0.15
head1.b[0] = 6
head1.Wh[1, 4] = 1.5
head1.Wh[1, 3] = 0.8
head1.Wh[1, 5] = 1
head1.Wh[1, 6] = 1
head1.Wh[1, 7] = -0.7
head1.Wh[1, 8] = 0.3
head1.b[1] = -2.125
head2.W[0, 0] = 1
head2.W[1, 0] = -1
head2.W[2, 1] = 1
head2.b[0] = -6
head2.b[1] = 6
head2.b[2] = -6
```