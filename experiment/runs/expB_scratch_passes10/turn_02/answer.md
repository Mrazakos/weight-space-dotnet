**Changes for pass 3**

- **Control evidence less strict on relative velocity:** the weight on `relv` goes from −5 to −3, so noisy tracking still registers control. It still switches off when the ball is unseen. This is aimed at the possession count, which is still short, most of all on DFL.
- **Cap on the "lazy players" dead evidence:** a conv/ReLU unit subtracts the excess above a threshold, so very slow players alone can no longer push D far up. This targets the false dead ball in the second excerpt, where a player held a near-stationary ball in live play.
- **New "ball at a player's feet" units:** one per team, firing when the nearest player is within about 0.8 m. They count mildly against dead, to separate a held ball from a free kick or goal kick with the taker standing off.

I expect the second excerpt to improve only partly; it still looks ambiguous against a real set piece.

```edits
# relax relv requirement
inp.W[0, 18] = -3
inp.W[1, 35] = -3

# cap lazy contribution: h12 = relu(L) where L = a4 - 6
conv.W[12, 4, 4] = 1
conv.b[12] = -6
head1.Wh[1, 12] = -1.5

# held-ball units (a13 home, a14 away)
inp.b[13] = 2
inp.W[13, 13] = -25
inp.b[14] = 2
inp.W[14, 30] = -25
head1.Wh[1, 13] = -0.5
head1.Wh[1, 14] = -0.5
```