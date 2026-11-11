**Changes for pass 6**

Pass 5 broke dead-ball recall (0.91 → 0.48). The cause is the new displacement term during ball gaps: the forward average holds the position before the gap and the backward one the position after it, and across a dead ball these usually differ a lot. So unseen dead stretches were being read as "ball moving", which is the reverse of what I intended.

The fix is to make the displacement term count only on frames where the ball is seen:

- The four absolute-value head units are switched off when the ball is unseen.
- The matching "stationary" bonus is also removed on unseen frames, so those frames behave as in pass 4.
- I'm also shrinking the term slightly (bonus +1.0, slope 0.25) until I see that it helps.

```edits
# displacement units only active when seen (h17 = 3 when unseen)
head1.Wh[4, 17] = -6
head1.Wh[5, 17] = -6
head1.Wh[6, 17] = -6
head1.Wh[7, 17] = -6
head2.W[2, 4:8] = -0.25
# stationary bonus +1.0 only when seen
head2.b[2] = -5.0
head1.Wh[1, 17] = -0.3333
```