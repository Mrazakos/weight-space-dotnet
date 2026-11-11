I'm keeping the network exactly as it stands after pass 10, which scored best on the practice matches (spellF1 0.5591, macroF1 0.8388). I'm not adding any untested change in the final reply.

The 0.5591 is uneven across sources: MET 0.76, SC 0.62, DFL 0.37. DFL stayed weak throughout: possession changes and dead balls inside its very long ball gaps are mostly invisible to this design.

The final design:

- **Possession:** forward and backward latches in GRU layer 1 hold the last and the next team with controlled possession. Control means the ball is seen, a player is close and moving with it, and no opponent is closer. A nearest-player signal breaks ties, and the forward bias is cancelled inside ball gaps when the two latches disagree.
- **Dead ball:** evidence is summed in the head from:
  - slow players (capped);
  - a seen, stationary ball, from per-frame speed and from the difference of forward and backward smoothed positions;
  - ball near or beyond a line, plus a memory of "last or next seen at a line" that applies during gaps;
  - long gaps;
  - nobody near the ball, or no players tracked.

  A fast ball and a ball at a player's feet count against dead.
- **Decoder scale:** a confident possession is worth about 1.3 nats per frame over the other team, so the decoder needs roughly a second of evidence to switch team.

```edits
```