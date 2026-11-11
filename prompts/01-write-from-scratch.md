Your task is to write, from scratch, the weights of a neural network.
There is no training: no gradient descent, no data, no code to run. You have no tools.
You decide what every part of the network should compute, and you write the numbers.
This is a single attempt. You get no feedback.

THE PROBLEM
For every frame of a football match (25 per second) say who has the ball:
H (home), A (away) or D (dead ball). A loose ball stays with the last team that had it.
Possession changes only when the other team gains control for about 2 seconds.
The tracking is automatic: the ball is not seen on about 40% of frames.

THE INPUT
48 numbers per frame. For example:
  x[0]  = 1 if the ball is seen          x[5]  = seconds since it was last seen / 10
  x[13] = distance from the ball to the nearest HOME player / 10 m
  x[30] = the same for the nearest AWAY player
  ... the full table, and the code that computes it, follow ...

THE OUTPUT
Three numbers per frame (H, A, D). A decoder then picks the label sequence and charges
16 for every H <-> A switch and 24 for every switch between live play and dead ball.

THE NETWORK
  a      = GELU(inp.W x + inp.b)                       128 units
  h      = a + relu(conv over 9 frames of a)
  state  = two layers of bidirectional GRU over h       (gates: reset, update, new)
  logits = head2.W GELU(head1 [state, h])               3 outputs
698,115 weights. EVERY WEIGHT STARTS AT 0. You set the ones you need.

HOW TO WRITE THE WEIGHTS
First describe your design briefly. Then end your reply with one block:

```edits
inp.W[3, 47] = -2.5              # one weight: into unit 3, from input 47
conv.W[10, 3, :] = 0.111         # a number fills a range: a 9-frame average
gru1f.Wiz[2, 20:22] = [-24, -24] # a list fills a range
gru1f.bhz[0:8] += 2              # in place
```
