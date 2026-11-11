**Changes for pass 4**

- **Dead-ball recall:** D precision is 0.91 against recall 0.83, so the dead score gets +0.3 across the board.
- **No players tracked (SC):** these stretches were dead in two of the three excerpts that showed them, so a new unit adds mild dead evidence when both teams have no tracked players.
- **Wider touchline zone:** the missed throw-in in the third excerpt had the ball about 3.5 m inside the line, outside my 1.5 m unit. A second, weaker unit now covers out to 5 m.
- **Possession changes inside ball gaps:** when the ball is unseen and the forward latch (last controller) disagrees with the backward latch (next controller), the forward bias is now cancelled. The nearest-player signal then decides where the switch falls, instead of it always landing at the end of the gap. This is aimed at the short possession count on DFL.

```edits
# D recall
head1.b[1] += 0.3

# a15: no players tracked at all
inp.b[15] = 2
inp.W[15, 20] = -2
inp.W[15, 37] = -2
head1.Wh[1, 15] = 0.4

# a16: wide touchline zone
inp.b[16] = 2
inp.W[16, 12] = -4
head1.Wh[1, 16] = 0.4

# a17: unseen indicator (3 when unseen)
inp.b[17] = 3
inp.W[17, 0] = -3

# m2: forward says H, backward says A, ball unseen -> cancel forward bias
head1.Wf[2, 0] = 3
head1.Wb[2, 0] = -3
head1.Wh[2, 17] = 1
head1.b[2] = -8
head2.W[0, 2] = -0.15
head2.W[1, 2] = 0.15
# m3: mirror case
head1.Wf[3, 0] = -3
head1.Wb[3, 0] = 3
head1.Wh[3, 17] = 1
head1.b[3] = -8
head2.W[0, 3] = 0.15
head2.W[1, 3] = -0.15
```