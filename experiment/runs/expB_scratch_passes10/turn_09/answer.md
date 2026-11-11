**Changes for pass 10**

- **Back to the pass 8 network:** the 1.3× logit scaling lowered the overall score (0.5530 vs 0.5587; MET up, SC down), so I'm restoring pass 8.
- **Looser trigger for the "exited at the line" memory:** in the third excerpt the ball was last seen about 0.5 m inside the goal line before vanishing, which was just short of the memory's threshold. It now triggers when the last (or next) sighting is within about 1 m of a line.
- **Slightly stronger memory weight in the head:** 0.8 → 1.0.

```edits
restore 8
gru1f.Win[4, 5] = 3
gru1f.Win[4, 6] = 3
gru1f.bin[4] = -1.5
gru1b.Win[4, 5] = 3
gru1b.Win[4, 6] = 3
gru1b.bin[4] = -1.5
head2.W[2, 9] = 1.0
head2.W[2, 10] = 1.0
```