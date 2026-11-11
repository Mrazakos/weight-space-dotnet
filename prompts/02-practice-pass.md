PRACTICE PASS 5 of 10.

Your edits: 32 statements read, 32 applied, 0 skipped.
The network now has 134 nonzero weights.

Scores on the practice matches:
  source   possessions found / true    spellF1
  DFL              425 / 596            0.3115
  SC               699 / 765            0.5111
  MET              233 / 269            0.6223

Per class:   H precision 0.610 recall 0.839    A precision 0.624 recall 0.862
             D precision 0.930 recall 0.478

History:  pass 1: 0.4797   pass 2: 0.4950   pass 3: 0.5482   pass 4: 0.5443   pass 5: 0.4631

Excerpt: missed dead ball (truth is D, the network says live play). Source MET,
a mistake lasting 131.5 s; the mistake starts at the row marked >. One row every 0.4 s:
   t(s) truth net   P(H)  P(A)  P(D) | seen  since  until  d1_H  d1_A  slow_H  slow_A
    1.2   H    H   0.793 0.207 0.000 |  1.00   0.00   0.00  0.02  0.28    0.18    0.18
    1.6   H    H   0.793 0.207 0.000 |  1.00   0.00   0.00  0.02  0.31    0.18    0.09
 >  2.0   D    H   0.709 0.290 0.001 |  1.00   0.00   0.00  0.02  0.12    0.18    0.00
    2.4   D    H   0.606 0.393 0.000 |  0.00   0.04   1.00  0.25  0.11    0.09    0.00
    ...

You have 5 practice passes left, then one final reply.
Reply with a short note on what you are changing and why, then one ```edits block.
It applies on top of your current network.
