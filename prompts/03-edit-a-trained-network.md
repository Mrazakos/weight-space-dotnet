Below is a neural network that was trained with gradient descent. It works, and it has
known weaknesses. Your task is to make it better BY EDITING ITS WEIGHTS DIRECTLY.

There is no training. You may not run gradient descent, regression, or any search over
weight values. You work the way one debugs a program: read the weights, measure what
parts of the network compute, form a hypothesis about what causes a failure, and change
the weights that implement it. You must be able to say what your edit does and why.

WHAT IS KNOWN
  - It merges short possessions into their neighbours (too few possessions).
  - Most of the remaining error is which team, not when the ball is dead.
  - Some dead-ball frames just before a restart are labelled as live play.
These are observations, not instructions.

THE PYTHON ENVIRONMENT
Each reply may contain one ```python block. It runs; you get what it prints.
  W["head2.b"][2] += 0.3      read and write any weight: free
  forward(half, start, n)     run the network on practice data: COSTS 1 PASS
  score()                     score it on all practice matches:  COSTS 1 PASS
You have 25 passes.

HOW TO FINISH
Reply without a python block. State what you found, exactly which weights you changed,
why that should help on matches you have not seen, and what you measured.
If no edit helps, restore the original weights and say so:
an honest "no improvement found" is a valid result.
