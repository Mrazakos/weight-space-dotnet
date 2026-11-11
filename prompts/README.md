# The prompts, reduced to their core ideas

Three short texts for slides. Each keeps the structure and the wording of a real prompt of the
experiment and drops the bulk (the full input table, the feature code, the exact formulas).
**They are for showing, not for sending**: a model cannot write a valid network from the short
version, because it does not say what the 48 inputs are.

| file | what it condenses | the real thing |
|---|---|---|
| [01-write-from-scratch.md](01-write-from-scratch.md) | the prompt of experiments A and B: write the network, no training | 24,154 characters: [`experiment/runs/expA_scratch_oneshot/turn_00/prompt.txt`](../experiment/runs/expA_scratch_oneshot/turn_00/prompt.txt) |
| [02-practice-pass.md](02-practice-pass.md) | one feedback message of experiment B (the pass that broke the network) | [`experiment/runs/expB_scratch_passes10/turn_05/message.txt`](../experiment/runs/expB_scratch_passes10/turn_05/message.txt) |
| [03-edit-a-trained-network.md](03-edit-a-trained-network.md) | the prompt of experiment C: improve a trained network by editing its weights | [`experiment/runs/expC_edit_trained/turn_00/prompt.txt`](../experiment/runs/expC_edit_trained/turn_00/prompt.txt) |

What every real prompt contains, in this order (the order of the prompts in the paper's logs):

1. **The task and its rules.** No training, no tools, how many attempts.
2. **The problem**, with the reference source code of the input.
3. **THE INPUT**: every number the network sees, with its formula and range.
4. **THE OUTPUT**: what the outputs mean and exactly how they are used afterwards.
5. **THE NETWORK**: the formulas and the name and shape of every tensor. "Every weight starts at 0."
6. **Scoring**: how the result is judged and what it is compared with.
7. **How to write the weights**: the edits language, one example per feature.

Two things in the short versions are worth pointing at on a slide:

- **"Every weight starts at 0."** The model is not adjusting anything. Whatever the network does,
  the model wrote.
- **The decoder's penalties are in the prompt.** The model used them: its design note says a
  possession change needs "about 2 s of clear control ... to pay the 32", which is the task's own
  2-second rule turned into arithmetic.

The numbers in `02-practice-pass.md` are those of the real pass-5 message. Columns and rows were
dropped to fit a slide; nothing was changed. The model's next reply read "Pass 5 broke dead-ball
recall (0.91 → 0.48)" off this message, named the cause and fixed it.
