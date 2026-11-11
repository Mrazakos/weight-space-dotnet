# Journal — weight-space programming on the possession model

Chronological log of what was done, why, and what came out. Numbers are spellF1 unless stated.
Dates are 2026.

## Nov 10 — setup

**Idea.** *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab;
[post](https://rikiyat.github.io/blog/weight-space-programming/),
[logs](https://github.com/RikiyaT/lm_understand-logs)) reports that a frontier LLM can write or edit
the weights of a neural network directly as text, guided by a description of the task, with no
gradient descent. We try it on our task: per-frame ball possession (H / A / D) from automatic
tracking.

**Decisions.**

1. *Same architecture as the trained model.* The network to write is `PossessionNet(48, 128)` from
   `solution/model.py`, 698,115 weights, all starting at 0. A hand-written checkpoint has the same
   format as `models/*.pt`, so it runs through `eval_fold.py` and `predict.py` unchanged, and the
   comparison "SGD on 2.8M labelled frames vs written by hand" is like for like: same features,
   same decoder, same four-view averaging. This mirrors the paper's Exp. 5, where the hand-written
   MinAtar policies have the shape of the PPO baselines.
2. *The writer is a fresh, tool-free model call.* Like the paper's `codex exec` with every tool
   disabled, each turn is `claude -p --tools "" --strict-mcp-config --safe-mode` run from an empty
   folder with a one-line system prompt. The session reports `tools: []` and `mcp_servers: []`;
   `run.json` records that as `audit.tool_free`. Model `claude-fable-5-2`.
3. *Three experiments*, following the paper's: **A** one reply, no feedback (Exp. 5); **B** the
   same with 10 practice passes of feedback (the idea of Exp. 6); **C** edit a trained network
   through a Python kernel with a budget of forward passes (Exp. 4 / 8).
4. *Two splits, external data only.* `benchmark` is the fixed held-out benchmark of
   `docs/TECHNICAL.md` (DFL 5–6, SkillCorner 15–19, Metrica game 3); writers never see it.
   `dev` (DFL 0, SkillCorner 0–2, Metrica game 1) is what feedback comes from in B and C. The
   provided match and its public ground truth appear in no prompt and no feedback.
5. *No empirical statistics in the from-scratch prompt*, except two pieces of general football
   knowledge (ball in play 55–65% of the time, roughly 100–150 possessions per half). Feature
   ranges are the analytic ones from `features.py`.

**Environment.** This clone had no data. External data and the match were linked (hardlinks and a
junction, nothing copied) from the team's other working copy with `wsp/setup_local_data.ps1`;
parsed with the existing `solution/*_parse.py`. Python 3.13 venv, `torch 2.6.0+cpu` and the pins of
`solution/requirements-solution.txt`. CPU only (i7-1260P).

**A snag: inference speed.** `model.predict` runs a half as ~17 windows one after another. On this
CPU the recurrent layers cost about 1 ms per frame, which would make one benchmark run of one model
take over two hours. The windows are independent, so `evaluate.predict_batched` sends all windows of
equal length, from all four views, through the network as one batch: same windows, same stitching,
log-probabilities equal to `predict` within 2·10⁻⁵ (unit-tested), about 12× faster. A benchmark run
of one model now takes ~12 minutes. `eval_fold.py` and `predict.py` are untouched and still work on
any checkpoint produced here; they are just slow without a GPU.

**Harness** (`wsp/harness/`): `netspec.py` (network + tensor aliases), `edits.py` (the edits
language), `llm.py` (one tool-free turn, logged), `build_prompt.py`, `evaluate.py`, `feedback.py`,
`guard.py` + `kernel.py` (analysis kernel for C), `run_scratch.py`, `run_edit.py`. 17 unit tests in
`wsp/tests/`. The only change to existing code is that `eval_fold.benchmark_halves()` takes the
match indices as parameters, defaulting to the benchmark.

## Nov 10 — reference rows on the benchmark

Scored with `wsp/harness/evaluate.py` (four views averaged, decoder penalties from
`models/decoder.json`), spellF1 / macroF1:

| | DFL | SkillCorner | Metrica g3 |
|---|---|---|---|
| nearest player + 4 s majority vote (`starter/baseline.py` logic) | 0.202 / 0.533 | 0.318 / 0.702 | 0.416 / 0.812 |
| trained by SGD, one model (`models/expB_bw3met6k_s0.pt`) | 0.724 / 0.912 | 0.726 / 0.913 | 0.842 / 0.934 |

The baseline invents about twice as many possessions as there are (5,734 against 2,843 over the 20
benchmark halves). The trained single model is a little below the published 3-model ensemble
(0.741 / 0.747 / 0.855), as expected.

## Nov 10 — experiment A: one reply, no feedback

`wsp/runs/expA_scratch_oneshot/turn_00/`. One fresh conversation, the 24,154-character prompt
(`prompt.txt`), one reply (`answer.md`), effort `xhigh`.

- 1,216 s wall time; 114,658 output tokens, of which 110,189 were reasoning. The reasoning itself is
  not exposed by the CLI (only an opaque signature, which the logs omit); the reply is 8,051
  characters: a design note and the edits block.
- `audit.tool_free: true` (the session listed no tools and no MCP servers; the reply has no tool call).
- 140 edit statements, 140 applied, 0 errors. **396 of 698,115 weights are nonzero.**

What it wrote (its own description, `answer.md`): hinge units for "a player is close to the ball and
moving with it", closeness to the lines, ball speed and gap length; 9-frame averages in the conv
layer; in the first GRU a *last-controller latch* (possession stays with the last team through gaps
and ball flights), forward and backward *dead-ball latches* set by "ball out" or "ball stationary"
and reset by live motion, slow and fast running averages of player speed whose difference says
"players are slower than usual", and a restart-taker trace; a head that outputs `H − A = 2p` and a
clipped dead-ball score, with the scale chosen against the decoder's penalties ("about 2 s of clear
control by the other team is needed to pay the 32 for a there-and-back possession change").

**Result on the held-out benchmark** (spellF1 / macroF1):

| | DFL | SkillCorner | Metrica g3 |
|---|---|---|---|
| nearest-player baseline | 0.202 / 0.533 | 0.318 / 0.702 | 0.416 / 0.812 |
| **A: written by hand, one shot, 396 weights** | **0.378 / 0.741** | **0.471 / 0.802** | **0.546 / 0.809** |
| trained by SGD, same architecture, one model | 0.724 / 0.912 | 0.726 / 0.913 | 0.842 / 0.934 |

So a network that has never seen a frame of data, written in one reply, clearly beats the baseline
on the ranking metric on all three sources (+0.13 to +0.18) and closes roughly a third of the gap
to the trained network. Where it falls short: it produces too few possessions (2,285 against 2,843
true; the trained model 2,623) and over-calls dead ball (D recall 0.94 but precision 0.79; H and A
recall about 0.70). On Metrica its D precision is only 0.67: it reads long ball gaps as stoppages.

**Usage note.** The reply cost about four times the output tokens I had estimated. Asked how to
budget B and C, the team chose to run C next, at effort `high`.

## Nov 10 — the hand-written network on the provided match (pipeline check)

The point of sharing the trained model's checkpoint format is that nothing else has to change:

```
python solution/predict.py --models wsp/networks/expA_scratch_oneshot/hand.pt --tta --out submission/Sigue_wsp_A.csv
python evaluation/validate_submission.py submission/Sigue_wsp_A.csv      ->  Valid, 199,319 rows
```

The file reports 261 possessions of mean 11.3 s. On the public 5 minutes it scores spellF1 0.339 /
macroF1 0.654 (the four shipped baselines: 0.038 to 0.424 / 0.58 to 0.62). As everywhere in this
repository, that window has 17 possessions and is used to check that the pipeline is wired
correctly — frames aligned, labels meaning what they should — and for nothing else. No prompt, no
feedback and no decision in this work uses the provided match.

The zero network (every weight 0) scores spellF1 0.000 on all three benchmark sources, as it must:
the decoder outputs a constant.

## Nov 10 — pipeline check, and experiment C paused

**Pipeline check passed.** The trained 3-model ensemble (`models/expB_bw3met6k_s{0,1,2}.pt`) scores
0.741 / 0.747 / 0.855 spellF1 on this machine: the numbers published in `docs/TECHNICAL.md`.

**Experiment C is PAUSED, not finished.** `wsp/runs/expC_edit_trained/` holds 15 turns (0-14) and
`tool_calls.jsonl`; 14 of 25 forward passes are used; **no weight has been changed yet**. The run
stopped because the account's usage limit was reached, not because of an error in the experiment.
All 15 turns were analysis. What the model had established:

- the trained network's logits are very sharp, so the decoder changes little; the network itself
  fixes the possession structure;
- both layer-2 GRU directions carry the team decision redundantly; the direct `h` path into the
  head hardly matters for it;
- on the DFL practice halves it finds 236 possessions against 273 true (under-splitting), and at
  true turnovers the median margin for the new team stays under 2 logits for 2 s, against 5 to 7
  at restarts: turnovers are where it is unsure;
- the SkillCorner and Metrica practice halves are largely memorised (0.93 and 1.0), because they
  were training matches, so only DFL is informative. A cleaner dev split is the first thing to fix.

Resume with the same command (finished turns are not re-sent; the kernel's variables are lost,
its pass count and weights are kept):

    python wsp/harness/run_edit.py --run expC_edit_trained --base models/expB_bw3met6k_s0.pt --budget 25 --max-turns 40 --block-minutes 30 --effort high

Harness friction seen in C: the code guard refused two harmless blocks (`_` as a loop variable,
`import time`).

**Not done:** experiment B (not started), C's edit and its benchmark score, `wsp/README.md`,
`wsp/REPORT.md`.

## Nov 11 — experiment C resumed and finished: no edit

Resumed from turn 15 with the same command. Two harness changes at the restart, both recorded in
the message the model received: a note that its variables were gone, and the guard now accepts the
single name `_`.

- 24 turns in total, 23 code blocks (4 refused by the guard), 20 of 25 forward passes,
  52,418 output tokens, about 10 minutes of model time and 42 minutes of kernel time.
- **0 of 698,115 weights changed.** `final_answer.md` opens with "no improvement found"; the
  model verified the working network bit-identical to the original.
- Benchmark, base and "edited" alike: 0.724 / 0.726 / 0.842.

It tested six candidate edits in an offline copy of the network it had built from stored inputs:
output scale, dead-ball bias, H-against-A bias, the update-gate biases of both GRU layers, and
removing each path into the head. The best was +0.007 on DFL with −0.015 on SkillCorner. Its
conclusion: every global knob is at its optimum, the remaining errors are low-margin and sit where
the inputs are ambiguous, and picking the best of the measured differences would be fitting noise
on matches the network was trained on.

Things that went wrong on our side, for whoever repeats this:

- **Wrong dev split for C.** SkillCorner and Metrica dev halves were training matches of the
  subject (0.90 and 0.98 there against 0.73 and 0.84 held-out). Only DFL was informative. C should
  get practice matches the subject has not seen, which means editing a model trained on fewer
  matches or holding more out.
- **The pause cost data.** Nine passes' worth of collected activations were lost with the kernel's
  memory; six passes went on re-collecting.
- **The guard is stricter than it needs to be** (`_`, `import time`, `globals`, `eval` refused).

## Nov 11 — where this leaves the submission

Unchanged. A is far below the trained model, C changed nothing, B was not run. The write-up is
`wsp/REPORT.md`.

## Nov 11 — side check: the hand-written network as a fourth ensemble member

Trained 3-model ensemble plus A's network, benchmark spellF1: 0.741 / 0.749 / 0.856 against
0.741 / 0.747 / 0.855 without it. No effect; the differences are far inside the noise band.

## Nov 11 — experiment B: ten practice passes

`wsp/runs/expB_scratch_passes10/`, effort `medium` at the team's request. The job was killed once
by the host for low memory after reply 4 and continued with the same command; no reply was re-sent.

- 11 replies, 35,667 output tokens (27,603 reasoning), 430 s of model time, all tool-free,
  0 edit statements rejected. Nonzero weights: 64 after the first reply, 179 at the end.
- Dev spellF1 over all halves by pass: 0.480, 0.495, 0.548, 0.544, 0.463, 0.558, 0.557, 0.559,
  0.553, 0.559. By source at pass 10: DFL 0.369, SkillCorner 0.617, Metrica 0.765.
- Pass 5 broke dead-ball recall (0.91 to 0.48) with a new ball-displacement term; reply 5 names
  the cause (across a ball gap the forward and backward position averages differ, so unseen
  stoppages read as "ball moving") and gates the term on the ball being seen.
- Pass 9 tried scaling all logits by 1.3; it scored lower, and reply 9 begins with `restore 8`.
- The final reply contains an empty edits block: the model kept the pass-10 network.

**Benchmark (never shown to the model):** 0.405 / 0.547 / 0.784 spellF1, 0.770 / 0.842 / 0.916
macroF1. Possessions found 2,692 of 2,843 (A: 2,285). On Metrica that is within 0.06 of the
trained single model (0.842) with 179 weights. On DFL it is 0.027 above A, inside the noise band;
the model's closing note attributes this to possession changes inside DFL's long ball gaps being
invisible to its design.

A and B are not a clean ablation (different effort, separately designed networks, A never scored
on dev). What B shows on its own: feedback raised its dev score by 0.08, nearly all by pass 6, and
the result held on unseen matches.

## Nov 11 — a harness bug in experiment B: one reply was applied twice

Found while porting the edits parser to C# (the port's replay test refused to reproduce the stored
network). **Cause:** `run_scratch.py` saved the working network right after applying a reply, before
the turn was complete. Experiment B was killed for low memory after reply 3 had been applied but
before its practice pass finished; on resume the driver loaded that saved network and applied
reply 3 again. Assignments (`=`) are idempotent, in-place statements are not, and reply 3 has
exactly one: `head1.b[1] += 0.3` (its "+0.3 to the dead score").

**Effect:** one weight. From pass 4 on, `head.0.bias[1]` was −1.525 in the network that was scored,
where the replies specify −1.825. Nothing else differs (checked tensor by tensor against a clean
replay of all eleven replies; the pass-9 `*=` scaling was undone by `restore 8`).

**What it does to the result:** almost nothing.

| benchmark spellF1 | DFL | SkillCorner | Metrica g3 |
|---|---|---|---|
| B as scored (dead-score bias +0.6) | 0.405 | 0.547 | 0.784 |
| B as written (each reply applied once, +0.3) | 0.402 | 0.549 | 0.783 |

`wsp/networks/expB_scratch_passes10/as_written.pt`, `wsp/eval/expB_as_written_benchmark.json`.
The reported numbers stay those of the network that was scored, because that is the network the
model received feedback on from pass 4 and chose to keep. What is *not* true is that the scored
network is exactly what the model wrote; it differs by that one bias.

**Fix:** the driver now checkpoints the network only when a turn is complete
(`after_turn_NN.npz`) and resumes from there, so an interrupted turn starts from the state before
its reply. Experiments A and C are unaffected (A has one turn; C edits through the kernel).

## Nov 11 — side check: B as a fourth ensemble member

Trained 3-model ensemble plus B's network: 0.737 / 0.743 / 0.851, against 0.741 / 0.747 / 0.855
without it. Slightly lower, inside the noise band. Neither hand-written network improves the
ensemble.
