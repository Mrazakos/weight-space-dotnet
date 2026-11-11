**Result: no improvement found. The working network is the original, unedited network** (all tensors verified bit-identical to `BASE`; 20 of 25 passes used).

## 1. What I found

Baseline on the practice halves (real pipeline): spellF1 0.727 on DFL, 0.904 on SC, 0.983 on MET. Only DFL sits near the held-out level you quoted, so I treated it as the representative source; SC and MET are largely known to the network.

- **The decoder is nearly idle.** The median H-vs-A logit gap is 5–12, so decoded and argmax frame accuracy are equal (0.9604 vs 0.9605). The network itself fixes the possession structure.
- **The team decision lives in the recurrent stack.** Either layer-2 direction alone gives 0.81–0.98 team accuracy; the direct `h` path alone gives 0.59–0.67, and removing it changes little.
- **Missed possessions are low-margin, not confident errors.** Their mean margin is about −0.5 to −2, against +5 to +10 on correct ones. The visible-ball cases I traced were ambiguous in the inputs: long clearances with both teams 20–40 m from the ball, or the ball 10 m from everyone.
- **Boundary timing is unbiased.** Median offset is 0.00–0.04 s for turnovers, stops and restarts on DFL and SC. DFL turnovers are noisy (interquartile range −0.28 to +0.46 s) but not systematically late or early.
- **The under-splitting is real but sits at a trade-off.** DFL has 497 predicted possessions against 596 true ones, yet every knob that adds possessions adds as many wrong ones.

## 2. Weights changed

None. I never assigned into `W`. Each candidate edit was tested in an offline re-implementation of the network that I built from the stored inputs. It reproduces `h` exactly and the logits to a mean error of 0.001–0.06, and it emulates the four-view averaging and decoder.

| Candidate edit | Hypothesis | Measured (four-view, 4 DFL + 2 SC chunks unless noted) |
|---|---|---|
| Scale `head2.W`, `head2.b` (×0.5 to ×2) | Logits mis-scaled against the 16/24 penalties | DFL best at ×1.0 (0.749; 0.730–0.743 elsewhere). SC 0.912 at ×0.7 vs 0.890, which is one or two possessions out of 75, with DFL moving the other way |
| Shift `head2.b[2]` (−1 to +1) | D-vs-live balance is off | DFL best at 0 (0.749; 0.724–0.746 otherwise); SC within ±0.008 |
| Shift `head2.b[0]` against `head2.b[1]` | A bias toward one output slot weakens the role-swap average | Refuted without labels: the bias is −0.10 to +0.37 logits with inconsistent sign across chunks |
| `gru2f/gru2b.biz` ∓0.5 | States too sticky, so short possessions never register | Mechanism confirmed (DFL possessions 144 → 154 at −0.5, 134 at +0.5, true 174), but DFL +0.007 and SC −0.015 |
| `gru1f/gru1b.biz` ∓0.5 | Same, layer 1 | DFL +0.003, SC −0.017 |
| Mean-ablate `head1.Wf`, `Wb` or `Wh` path (single view, 4 chunks) | One path injects noise | Removing either recurrent path costs spellF1 on every chunk; removing `h` is mixed (−0.07 to +0.01) |

A read of the weights showed nothing structural: no dead units in `m`, home and away input columns balanced, no non-finite values.

## 3. Why I am not shipping an edit

- **Every global knob is already at its optimum.** The measured changes are a few possessions out of 75–174 and flip sign between sources. Picking the best of them would be fitting noise on matches the network was trained on.
- **The remaining errors look information-limited.** I found no bug to repair: the network is uncertain where the inputs are ambiguous, and its bias and timing are centred.
- **The one real structural effect is not reachable by an edit.** The role-swapped views disagree by about 0.7–0.8 logits in ambiguous frames, but the disagreement is frame-specific. The view averaging already absorbs it (about +0.02 spellF1 on DFL), and a single network of this width cannot be made role-symmetric by changing weights.
- **Deeper edits could not be validated here.** Reducing reliance on team-shape features, for example, would look harmful on practice matches the network has seen even if it helped on unseen ones.

## 4. What I measured

- One `score()` for the baseline; I did not run a final one because the weights are unchanged.
- 19 `forward()` passes. Nine of them were lost when the Python process restarted mid-run and six were spent re-collecting inputs, so the final sensitivity tests rest on 6 chunks (4 DFL, 2 SC; 144,000 frames) rather than the 13 I had first collected.
- On that data: error anatomy by possession length and cause, path ablations, event-aligned margin profiles and boundary offsets, per-view accuracy and role-swap bias, and the sensitivities in the table above.