# A talk from this repository

A storyline for about 25 minutes, with the asset for each slide. Every number is in
[experiment/REPORT.md](experiment/REPORT.md).

## One sentence

A language model wrote a working neural network as text, .NET turned the text into a network and
ran it, and the result is good enough to be interesting and honest enough to show where it fails.

## Slides

| # | slide | show | say |
|---|---|---|---|
| 1 | The question | title only | "A neural network is trained. What if it were written?" |
| 2 | The task | a still: `media/replay-expB-light-14s.png`, pitch only | For every frame: home, away or dead ball. The ball is unseen on 40% of frames. |
| 3 | How it is normally done | `/charts/results.svg`, cover all rows but the orange ones | 698,115 weights, 2.8 million labelled frames, a GPU. |
| 4 | The idea | the paper's title | Give the model the task and the network, every weight at zero. It replies with the weights. |
| 5 | The prompt | `prompts/01-write-from-scratch.md` | Point at two lines: "EVERY WEIGHT STARTS AT 0" and the decoder's penalties 16 and 24. |
| 6 | The reply | four lines of an edits block (README) | It is a program. Unit 0 reads a distance; unit 20 averages it over 9 frames; a gate opens. |
| 7 | **Demo 1: Write** | the app, `/write`, recording A | 396 of 698,115. Everything grey is still zero. |
| 8 | The .NET in it | the `IChatClient` snippet (README) | The model is an interface. A recording, a CLI, a hosted model: same code. |
| 9 | **Demo 2: Watch it play** | `media/replay-expA-dark.mp4`, 20 s of it | Written in one reply, never saw a frame. It is roughly right, and it over-calls dead ball. |
| 10 | Feedback | `prompts/02-practice-pass.md` | Now tell it how it did. Scores, a confusion matrix, a few frames where it was wrong. |
| 11 | The pass that broke it | `/charts/passes.svg` | Pass 5: a new idea, much worse. Pass 6: "the cause is…", and it fixes its own bug. No gradient anywhere. |
| 12 | **Demo 3: after feedback** | `media/replay-expB-dark.mp4` | 179 weights. Watch the latch hold possession through the shaded gaps. |
| 13 | Results | `/charts/results.svg` | Above the baseline everywhere. 0.784 against 0.842 where the ball is visible. 0.405 against 0.724 where it is hidden. |
| 14 | What did not work | text | Editing the *trained* network: 24 turns, six candidate edits, an honest "no improvement found". |
| 15 | The bug the port found | the `ReplayTests` doc comment | Rewriting the parser in C# and testing it against the recorded replies exposed a bug in the Python harness: one reply had been applied twice. |
| 16 | What it means | text | Knowledge of the task plus a little feedback goes far. What training adds is inference when the input is missing. And the weights can be read. |

## Demo script

Start the app before the talk: `dotnet run --project src/WeightSpace.Web`.

1. **Write** (`/write`): leave "Recording: experiment A" selected, press *Write the network*.
   About two seconds. Then switch to "experiment B" and press again: eleven replies, the weight
   count and the practice score tick up, and pass 5 visibly drops.
2. **Watch it play** (`/replay`): pick "A — written in one reply", play 15 seconds; switch to
   "B — final network" at the same position. Use *Dark stage* on a dark slide theme.
3. Optional, if the room has network and patience: `/write` with "Live: Claude through its CLI",
   effort *low*. It takes minutes and the network will be a different one each time. Say so first.

## If the demo fails

The four videos in `media/` are the replay page recorded as MP4. `media/page-write.png` and
`media/weights-expB.png` stand in for the Write step. Nothing in the talk depends on a network
connection unless you choose step 3.

## Questions to expect

- **"Is this better than training?"** No. Training wins on every source; see slide 13.
- **"Did it see the test data?"** The benchmark matches appear in no prompt and no feedback.
  Network A saw no data at all. Network B saw aggregate scores and 30 short excerpts from five
  other matches.
- **"Is it reproducible?"** The recorded replies are in the repository and the tests rebuild the
  networks from them. A new reply to the same prompt is a different network; we ran each experiment
  once and did not measure that spread.
- **"Why not just write rules?"** You could; this network *is* rules, expressed as weights. The
  point is that the same artefact can then be trained, edited or combined like any other network,
  and that a model can produce it from a description.
- **"Does the model's code run on my machine?"** In the two writing experiments, no: a reply is
  parsed, never executed.
