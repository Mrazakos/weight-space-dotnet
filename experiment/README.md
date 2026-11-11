# The experiment

A copy of the `wsp/` folder of the hackathon repository where the experiment was run, as it stood
when this repository was made.

- **[REPORT.md](REPORT.md)** — the write-up: idea, protocol, three experiments, results, limits.
- **[JOURNAL.md](JOURNAL.md)** — what was done in order, with the numbers and the mistakes.
- `runs/<experiment>/turn_NN/` — per turn: what was sent (`prompt.txt` / `message.txt`), the reply
  (`answer.md`), the event stream, token counts and the tool-free audit (`run.json`), and how the
  edits were read (`apply.json`). Experiment C's code and outputs are in `tool_calls.jsonl`.
- `networks/` — the hand-written networks as PyTorch checkpoints, and experiment B after each pass.
- `eval/` — the full benchmark result behind every number.
- `fig/` — the figures of the report.
- `harness/`, `tests/` — the Python that ran it.

The Python harness is here for reading. It imports the feature code, the model and the metric of
the hackathon solution (`solution/`, `evaluation/`) and needs the external tracking data, none of
which is in this repository, so it does not run from here. The .NET projects in `src/` need none of
that.

One file is deliberately missing: experiment C's `edited.pt`. The model changed no weight, so that
file is the trained model itself, which belongs to the hackathon team's repository.
