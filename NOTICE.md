# Notices

## The idea

This repository applies the method of *Sparks of Weight-Space Programming* by Rikiya Takehi,
Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL):
[post](https://rikiyat.github.io/blog/weight-space-programming/),
[logs](https://github.com/RikiyaT/lm_understand-logs). The prompt structure and the edits language
follow theirs. Nothing of their code or data is included.

## Data

- **The demo clip** (`data/clip/`) is 100 seconds of **Metrica Sports' public sample data**,
  Sample Game 3, first half: tracking positions, and possession labels derived from its event
  data. Source: <https://github.com/metrica-sports/sample-data>, published by Metrica Sports for
  the football-analytics community. Thank you to Metrica Sports for making it available.
- **The benchmark numbers** (`data/results/`, `experiment/eval/`) were computed on three external
  sources, none of which is redistributed here: the DFL open dataset (Bassek et al., 2025,
  *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data,
  CC BY 4.0, doi:10.6084/m9.figshare.28196177), SkillCorner Open Data (MIT), and the Metrica
  sample data above.
- **No data of the competition match** (More than a Hack 2027, Metrica Sports / FC Barcelona /
  Mobile World Capital) is in this repository: no tracking, no video, no labels, no frame-level
  output.

## The task and the trained baseline

The possession task, its metric (spellF1) and its rules are those of *More than a Hack 2027,
Phase 1*. The network architecture, the input features and the trained model that the
hand-written networks are compared with come from Team Sigue's solution to that challenge. The
trained weights are not included here; the demo shows only that model's labels on the clip.

## The models

The networks in `data/networks/` and `experiment/networks/` were written by Anthropic's Claude
(`claude-fable-5-2`) in the runs logged under `experiment/runs/`.
