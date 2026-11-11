"""Export what the .NET demo needs from the Python experiment (run once; outputs are committed).

    cd <hackathon repo>            # the checkout that holds wsp/, solution/, external/processed/
    .venv/Scripts/python <this repo>/tools/export_demo_data.py --out <this repo>/data

Writes
    networks/<name>.json     the hand-written networks, sparse: per tensor a shape, flat indices, values
    clip/clip.json           100 s of a held-out half: positions (decimetres), labels, metadata
    clip/features.bin        float32 [4 views][frames][48]: the network's input for that stretch,
                             with 1000 frames of context on each side (features come from the
                             Python feature code; the .NET side does not recompute them)
    reference/<name>.bin     PyTorch's result on the same input, for the parity tests:
                             float32 [frames][3] averaged probabilities, then uint8 [frames] labels
    results/benchmark.json   the numbers behind the charts

The clip is from Metrica Sports' public sample data (sample game 3), see NOTICE.md. Nothing here
comes from the competition match.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "wsp", "harness"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import evaluate  # noqa: E402
import netspec  # noqa: E402
from features import home_attacks_positive  # noqa: E402
from model import viterbi  # noqa: E402

HALF, N, MARGIN, FPS = 18, 2500, 1000, 25      # benchmark half 18 = Metrica sample game 3, first half
NETWORKS = {
    "expA": ("wsp/networks/expA_scratch_oneshot/hand.pt",
             "A: written in one reply, no data, no feedback",
             [{"label": "last-controller latch", "tensor": "gru1f", "unit": 2, "min": -1, "max": 1},
              {"label": "dead-ball latch", "tensor": "gru1f", "unit": 3, "min": 0, "max": 1}]),
    "expB": ("wsp/networks/expB_scratch_passes10/hand.pt",
             "B: written, then revised over 10 practice passes",
             [{"label": "forward possession latch (last controller)", "tensor": "gru1f", "unit": 0, "min": -1, "max": 1},
              {"label": "backward possession latch (next controller)", "tensor": "gru1b", "unit": 0, "min": -1, "max": 1}]),
}


def sparse(state, name, description, probes, extra=None):
    tensors = {}
    for key, arr in state.items():
        flat = arr.reshape(-1)
        idx = np.flatnonzero(flat)
        tensors[key] = {"shape": list(arr.shape), "i": idx.tolist(),
                        "v": [float(np.float32(x)) for x in flat[idx]]}
    out = {"name": name, "description": description, "nonzero": netspec.n_nonzero(state),
           "total": netspec.N_PARAMS, "probes": probes, "tensors": tensors}
    out.update(extra or {})
    return out


def reference(state, views):
    """Exactly what the .NET side computes: one window, four views averaged, decoder over it all."""
    net = evaluate.net_from_state(state)
    probs = []
    with torch.no_grad():
        for X, swapped in views:
            p = torch.softmax(net(torch.from_numpy(X)[None])[0], -1).numpy()
            probs.append(p[:, [1, 0, 2]] if swapped else p)
    p = np.mean(probs, 0).astype(np.float32)
    sh, sa = evaluate.decoder()
    return p, viterbi(np.log(np.clip(p, 1e-6, 1)), sh, sa).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = ap.parse_args().out
    for sub in ("networks", "clip", "reference", "results"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    half = evaluate.load_half("benchmark", HALF)
    y = half["y"]
    changes = np.concatenate(([0], np.cumsum(np.diff(y) != 0)))       # busiest stretch, as in figures.py
    cand = np.arange(2000, len(y) - N - 2000, 250)
    start = int(cand[np.argmax(changes[cand + N] - changes[cand])])
    a0, b0 = start - MARGIN, start + N + MARGIN
    views = [(np.ascontiguousarray(X[a0:b0], np.float32), s) for X, s in half["views"]]
    np.stack([X for X, _ in views]).astype("<f4").tofile(os.path.join(out, "clip", "features.bin"))

    z = np.load(sorted(glob.glob(os.path.join(ROOT, "external", "processed", "MET-*.npz")))[2])
    ball, home, away = z["h1_ball"], z["h1_home"], z["h1_away"]
    assert len(ball) == len(y), "half 18 is not MET-3 first half"

    def deci(a):                                                        # metres -> decimetres, NaN -> null
        q = np.round(a[start:start + N] * 10)
        return [[None if not np.isfinite(v) else int(v) for v in row.reshape(-1)] for row in q]

    sh, sa = evaluate.decoder()
    labels = {"truth": y[a0:b0].astype(int).tolist()}
    trained = netspec.from_checkpoint(os.path.join(ROOT, "models", "expB_bw3met6k_s0.pt"))
    labels["trained"] = reference(trained, views)[1].astype(int).tolist()

    for name, (path, description, probes) in NETWORKS.items():
        state = netspec.from_checkpoint(os.path.join(ROOT, path))
        with open(os.path.join(out, "networks", name + ".json"), "w") as fh:
            json.dump(sparse(state, name, description, probes), fh)
        p, lab = reference(state, views)
        with open(os.path.join(out, "reference", name + ".bin"), "wb") as fh:
            fh.write(p.astype("<f4").tobytes())
            fh.write(lab.tobytes())
        print(name, "nonzero", netspec.n_nonzero(state), "clip accuracy %.3f" % (lab[MARGIN:-MARGIN] == y[start:start + N]).mean())

    # experiment B after every practice pass, to replay how the network changed
    history = json.load(open(os.path.join(ROOT, "wsp", "runs", "expB_scratch_passes10", "state.json")))["history"]
    for n in range(1, 11):
        state = netspec.load_sparse(os.path.join(ROOT, "wsp", "networks", "expB_scratch_passes10", "pass_%02d.npz" % n))
        dev = json.load(open(os.path.join(ROOT, "wsp", "runs", "expB_scratch_passes10", "turn_%02d" % (n - 1), "dev_eval.json")))
        extra = {"pass": n, "dev": {s: {"spell_f1": v["spell_f1"], "macro_f1": v["macro_f1"],
                                        "n_pred_spells": v["n_pred_spells"], "n_gt_spells": v["n_gt_spells"]}
                                    for s, v in dev["sources"].items()}}
        with open(os.path.join(out, "networks", "expB_pass%02d.json" % n), "w") as fh:
            json.dump(sparse(state, "expB_pass%02d" % n, "B after practice pass %d" % n,
                             NETWORKS["expB"][2], extra), fh)

    clip = {
        "source": "Metrica Sports sample data, Sample Game 3, first half (github.com/metrica-sports/sample-data)",
        "fps": FPS, "frames": N, "margin": MARGIN, "start_frame": start, "features": 48, "views": 4,
        "view_swapped": [bool(s) for _, s in views],
        "pitch": [105.0, 68.0], "units": "decimetres, pitch centre at the origin",
        "home_attacks_positive_x": bool(home_attacks_positive(home, away)),
        "decoder": {"switch_ha": sh, "switch_alive": sa},
        "labels": labels, "label_names": ["H", "A", "D"],
        "ball": deci(ball), "home": deci(home), "away": deci(away),
    }
    with open(os.path.join(out, "clip", "clip.json"), "w") as fh:
        json.dump(clip, fh, separators=(",", ":"))

    rows = [("Nearest player + 4 s vote", "none", "ref_nearest_benchmark.json", None),
            ("A  written in one reply", "written", "expA_scratch_oneshot_benchmark.json", None),
            ("B  written, 10 practice passes", "written", "expB_scratch_passes10_benchmark.json", None),
            ("Trained, one model", "trained", "trained_single_benchmark.json", None),
            ("Trained, 3-model ensemble", "trained", "trained_ensemble3_benchmark.json", None)]
    bench = []
    for label, kind, fname, key in rows:
        r = json.load(open(os.path.join(ROOT, "wsp", "eval", fname)))
        bench.append({"label": label, "kind": kind,
                      "sources": {s: {k: r["sources"][s][k] for k in
                                      ("spell_f1", "macro_f1", "n_pred_spells", "n_gt_spells")}
                                  for s in ("DFL", "SC", "MET")}})
    with open(os.path.join(out, "results", "benchmark.json"), "w") as fh:
        json.dump({"metric": "spellF1 on the held-out benchmark",
                   "source_names": {"DFL": "DFL (optical, degraded)", "SC": "SkillCorner (broadcast)",
                                    "MET": "Metrica game 3"},
                   "weights": {"A  written in one reply": 396, "B  written, 10 practice passes": 179,
                               "Trained, one model": 698115, "Trained, 3-model ensemble": 3 * 698115},
                   "rows": bench, "expB_dev_history": history}, fh, indent=1)
    print("clip start frame", start, "frames", N, "+/-", MARGIN, "players", home.shape[1], away.shape[1])


if __name__ == "__main__":
    main()
