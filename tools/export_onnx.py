"""Export the architecture as ONNX, and check ONNX files against PyTorch.

    cd <hackathon repo>
    .venv/Scripts/python <this repo>/tools/export_onnx.py --out <this repo>/data/onnx
    .venv/Scripts/python <this repo>/tools/export_onnx.py --verify some.onnx --against expB

Writes
    possessionnet.onnx    the architecture with EVERY WEIGHT 0: the template. All 24 weight tensors
                          are present as ordinary initializers, stored as raw float32 bytes.
    expA.onnx, expB.onnx  the two hand-written networks: the template with their weights in it.
    manifest.json         initializer names and shapes, and the state-dict key each one holds.

The .NET side needs no ONNX package for these files. WeightSpace.Core reads the initializers
straight out of the protobuf, and writes a network to ONNX by copying the template and overwriting
the weight bytes in place, which works because every network has the same tensors of the same
size. --verify is how a file written by .NET is checked: it runs the file in ONNX Runtime and
compares with PyTorch on the same input.

Two things about the export itself:

- No constant folding. PyTorch stores a GRU's gates as (reset, update, new), ONNX wants (update,
  reset, new) stacked per direction. With folding off that conversion stays in the graph as
  Slice / Concat nodes on the ORIGINAL tensors, so the initializers keep PyTorch's names and layout.
- The exporter merges initializers that hold identical values, and all-zero tensors of equal shape
  would collapse into one. So the graph is exported once with distinct random weights, which keeps
  one initializer per tensor, and the values are set afterwards.

Graph:   x [1, T, 48]  ->  logits [1, T, 3] (H, A, D),  gru1 [1, T, 256] (layer-1 states, forward
then backward: what the app plots as the hand-written "latch" units).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "wsp", "harness"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import netspec  # noqa: E402
from model import PossessionNet  # noqa: E402

H = netspec.HIDDEN
NETWORKS = {"expA": "wsp/networks/expA_scratch_oneshot/hand.pt",
            "expB": "wsp/networks/expB_scratch_passes10/hand.pt"}


class Exportable(nn.Module):
    """PossessionNet with its two GRU layers as separate modules, so layer 1 is an output too.
    Same computation; state-dict key gru.*_l0* lives in gru1.*_l0*, gru.*_l1* in gru2.*_l0*."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(netspec.N_FEAT, H), nn.GELU())
        self.conv = nn.Conv1d(H, H, 9, padding=4)
        self.gru1 = nn.GRU(H, H, num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(2 * H, H, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Linear(H, 3))

    def forward(self, x):
        a = self.inp(x)
        h = a + torch.relu(self.conv(a.transpose(1, 2))).transpose(1, 2)
        g1, _ = self.gru1(h)
        g2, _ = self.gru2(g1)
        return self.head(torch.cat([g2, h], -1)), g1


def onnx_name(key):
    """state-dict key of PossessionNet -> initializer name in the ONNX graph."""
    if key.startswith("gru."):
        layer = 1 if "_l0" in key else 2
        return "gru%d.%s" % (layer, key[4:].replace("_l1", "_l0"))
    return key


def with_weights(template, state):
    """A copy of the template model with this network's weights as its initializers."""
    import onnx
    from onnx import numpy_helper
    model = onnx.ModelProto()
    model.CopyFrom(template)
    wanted = {onnx_name(k): v for k, v in state.items()}
    done = set()
    for init in model.graph.initializer:
        if init.name in wanted:
            value = np.ascontiguousarray(wanted[init.name], np.float32)
            assert tuple(init.dims) == value.shape, init.name
            init.CopyFrom(numpy_helper.from_array(value, init.name))        # raw float32 bytes
            done.add(init.name)
    assert done == set(wanted), sorted(set(wanted) - done)
    return model


def reference(state, x):
    net = PossessionNet(netspec.N_FEAT, hidden=H)
    net.load_state_dict(netspec.to_torch_state(state))
    with torch.no_grad():
        return net.eval()(torch.from_numpy(x)).numpy()


def run_onnx(path, x):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(path, options).run(None, {"x": x})[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="folder to export into")
    ap.add_argument("--verify", help="an .onnx file to check against PyTorch")
    ap.add_argument("--against", choices=sorted(NETWORKS), help="which network --verify should equal")
    args = ap.parse_args()
    x = np.random.default_rng(0).standard_normal((1, 700, netspec.N_FEAT)).astype(np.float32)

    if args.verify:
        state = netspec.from_checkpoint(os.path.join(ROOT, NETWORKS[args.against]))
        diff = np.abs(run_onnx(args.verify, x) - reference(state, x)).max()
        print("%s vs PyTorch %s: max |difference| of the logits %.1e  ->  %s" % (
            os.path.basename(args.verify), args.against, diff, "OK" if diff < 1e-3 else "MISMATCH"))
        sys.exit(0 if diff < 1e-3 else 1)

    import onnx
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "possessionnet.onnx")
    torch.manual_seed(0)
    torch.onnx.export(Exportable().eval(), (torch.zeros(1, 64, netspec.N_FEAT),), path,
                      input_names=["x"], output_names=["logits", "gru1"],
                      dynamic_axes={"x": {1: "frames"}, "logits": {1: "frames"}, "gru1": {1: "frames"}},
                      opset_version=17, do_constant_folding=False, dynamo=False)
    zero = netspec.zero_state()
    template = with_weights(onnx.load(path), zero)
    onnx.checker.check_model(template)
    onnx.save(template, path)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"input": "x [1, frames, 48]", "outputs": ["logits [1, frames, 3]", "gru1 [1, frames, 256]"],
                   "opset": 17,
                   "initializers": {onnx_name(k): {"state_dict_key": k, "shape": list(v.shape)}
                                    for k, v in zero.items()}}, fh, indent=1)

    for name, ckpt in NETWORKS.items():
        state = netspec.from_checkpoint(os.path.join(ROOT, ckpt))
        file = os.path.join(args.out, name + ".onnx")
        onnx.save(with_weights(template, state), file)
        print("%s.onnx vs PyTorch: max |difference| of the logits %.1e" % (
            name, np.abs(run_onnx(file, x) - reference(state, x)).max()))
    print("zero template: max |logit| %.1e" % np.abs(run_onnx(path, x)).max())
    for f in sorted(os.listdir(args.out)):
        print("%-22s %9d bytes" % (f, os.path.getsize(os.path.join(args.out, f))))


if __name__ == "__main__":
    main()
