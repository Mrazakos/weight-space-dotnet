"""Export the architecture as ONNX (run once; outputs are committed).

    cd <hackathon repo>
    .venv/Scripts/python <this repo>/tools/export_onnx.py --out <this repo>/data/onnx

Writes
    possessionnet.onnx    the architecture with EVERY WEIGHT 0, and with the 24 weight tensors
                          declared as INPUTS of the graph (with zero as their default value). The
                          .NET side loads this one file and feeds the weights with the frames, so
                          a network written a second ago runs without any Python.
    expA.onnx, expB.onnx  the two hand-written networks as self-contained files, for Netron and for
                          any other ONNX runtime.
    manifest.json         initializer names and shapes, and the state-dict key each one holds.

The graph is exported WITHOUT constant folding on purpose. PyTorch stores a GRU's gates as
(reset, update, new) and ONNX wants (update, reset, new), stacked per direction; with folding off
that conversion stays in the graph as Slice / Concat nodes on the ORIGINAL tensors, so the
weight tensors keep PyTorch's names and layout and can be supplied one for one.

Why inputs and not SessionOptions.AddInitializer: ONNX Runtime's optimizer folds those Slice /
Concat nodes over the file's own (zero) values before an overriding initializer is applied, and the
override then silently does nothing for the GRU and linear weights. An initializer that is also a
graph input is not treated as a constant, so feeding the weights is the reliable way.

Inputs / outputs:   x [1, T, 48]  ->  logits [1, T, 3] (H, A, D),  gru1 [1, T, 256] (layer-1 states,
forward then backward: what the app plots as the hand-written "latch" units).
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
    """state-dict key of PossessionNet -> parameter name in Exportable (= ONNX initializer name)."""
    if key.startswith("gru."):
        layer = 1 if "_l0" in key else 2
        return "gru%d.%s" % (layer, key[4:].replace("_l1", "_l0"))
    return key


def load(module, state):
    module.load_state_dict({onnx_name(k): torch.from_numpy(v.copy()) for k, v in state.items()}, strict=True)
    return module.eval()


def export(module, path, weights_as_inputs=False):
    x = torch.zeros(1, 64, netspec.N_FEAT)
    torch.onnx.export(module, (x,), path, input_names=["x"], output_names=["logits", "gru1"],
                      dynamic_axes={"x": {1: "frames"}, "logits": {1: "frames"}, "gru1": {1: "frames"}},
                      opset_version=17, do_constant_folding=False, dynamo=False,
                      keep_initializers_as_inputs=weights_as_inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)

    zero = netspec.zero_state()
    # The exporter merges initializers that hold identical values, and twenty all-zero tensors
    # would collapse into a few. So the graph is exported with distinct random weights, which keeps
    # one initializer per tensor, and the values are set to zero afterwards.
    import onnx
    from onnx import numpy_helper
    torch.manual_seed(0)
    template = os.path.join(out, "possessionnet.onnx")
    export(Exportable().eval(), template, weights_as_inputs=True)
    model = onnx.load(template)
    wanted = {onnx_name(k): v.shape for k, v in zero.items()}
    found = {}
    for init in model.graph.initializer:
        if init.name in wanted:
            assert tuple(init.dims) == tuple(wanted[init.name]), init.name
            init.CopyFrom(numpy_helper.from_array(np.zeros(wanted[init.name], np.float32), init.name))
            found[init.name] = True
    assert len(found) == len(wanted), sorted(set(wanted) - set(found))
    onnx.checker.check_model(model)
    onnx.save(model, template)
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump({"input": "x [1, frames, 48], plus the 24 weight tensors below (optional; default 0)", "outputs": ["logits [1, frames, 3]", "gru1 [1, frames, 256]"],
                   "opset": 17,
                   "initializers": {onnx_name(k): {"state_dict_key": k, "shape": list(v.shape)}
                                    for k, v in zero.items()}}, fh, indent=1)

    networks = {"expA": "wsp/networks/expA_scratch_oneshot/hand.pt",
                "expB": "wsp/networks/expB_scratch_passes10/hand.pt"}
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 700, netspec.N_FEAT)).astype(np.float32)
    for name, ckpt in networks.items():
        state = netspec.from_checkpoint(os.path.join(ROOT, ckpt))
        module = load(Exportable(), state)
        export(module, os.path.join(out, name + ".onnx"))

        # the same input through four things that must agree
        reference = PossessionNet(netspec.N_FEAT, hidden=H)
        reference.load_state_dict(netspec.to_torch_state(state))
        with torch.no_grad():
            want = reference.eval()(torch.from_numpy(x)).numpy()
            mine, _ = module(torch.from_numpy(x))
        own = ort.InferenceSession(os.path.join(out, name + ".onnx")).run(None, {"x": x})[0]
        # the zero graph with this network's weights fed as inputs, as .NET does it
        feeds = {onnx_name(key): np.ascontiguousarray(value) for key, value in state.items()}
        feeds["x"] = x
        injected = ort.InferenceSession(os.path.join(out, "possessionnet.onnx")).run(None, feeds)[0]
        print("%s: max |difference| to PossessionNet   wrapper %.1e   %s.onnx %.1e   zero graph + fed weights %.1e" % (
            name, np.abs(mine.numpy() - want).max(), name, np.abs(own - want).max(), np.abs(injected - want).max()))

    for f in sorted(os.listdir(out)):
        print("%-22s %9d bytes" % (f, os.path.getsize(os.path.join(out, f))))


if __name__ == "__main__":
    main()
