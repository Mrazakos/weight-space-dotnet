# ONNX in this project: presentation notes

## What is ONNX?
**ONNX (Open Neural Network Exchange)** is an open file format for machine-learning models. A
`.onnx` file holds two things:

1. the **graph**: the operations (matrix multiply, GELU, convolution, GRU...) and how they connect;
2. the **weights** (called *initializers*): the numbers those operations use.

It is a standard, so a model trained in one framework (PyTorch, TensorFlow, scikit-learn) can be
run somewhere else, in another language, without the original framework.
It started at Microsoft and Facebook in 2017 and is now a Linux Foundation project.

**ONNX Runtime** is the engine that *runs* `.onnx` files. It is a separate piece of software, open
source (MIT), made and maintained by Microsoft, with bindings for C#, Python, C++, Java, JavaScript.

Analogy: ONNX is like a PDF for models; ONNX Runtime is the PDF reader.

## Why is it good?
- **Train anywhere, run anywhere.** The model is made in Python/PyTorch, and the app is in C#. No Python in production.
- **No framework dependency.** The .NET app needs one NuGet package, not PyTorch (gigabytes).
- **Fast.** ONNX Runtime optimizes the graph (fuses operations, uses SIMD, can use GPU / NPU through "execution providers").
- **Portable.** Same file runs on Windows, Linux, macOS, mobile, browser (WebAssembly).
- **Inspectable.** Open the file in [Netron](https://netron.app) and see the architecture.
- **Stable and supported.** Microsoft uses it in Windows, Office, Bing; it is the standard way to run models in .NET.

## When is it useful?
- A data-science team trains in Python and a .NET team ships the product.
- You want inference without installing a big ML framework (desktop apps, servers, edge devices).
- You need the same model on several platforms or languages.
- You want hardware acceleration without writing GPU code.

It is **not** the right tool for *training* (it is for inference), and for a tiny model it can be
slower than hand-written code, as we saw below.

## Are we using the official library?
**Yes.** The NuGet package is **`Microsoft.ML.OnnxRuntime`** (version 1.22.0), published by
Microsoft: the official ONNX Runtime for .NET. We only call its public API (`InferenceSession`,
`DenseTensor`, `NamedOnnxValue`). The only thing we wrote ourselves is a ~60-line protobuf reader
(`OnnxFile.cs`), used to put weights into the file (see below).

## How it works in this project
```
PyTorch model ──tools/export_onnx.py──▶ possessionnet.onnx (graph, weights = 0)
                                                │
   Network (698,115 weights, hand-written) ─────┤  OnnxPossessionEngine writes the weights
                                                ▼  into a copy of the file's bytes
                                       InferenceSession (ONNX Runtime)
                                                ▼
                         logits [1, T, 3]  +  gru1 states [1, T, 256]
```
- Code: `src/WeightSpace.Onnx/` (`OnnxPossessionEngine`, `OnnxFile`, `Engines`).
- The model takes `x [1, frames, 48]` and returns `logits [1, frames, 3]` (home / away / dead ball) and `gru1 [1, frames, 256]`.
- The engine plugs into the existing `IPossessionEngine` interface, so the rest of the app did not change.
- **Neat part:** the weights are not trained. Claude *writes* them as edits, and we load them into
  the same ONNX graph. So ONNX is the "body", and the written weights are the "brain".
- Why we patch bytes: ONNX Runtime's `AddInitializer` does not override weights stored as constants in the
  file (we tried: the output stayed wrong). Writing into the file bytes works.

## Results (this machine, 100 s clip, 4 views x 4,500 frames)
| Engine | Time | Possessions found | Match with PyTorch |
|---|---|---|---|
| ONNX Runtime | ~250-390 ms | 10 | probabilities within 1e-3 (tested) |
| Plain C# (hand-written) | ~135 ms | 10 | within 1e-3 (tested) |

Honest takeaway: both engines agree. The plain C# one is faster here because the networks are
>99.9% zeros (179 of 698,115 weights set) and it skips zero units; ONNX Runtime does the full dense
math. ONNX's advantage is portability and generality, not speed on this special case.

## How to run it
```
dotnet test                                                        # 22 tests, ONNX vs PyTorch reference
dotnet run --project src/WeightSpace.Cli -- run expB               # ONNX Runtime (default)
dotnet run --project src/WeightSpace.Cli -- run expB --engine csharp
dotnet run --project src/WeightSpace.Web                           # app; WEIGHTSPACE_ENGINE=csharp to switch
```

## Possible slide outline
1. The problem: model in Python, app in .NET
2. What ONNX / ONNX Runtime are (PDF analogy)
3. Why: portable, fast, no framework, inspectable (show Netron screenshot of `expB.onnx`)
4. Our pipeline diagram
5. Proof: parity with PyTorch + same labels on both engines
6. Honest benchmark and when *not* to use it
