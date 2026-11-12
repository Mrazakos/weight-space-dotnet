using WeightSpace.Core;

namespace WeightSpace.Onnx;

/// <summary>Chooses how the network runs: ONNX Runtime by default, plain C# on request.</summary>
public static class Engines
{
    /// <param name="name">"onnx" or "csharp"; when null, WEIGHTSPACE_ENGINE, else "onnx".</param>
    public static Func<Network, IPossessionEngine> Factory(string? name = null)
    {
        name ??= Environment.GetEnvironmentVariable("WEIGHTSPACE_ENGINE");
        return (name ?? "onnx").ToLowerInvariant() switch
        {
            "onnx" => n => new OnnxPossessionEngine(n),
            "csharp" or "c#" => n => new PossessionNet(n),
            _ => throw new ArgumentException($"unknown engine '{name}' (onnx or csharp)"),
        };
    }
}
