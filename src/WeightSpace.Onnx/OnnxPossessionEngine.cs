using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using WeightSpace.Core;

namespace WeightSpace.Onnx;

/// <summary>
/// PossessionNet run by ONNX Runtime. The graph is the exported template
/// (data/onnx/possessionnet.onnx); the weights of the given <see cref="Network"/> replace the
/// template's initializers, so any network, including one written a moment ago, runs through ONNX.
/// </summary>
public sealed class OnnxPossessionEngine : IPossessionEngine
{
    public string Name => "ONNX Runtime";

    private readonly InferenceSession _session;

    public OnnxPossessionEngine(Network net, string? templatePath = null)
    {
        // ONNX Runtime will not override constant initializers, so the weights are written into a
        // copy of the template's bytes (every initializer there is one raw float32 blob).
        byte[] model = File.ReadAllBytes(templatePath ?? Repo.Data("onnx", "possessionnet.onnx"));
        var blobs = OnnxFile.RawInitializers(model);
        foreach (var (key, tensor) in net.Tensors)
        {
            var (offset, length) = blobs[OnnxName(key)];
            if (length != tensor.Data.Length * sizeof(float))
                throw new InvalidDataException($"{key}: template holds {length} bytes, network has {tensor.Data.Length} floats");
            System.Buffer.BlockCopy(tensor.Data, 0, model, offset, length);
        }
        _session = new InferenceSession(model);
    }

    /// <summary>State-dict key to ONNX initializer name: the export splits the two GRU layers into gru1 and gru2.</summary>
    public static string OnnxName(string key) =>
        key.Replace("_l1", "_l0").Replace("gru.", key.Contains("_l1") ? "gru2." : "gru1.");

    public ForwardResult Forward(ReadOnlySpan<float> x)
    {
        const int F = Network.Features, H = Network.Hidden;
        int T = x.Length / F;
        var input = new DenseTensor<float>(x.ToArray(), [1, T, F]);
        using var outputs = _session.Run([NamedOnnxValue.CreateFromTensor("x", input)]);
        var logits = outputs.First(o => o.Name == "logits").AsTensor<float>().ToArray();
        var gru1 = outputs.First(o => o.Name == "gru1").AsTensor<float>().ToArray();

        var result = new ForwardResult(T);
        logits.CopyTo(result.Logits, 0);
        var (f, b) = (new float[T * H], new float[T * H]);
        for (int t = 0; t < T; t++)
        {
            Array.Copy(gru1, t * 2 * H, f, t * H, H);
            Array.Copy(gru1, t * 2 * H + H, b, t * H, H);
        }
        result.Gru1F = f;
        result.Gru1B = b;
        return result;
    }

    public void Dispose() => _session.Dispose();
}
