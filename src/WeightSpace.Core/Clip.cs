using System.Runtime.InteropServices;
using System.Text.Json;

namespace WeightSpace.Core;

/// <summary>
/// 100 seconds of a held-out half: what the tracking saw, the reference labels, and the network's
/// input for that stretch (four symmetric views, with context on each side). The features come
/// from the Python feature code of the experiment; this side only reads them.
/// </summary>
public sealed class Clip
{
    public string Source { get; init; } = "";
    public int Fps { get; init; }
    public int Frames { get; init; }
    public int Margin { get; init; }
    public int StartFrame { get; init; }
    public double PitchLength { get; init; }
    public double PitchWidth { get; init; }
    public bool HomeAttacksPositiveX { get; init; }
    public double SwitchHA { get; init; }
    public double SwitchAlive { get; init; }
    public bool[] ViewSwapped { get; init; } = [];
    /// <summary>Reference labels over frames + 2 * margin. 0 H, 1 A, 2 D.</summary>
    public byte[] Truth { get; init; } = [];
    /// <summary>Labels of the SGD-trained network over the same stretch, precomputed.</summary>
    public byte[] Trained { get; init; } = [];
    /// <summary>The network input, one array per view, each [(frames + 2 * margin) * 48].</summary>
    public float[][] Views { get; init; } = [];
    /// <summary>The raw positions JSON of the central frames, passed to the browser as is.</summary>
    public string PositionsJson { get; init; } = "";

    public int TotalFrames => Frames + 2 * Margin;

    public static Clip Load(string folder)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(folder, "clip.json")));
        var r = doc.RootElement;
        int frames = r.GetProperty("frames").GetInt32(), margin = r.GetProperty("margin").GetInt32();
        int total = frames + 2 * margin, nViews = r.GetProperty("views").GetInt32();
        int nFeat = r.GetProperty("features").GetInt32();

        var all = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(folder, "features.bin"))).ToArray();
        if (all.Length != nViews * total * nFeat)
            throw new InvalidDataException($"features.bin has {all.Length} values, expected {nViews * total * nFeat}");
        var views = new float[nViews][];
        for (int v = 0; v < nViews; v++) views[v] = all.AsSpan(v * total * nFeat, total * nFeat).ToArray();

        static byte[] Labels(JsonElement e) => [.. e.EnumerateArray().Select(x => (byte)x.GetInt32())];
        var labels = r.GetProperty("labels");
        var positions = new Dictionary<string, JsonElement>
        {
            ["ball"] = r.GetProperty("ball"), ["home"] = r.GetProperty("home"), ["away"] = r.GetProperty("away"),
        };
        return new Clip
        {
            Source = r.GetProperty("source").GetString()!,
            Fps = r.GetProperty("fps").GetInt32(),
            Frames = frames,
            Margin = margin,
            StartFrame = r.GetProperty("start_frame").GetInt32(),
            PitchLength = r.GetProperty("pitch")[0].GetDouble(),
            PitchWidth = r.GetProperty("pitch")[1].GetDouble(),
            HomeAttacksPositiveX = r.GetProperty("home_attacks_positive_x").GetBoolean(),
            SwitchHA = r.GetProperty("decoder").GetProperty("switch_ha").GetDouble(),
            SwitchAlive = r.GetProperty("decoder").GetProperty("switch_alive").GetDouble(),
            ViewSwapped = [.. r.GetProperty("view_swapped").EnumerateArray().Select(x => x.GetBoolean())],
            Truth = Labels(labels.GetProperty("truth")),
            Trained = Labels(labels.GetProperty("trained")),
            Views = views,
            PositionsJson = JsonSerializer.Serialize(positions),
        };
    }
}

/// <summary>What a network says about a clip, over frames + 2 * margin.</summary>
public sealed class ClipResult
{
    /// <summary>[T * 3] probabilities, averaged over the four views.</summary>
    public float[] Probabilities { get; init; } = [];
    /// <summary>Decoded labels.</summary>
    public byte[] Labels { get; init; } = [];
    /// <summary>The probed recurrent units of the original view.</summary>
    public List<(Probe Probe, float[] Trace)> Traces { get; init; } = [];
    public TimeSpan Elapsed { get; init; }
    /// <summary>What ran the network: "ONNX Runtime" or "plain C#".</summary>
    public string Engine { get; init; } = "";

    /// <summary>Share of the central frames whose label equals the reference.</summary>
    public double Accuracy(Clip clip) =>
        Enumerable.Range(clip.Margin, clip.Frames).Count(t => Labels[t] == clip.Truth[t]) / (double)clip.Frames;

    /// <summary>Number of possessions (maximal runs of H or of A) in the central frames.</summary>
    public static int Possessions(ReadOnlySpan<byte> labels)
    {
        int n = 0;
        for (int t = 0; t < labels.Length; t++)
            if (labels[t] < 2 && (t == 0 || labels[t - 1] != labels[t])) n++;
        return n;
    }
}

/// <summary>The inference pipeline of the experiment: four views, averaged, then the decoder.</summary>
public static class Pipeline
{
    /// <param name="engine">How to run the network; plain C# when omitted.</param>
    public static ClipResult Run(Network network, Clip clip, Func<Network, IPossessionEngine>? engine = null)
    {
        using var net = engine is null ? new PossessionNet(network) : engine(network);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        int T = clip.TotalFrames;
        var results = new ForwardResult[clip.Views.Length];
        Parallel.For(0, clip.Views.Length, v => results[v] = net.Forward(clip.Views[v]));

        var probs = new float[T * 3];
        for (int v = 0; v < results.Length; v++)
        {
            var logits = results[v].Logits;
            bool swapped = clip.ViewSwapped[v];                  // roles exchanged: swap H and A back
            for (int t = 0; t < T; t++)
            {
                float h = logits[t * 3], a = logits[t * 3 + 1], d = logits[t * 3 + 2];
                float max = MathF.Max(h, MathF.Max(a, d));
                float eh = MathF.Exp(h - max), ea = MathF.Exp(a - max), ed = MathF.Exp(d - max);
                float sum = eh + ea + ed;
                probs[t * 3] += (swapped ? ea : eh) / sum / results.Length;
                probs[t * 3 + 1] += (swapped ? eh : ea) / sum / results.Length;
                probs[t * 3 + 2] += ed / sum / results.Length;
            }
        }
        var labels = Decoder.Viterbi(probs, clip.SwitchHA, clip.SwitchAlive);
        return new ClipResult
        {
            Probabilities = probs,
            Labels = labels,
            Traces = [.. network.Probes.Select(p => (p, results[0].Trace(p)))],
            Elapsed = clock.Elapsed,
            Engine = net.Name,
        };
    }
}
