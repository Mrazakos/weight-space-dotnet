using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using WeightSpace.Core;
using WeightSpace.Onnx;

namespace WeightSpace.Web;

/// <summary>Everything the pages show, loaded once: the clip, the stored networks, the results.</summary>
public sealed class DemoData
{
    public Clip Clip { get; } = Clip.Load(Repo.Data("clip"));
    public Benchmark Benchmark { get; } = Benchmark.Load(Repo.Data("results", "benchmark.json"));

    private readonly ConcurrentDictionary<string, Network> _networks = new();
    private readonly ConcurrentDictionary<string, ClipResult> _results = new();
    private readonly ConcurrentDictionary<string, string> _replayJson = new();

    /// <summary>Stored networks, in the order the story tells them.</summary>
    public IReadOnlyList<(string Id, string Title)> NetworkList { get; } =
    [
        ("expA", "A — written in one reply"),
        .. Enumerable.Range(1, 10).Select(n => ($"expB_pass{n:00}", $"B — after practice pass {n}")),
        ("expB", "B — final network"),
    ];

    /// <summary>Experiment B's score on the practice matches after each pass, per tracking source.</summary>
    public IReadOnlyList<IReadOnlyDictionary<string, double>> PassScores { get; } =
    [
        .. Enumerable.Range(1, 10).Select(n =>
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(Repo.Data("networks", $"expB_pass{n:00}.json")));
            return (IReadOnlyDictionary<string, double>)doc.RootElement.GetProperty("dev").EnumerateObject()
                .ToDictionary(s => s.Name, s => s.Value.GetProperty("spell_f1").GetDouble());
        }),
    ];

    /// <summary>The short, slide-ready prompts in prompts/, by file name without extension.</summary>
    public string CorePrompt(string name) => File.ReadAllText(Repo.Prompts(name + ".md"));

    /// <summary>
    /// Where each tensor sits on the weight map: a rectangle per tensor on a 1600 x 640 canvas.
    /// A weight with flat index i is drawn at row i / cols, column i % cols of its rectangle.
    /// </summary>
    public string WeightMapLayoutJson { get; } = JsonSerializer.Serialize(WeightMapLayout());

    private static List<object> WeightMapLayout()
    {
        var panels = new List<object>();
        void Add(string key, string label, double x, double y, double w, double h, bool gates = false)
        {
            int[] shape = Core.Network.Shapes[key];
            int rows = shape[0], cols = shape.Skip(1).Aggregate(1, (a, b) => a * b);
            panels.Add(new { key, label, x, y, w, h, rows, cols, gates });
        }
        const double top = 56, h1 = 210, top2 = 356, h2 = 250;
        Add("inp.0.weight", "input 128 × 48", 20, top, 90, h1);
        Add("inp.0.bias", "", 114, top, 8, h1);
        Add("conv.weight", "convolution  128 × 128 × 9 frames", 150, top, 760, h1);
        Add("conv.bias", "", 914, top, 8, h1);
        Add("head.0.weight", "head  128 × 384", 950, top, 430, h1);
        Add("head.0.bias", "", 1384, top, 8, h1);
        Add("head.2.weight", "outputs H, A, D  3 × 128", 1420, top, 160, 36);
        Add("head.2.bias", "", 1584, top, 8, 36);
        double x = 70;
        foreach (var (layer, name, wi) in new[] { (0, "GRU 1", 110.0), (1, "GRU 2", 220.0) })
            foreach (var (suffix, dir) in new[] { ("", "forward"), ("_reverse", "backward") })
            {
                Add($"gru.weight_ih_l{layer}{suffix}", $"{name} {dir}", x, top2, wi, h2, gates: true);
                Add($"gru.weight_hh_l{layer}{suffix}", "own state", x + wi + 12, top2, 110, h2, gates: true);
                Add($"gru.bias_ih_l{layer}{suffix}", "", x + wi + 126, top2, 8, h2, gates: true);
                Add($"gru.bias_hh_l{layer}{suffix}", "", x + wi + 138, top2, 8, h2, gates: true);
                x += wi + 186;
            }
        return panels;
    }

    public Network Network(string id) => _networks.GetOrAdd(id, key =>
    {
        if (!NetworkList.Any(n => n.Id == key)) throw new KeyNotFoundException(key);
        return Core.Network.Load(Repo.Data("networks", key + ".json"));
    });

    public ClipResult Result(string id) => _results.GetOrAdd(id, key => Pipeline.Run(Network(key), Clip, Engines.Factory()));

    /// <summary>A network written during this session (the Write page) can be replayed too.</summary>
    public void Register(string id, Network network)
    {
        _networks[id] = network;
        _results.TryRemove(id, out _);
        _replayJson.TryRemove(id, out _);
    }

    public bool Has(string id) => _networks.ContainsKey(id) || NetworkList.Any(n => n.Id == id);

    /// <summary>The JSON the replay canvas draws from: positions, three label tracks, probabilities, traces.</summary>
    public string ReplayJson(string id) => _replayJson.GetOrAdd(id, key =>
    {
        var net = Network(key);
        var result = Result(key);
        int m = Clip.Margin, n = Clip.Frames;
        static int[] Centre(byte[] labels, int m, int n) => [.. labels.Skip(m).Take(n).Select(b => (int)b)];
        int trainedRight = Enumerable.Range(m, n).Count(t => Clip.Trained[t] == Clip.Truth[t]);
        string head = JsonSerializer.Serialize(new
        {
            id = key,
            name = net.Name,
            description = net.Description,
            nonzero = net.NonZero,
            total = Core.Network.TotalWeights,
            fps = Clip.Fps,
            frames = n,
            pitch = new[] { Clip.PitchLength, Clip.PitchWidth },
            source = Clip.Source,
            homeAttacksPositiveX = Clip.HomeAttacksPositiveX,
            inferenceMs = Math.Round(result.Elapsed.TotalMilliseconds),
            labels = new
            {
                truth = Centre(Clip.Truth, m, n), net = Centre(result.Labels, m, n), trained = Centre(Clip.Trained, m, n),
            },
            probs = result.Probabilities.Skip(m * 3).Take(n * 3).Select(p => Math.Round(p, 3)),
            traces = result.Traces.Select(t => new
            {
                label = t.Probe.Label, unit = $"{t.Probe.Layer}[{t.Probe.Unit}]", min = t.Probe.Min, max = t.Probe.Max,
                values = t.Trace.Skip(m).Take(n).Select(v => Math.Round(v, 3)),
            }),
            accuracy = new { net = result.Accuracy(Clip), trained = trainedRight / (double)n },
            possessions = new
            {
                truth = ClipResult.Possessions(Clip.Truth.AsSpan(m, n)),
                net = ClipResult.Possessions(result.Labels.AsSpan(m, n)),
                trained = ClipResult.Possessions(Clip.Trained.AsSpan(m, n)),
            },
        });
        // splice the positions in as they are stored, without parsing them again
        return new StringBuilder(head, head.Length + Clip.PositionsJson.Length + 20)
            .Remove(head.Length - 1, 1).Append(",\"positions\":").Append(Clip.PositionsJson).Append('}').ToString();
    });
}

/// <summary>The held-out benchmark numbers of the experiment (data/results/benchmark.json).</summary>
public sealed record Benchmark(IReadOnlyList<Benchmark.Row> Rows, IReadOnlyDictionary<string, string> SourceNames,
    IReadOnlyDictionary<string, long> Weights, IReadOnlyList<(int Pass, double SpellF1, double MacroF1)> PassHistory)
{
    public sealed record Row(string Label, string Kind, IReadOnlyDictionary<string, Score> Sources);
    public sealed record Score(double SpellF1, double MacroF1, int Predicted, int True);
    public static readonly string[] SourceOrder = ["DFL", "SC", "MET"];

    public static Benchmark Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var r = doc.RootElement;
        var rows = r.GetProperty("rows").EnumerateArray().Select(row => new Row(
            row.GetProperty("label").GetString()!, row.GetProperty("kind").GetString()!,
            row.GetProperty("sources").EnumerateObject().ToDictionary(s => s.Name, s => new Score(
                s.Value.GetProperty("spell_f1").GetDouble(), s.Value.GetProperty("macro_f1").GetDouble(),
                s.Value.GetProperty("n_pred_spells").GetInt32(), s.Value.GetProperty("n_gt_spells").GetInt32())))).ToList();
        return new Benchmark(rows,
            r.GetProperty("source_names").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!),
            r.GetProperty("weights").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt64()),
            [.. r.GetProperty("expB_dev_history").EnumerateArray().Select(h => (h[0].GetInt32(), h[1].GetDouble(), h[2].GetDouble()))]);
    }
}
