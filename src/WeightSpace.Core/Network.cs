using System.Text.Json;

namespace WeightSpace.Core;

/// <summary>One named tensor of the network: a shape and row-major float32 data.</summary>
public sealed class Tensor(int[] shape)
{
    public int[] Shape { get; } = shape;
    public float[] Data { get; } = new float[shape.Aggregate(1, (a, b) => a * b)];
    public int NonZero => Data.Count(v => v != 0f);
}

/// <summary>
/// The weights of PossessionNet(48, 128): 698,115 numbers in 24 tensors, keyed by their PyTorch
/// state-dict names. A hand-written network is this object with almost everything left at zero.
/// </summary>
public sealed class Network
{
    public const int Features = 48, Hidden = 128, Kernel = 9;

    public string Name { get; set; } = "zero";
    public string Description { get; set; } = "every weight 0";
    public List<Probe> Probes { get; set; } = [];
    public Dictionary<string, Tensor> Tensors { get; } = [];

    /// <summary>State-dict key -> shape, in PyTorch's (out, in) orientation.</summary>
    public static IReadOnlyDictionary<string, int[]> Shapes { get; } = BuildShapes();

    public static int TotalWeights => Shapes.Values.Sum(s => s.Aggregate(1, (a, b) => a * b));
    public int NonZero => Tensors.Values.Sum(t => t.NonZero);

    public Network()
    {
        foreach (var (key, shape) in Shapes) Tensors[key] = new Tensor(shape);
    }

    public Tensor this[string key] => Tensors[key];

    private static Dictionary<string, int[]> BuildShapes()
    {
        const int h = Hidden;
        var s = new Dictionary<string, int[]>
        {
            ["inp.0.weight"] = [h, Features], ["inp.0.bias"] = [h],
            ["conv.weight"] = [h, h, Kernel], ["conv.bias"] = [h],
            ["head.0.weight"] = [h, 3 * h], ["head.0.bias"] = [h],
            ["head.2.weight"] = [3, h], ["head.2.bias"] = [3],
        };
        foreach (var (layer, nIn) in new[] { (0, h), (1, 2 * h) })
            foreach (var suffix in new[] { "", "_reverse" })
            {
                s[$"gru.weight_ih_l{layer}{suffix}"] = [3 * h, nIn];
                s[$"gru.weight_hh_l{layer}{suffix}"] = [3 * h, h];
                s[$"gru.bias_ih_l{layer}{suffix}"] = [3 * h];
                s[$"gru.bias_hh_l{layer}{suffix}"] = [3 * h];
            }
        return s;
    }

    public Network Clone()
    {
        var copy = new Network { Name = Name, Description = Description, Probes = [.. Probes] };
        foreach (var (key, t) in Tensors) t.Data.CopyTo(copy.Tensors[key].Data, 0);
        return copy;
    }

    /// <summary>Load the sparse JSON written by tools/export_demo_data.py.</summary>
    public static Network Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = doc.RootElement;
        var net = new Network
        {
            Name = root.GetProperty("name").GetString()!,
            Description = root.GetProperty("description").GetString()!,
        };
        if (root.TryGetProperty("probes", out var probes))
            foreach (var p in probes.EnumerateArray())
                net.Probes.Add(new Probe(p.GetProperty("label").GetString()!, p.GetProperty("tensor").GetString()!,
                    p.GetProperty("unit").GetInt32(), p.GetProperty("min").GetSingle(), p.GetProperty("max").GetSingle()));
        foreach (var t in root.GetProperty("tensors").EnumerateObject())
        {
            var data = net.Tensors[t.Name].Data;
            var idx = t.Value.GetProperty("i").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var val = t.Value.GetProperty("v").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            for (int n = 0; n < idx.Length; n++) data[idx[n]] = val[n];
        }
        return net;
    }

    /// <summary>Write the same sparse JSON: only the weights that are not zero.</summary>
    public void Save(string path)
    {
        var tensors = Tensors.ToDictionary(kv => kv.Key, kv =>
        {
            var idx = Enumerable.Range(0, kv.Value.Data.Length).Where(i => kv.Value.Data[i] != 0f).ToArray();
            return new { shape = kv.Value.Shape, i = idx, v = idx.Select(i => kv.Value.Data[i]).ToArray() };
        });
        var probes = Probes.Select(p => new { label = p.Label, tensor = p.Layer, unit = p.Unit, min = p.Min, max = p.Max });
        File.WriteAllText(path, JsonSerializer.Serialize(new
        {
            name = Name, description = Description, nonzero = NonZero, total = TotalWeights, probes, tensors,
        }));
    }
}

/// <summary>A recurrent unit worth watching: which layer-1 GRU state to plot, and how to scale it.</summary>
public sealed record Probe(string Label, string Layer, int Unit, float Min, float Max);
