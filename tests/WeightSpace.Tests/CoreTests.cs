using System.Runtime.InteropServices;
using WeightSpace.Core;
using WeightSpace.Onnx;

namespace WeightSpace.Tests;

public class NetworkTests
{
    [Fact]
    public void The_architecture_has_698115_weights()
    {
        Assert.Equal(698_115, Network.TotalWeights);
        Assert.Equal(0, new Network().NonZero);
    }

    [Theory]
    [InlineData("expA", 396)]
    [InlineData("expB", 179)]
    public void Stored_networks_load_with_the_expected_number_of_weights(string name, int nonzero)
    {
        var net = Network.Load(Repo.Data("networks", name + ".json"));
        Assert.Equal(nonzero, net.NonZero);
        Assert.NotEmpty(net.Probes);
    }

    [Fact]
    public void A_zero_network_outputs_zero()
    {
        var x = new float[50 * Network.Features];
        Random.Shared.NextBytes(MemoryMarshal.AsBytes(x.AsSpan()));
        var result = new PossessionNet(new Network()).Forward(x.Select(v => v % 3f).ToArray());
        Assert.All(result.Logits, v => Assert.Equal(0f, v));
    }
}

/// <summary>The C# forward pass and decoder against PyTorch on the same input.</summary>
public class ParityTests
{
    private static readonly Clip Clip = Clip.Load(Repo.Data("clip"));

    [Theory]
    [InlineData("expA")]
    [InlineData("expB")]
    public void Probabilities_and_labels_match_the_PyTorch_reference(string name)
    {
        var result = Pipeline.Run(Network.Load(Repo.Data("networks", name + ".json")), Clip);

        byte[] raw = File.ReadAllBytes(Repo.Data("reference", name + ".bin"));
        int T = Clip.TotalFrames;
        var probs = MemoryMarshal.Cast<byte, float>(raw.AsSpan(0, T * 3 * sizeof(float)));
        var labels = raw.AsSpan(T * 3 * sizeof(float), T);

        float worst = 0f;
        for (int i = 0; i < probs.Length; i++) worst = MathF.Max(worst, MathF.Abs(probs[i] - result.Probabilities[i]));
        Assert.True(worst < 1e-3f, $"largest probability difference {worst}");
        Assert.Equal(labels.ToArray(), result.Labels);
    }

    [Theory]
    [InlineData("expA")]
    [InlineData("expB")]
    public void ONNX_Runtime_matches_the_PyTorch_reference(string name)
    {
        var result = Pipeline.Run(Network.Load(Repo.Data("networks", name + ".json")), Clip, Engines.Factory("onnx"));
        Assert.Equal("ONNX Runtime", result.Engine);

        byte[] raw = File.ReadAllBytes(Repo.Data("reference", name + ".bin"));
        int T = Clip.TotalFrames;
        var probs = MemoryMarshal.Cast<byte, float>(raw.AsSpan(0, T * 3 * sizeof(float)));
        float worst = 0f;
        for (int i = 0; i < probs.Length; i++) worst = MathF.Max(worst, MathF.Abs(probs[i] - result.Probabilities[i]));
        Assert.True(worst < 1e-3f, $"largest probability difference {worst}");
        Assert.Equal(raw.AsSpan(T * 3 * sizeof(float), T).ToArray(), result.Labels);
    }

    [Fact]
    public void ONNX_and_C_sharp_agree_on_logits_and_layer_1_states()
    {
        var net = Network.Load(Repo.Data("networks", "expB.json"));
        var x = Clip.Views[0];
        using var onnx = new OnnxPossessionEngine(net);
        var a = onnx.Forward(x);
        var b = new PossessionNet(net).Forward(x);
        Assert.True(a.Logits.Zip(b.Logits).Max(p => MathF.Abs(p.First - p.Second)) < 1e-3f);
        Assert.True(a.Gru1F.Zip(b.Gru1F).Max(p => MathF.Abs(p.First - p.Second)) < 1e-3f);
        Assert.True(a.Gru1B.Zip(b.Gru1B).Max(p => MathF.Abs(p.First - p.Second)) < 1e-3f);
    }

    [Fact]
    public void The_trained_labels_and_the_truth_cover_the_clip()
    {
        Assert.Equal(Clip.TotalFrames, Clip.Truth.Length);
        Assert.Equal(Clip.TotalFrames, Clip.Trained.Length);
        Assert.Equal(4, Clip.Views.Length);
    }
}

public class DecoderTests
{
    [Fact]
    public void A_single_frame_of_evidence_does_not_pay_for_a_switch()
    {
        var probs = new float[200 * 3];
        for (int t = 0; t < 200; t++) (probs[t * 3], probs[t * 3 + 1], probs[t * 3 + 2]) = (0.90f, 0.05f, 0.05f);
        (probs[100 * 3], probs[100 * 3 + 1]) = (0.05f, 0.90f);                     // one frame says "away"
        Assert.All(Decoder.Viterbi(probs), label => Assert.Equal(0, label));
    }

    [Fact]
    public void Sustained_evidence_does()
    {
        var probs = new float[200 * 3];
        for (int t = 0; t < 200; t++)
            (probs[t * 3], probs[t * 3 + 1], probs[t * 3 + 2]) = t is >= 80 and < 140 ? (0.02f, 0.96f, 0.02f) : (0.96f, 0.02f, 0.02f);
        byte[] labels = Decoder.Viterbi(probs);
        Assert.Equal(0, labels[10]);
        Assert.Equal(1, labels[110]);
        Assert.Equal(0, labels[190]);
    }
}

public class EditsTests
{
    private static string Block(string body) => $"design text\n\n```edits\n{body}\n```\n";

    [Fact]
    public void Scalars_ranges_lists_and_in_place_operators()
    {
        var net = new Network();
        var report = Edits.Apply(net, Block("""
            inp.W[3, 47] = -2.5   # one weight
            conv.W[10, 3, :] = 0.5
            head2.W[0, 0:3] = [4, -4, 0.5]
            gru1f.bhz[0:8] += 2
            gru1f.bhz[0:8:2] *= 3
            inp.b[-1] -= 1
            head2.W[0:2, 5:7] = [[1, -1],
                                 [-1, 1]]
            """));
        Assert.Empty(report.Errors);
        Assert.Equal(7, report.Applied);
        Assert.Equal(-2.5f, net["inp.0.weight"].Data[3 * 48 + 47]);
        Assert.All(Enumerable.Range(0, 9), k => Assert.Equal(0.5f, net["conv.weight"].Data[(10 * 128 + 3) * 9 + k]));
        Assert.Equal([4f, -4f, 0.5f], net["head.2.weight"].Data[..3]);
        Assert.Equal([6f, 2f, 6f, 2f, 6f, 2f, 6f, 2f], net["gru.bias_hh_l0"].Data[128..136]);   // z gate = rows 128..255
        Assert.Equal(-1f, net["inp.0.bias"].Data[127]);
        Assert.Equal([1f, -1f], net["head.2.weight"].Data[5..7]);
        Assert.Equal([-1f, 1f], net["head.2.weight"].Data[(128 + 5)..(128 + 7)]);
    }

    [Fact]
    public void Gate_and_head_aliases_point_at_the_right_rows_and_columns()
    {
        var net = new Network();
        Edits.Apply(net, Block("gru1f.Wiz[5, 7] = 1\ngru2b.Whn[2, 3] = 2\ngru2f.bir[4] = 3\nhead1.Wb[1, 6] = 4"));
        Assert.Equal(1f, net["gru.weight_ih_l0"].Data[(128 + 5) * 128 + 7]);
        Assert.Equal(2f, net["gru.weight_hh_l1_reverse"].Data[(256 + 2) * 128 + 3]);
        Assert.Equal(3f, net["gru.bias_ih_l1"].Data[4]);
        Assert.Equal(4f, net["head.0.weight"].Data[1 * 384 + 128 + 6]);
        Assert.Equal(4, net.NonZero);
    }

    [Fact]
    public void Bad_statements_are_skipped_and_reported_and_nothing_is_executed()
    {
        var net = new Network();
        var report = Edits.Apply(net, Block("""
            nope.W[0] = 1
            inp.W[200, 0] = 1
            inp.W[0, 0:99] = 1
            inp.W[0, 0:3] = [1, 2]
            inp.W[0, 0] = __import__('os')
            inp.W[0, 0] = 1e9
            inp.W[1+1, 0] = 1
            inp.W[0, 0:2] = [[1, 2], 3]
            inp.W[0, 1] = 7
            """));
        Assert.Equal(9, report.Statements);
        Assert.Equal(1, report.Applied);
        Assert.Equal(1, net.NonZero);
        Assert.Equal(7f, net["inp.0.weight"].Data[1]);
    }

    [Fact]
    public void Only_the_last_block_counts()
    {
        var net = new Network();
        Edits.Apply(net, Block("inp.b[0] = 1") + "second thoughts\n" + Block("inp.b[1] = 2"));
        Assert.Equal([0f, 2f], net["inp.0.bias"].Data[..2]);
    }
}

/// <summary>The recorded replies of the experiment, parsed here, give the stored networks.</summary>
public class ReplayTests
{
    private static void AssertSame(Network expected, Network actual)
    {
        foreach (var (key, t) in expected.Tensors)
        {
            int at = Enumerable.Range(0, t.Data.Length).FirstOrDefault(i => t.Data[i] != actual[key].Data[i], -1);
            Assert.True(at < 0, at < 0 ? "" : $"tensor {key}[{at}]: expected {t.Data[at]:R}, got {actual[key].Data[at]:R}");
        }
    }

    [Fact]
    public void The_one_shot_reply_parses_to_the_network_of_experiment_A()
    {
        var net = new Network();
        var report = Edits.Apply(net, Repo.Reply("expA_scratch_oneshot", 0));
        Assert.Equal(140, report.Statements);
        Assert.Empty(report.Errors);
        AssertSame(Network.Load(Repo.Data("networks", "expA.json")), net);
    }

    /// <summary>
    /// Replaying B's eleven replies, each applied once, gives the stored networks except for one
    /// weight. The experiment's run was interrupted after reply 3 and its driver then applied that
    /// reply a second time, so its one in-place statement, `head1.b[1] += 0.3`, counted twice.
    /// The networks in data/networks are the ones that were actually scored (and that the model
    /// received feedback on from pass 4), so they carry -1.525 where the replies say -1.825.
    /// This test pins that difference down to exactly that weight. See experiment/JOURNAL.md.
    /// </summary>
    [Fact]
    public void Eleven_replies_parse_to_every_pass_of_experiment_B_but_for_one_doubled_increment()
    {
        var net = new Network();
        var passes = new Dictionary<int, Network>();
        for (int turn = 0; turn <= 10; turn++)
        {
            var report = Edits.Apply(net, Repo.Reply("expB_scratch_passes10", turn),
                reset: () => new Network(), restore: n => passes.GetValueOrDefault(n));
            Assert.Empty(report.Errors);
            if (turn < 10)
            {
                passes[turn + 1] = net.Clone();
                AssertSame(Network.Load(Repo.Data("networks", $"expB_pass{turn + 1:00}.json")), AsScored(net, turn + 1));
            }
        }
        Assert.Equal(-1.825f, net["head.0.bias"].Data[1]);
        AssertSame(Network.Load(Repo.Data("networks", "expB.json")), AsScored(net, 10));

        static Network AsScored(Network written, int pass)
        {
            var scored = written.Clone();
            if (pass >= 4) scored["head.0.bias"].Data[1] = (float)((double)scored["head.0.bias"].Data[1] + 0.3);
            return scored;
        }
    }
}
