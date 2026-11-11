using Microsoft.Extensions.AI;
using WeightSpace.AI;
using WeightSpace.Core;

namespace WeightSpace.Tests;

public class WriterTests
{
    [Theory]
    [InlineData(7)]         // chunks that cut lines in odd places
    [InlineData(4096)]
    public async Task Streaming_the_recorded_reply_through_IChatClient_writes_network_A(int chunk)
    {
        IChatClient client = new ReplayChatClient(Repo.Reply("expA_scratch_oneshot", 0), chunk);
        var net = new Network();
        var edits = new List<WriterEvent.Edit>();
        WriterEvent.Done? done = null;
        await foreach (var e in new WeightWriter(client).WriteAsync(Repo.Sent("expA_scratch_oneshot", 0), net))
        {
            if (e is WriterEvent.Edit edit) edits.Add(edit);
            if (e is WriterEvent.Done d) done = d;
        }
        Assert.NotNull(done);
        Assert.Equal((140, 140, 0), (done.Statements, done.Applied, done.Refused));
        Assert.Equal(140, edits.Count);
        Assert.Equal(396, edits[^1].NonZeroWeights);

        var expected = Network.Load(Repo.Data("networks", "expA.json"));
        foreach (var (key, t) in expected.Tensors) Assert.True(t.Data.AsSpan().SequenceEqual(net[key].Data), key);
    }

    [Fact]
    public async Task A_reply_without_an_edits_block_changes_nothing()
    {
        var net = new Network();
        await foreach (var _ in new WeightWriter(new ReplayChatClient("I would rather not.")).WriteAsync("write weights", net)) { }
        Assert.Equal(0, net.NonZero);
    }

    [Fact]
    public async Task Only_the_last_edits_block_of_a_streamed_reply_counts()
    {
        var net = new Network();
        string reply = "```edits\ninp.b[0] = 1\n```\nsecond thoughts\n```edits\ninp.b[1] = 2\n```";
        await foreach (var _ in new WeightWriter(new ReplayChatClient(reply, 5)).WriteAsync("write weights", net)) { }
        Assert.Equal([0f, 2f], net["inp.0.bias"].Data[..2]);
    }
}
