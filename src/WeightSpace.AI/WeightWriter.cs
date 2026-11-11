using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.Extensions.AI;
using WeightSpace.Core;

namespace WeightSpace.AI;

/// <summary>Something that happened while a model was writing a network.</summary>
public abstract record WriterEvent
{
    /// <summary>A piece of the reply arrived.</summary>
    public sealed record Text(string Delta) : WriterEvent;
    /// <summary>The model is reasoning; this many reasoning tokens so far.</summary>
    public sealed record Thinking(int Tokens) : WriterEvent;
    /// <summary>One statement of the edits block was applied to the network (or refused).</summary>
    public sealed record Edit(EditOutcome Outcome, int NonZeroWeights) : WriterEvent;
    /// <summary>The reply is complete.</summary>
    public sealed record Done(string Reply, int Statements, int Applied, int Refused) : WriterEvent;
}

/// <summary>
/// Weight-space programming in one loop: send a prompt to any IChatClient, and apply the edits
/// block of the reply to a network statement by statement while the text is still arriving.
/// The model on the other side can be a recording, a local model or a hosted one; this class only
/// sees the Microsoft.Extensions.AI abstraction.
/// </summary>
public sealed class WeightWriter(IChatClient client)
{
    /// <param name="prompt">The message to send.</param>
    /// <param name="network">Changed in place as statements arrive.</param>
    /// <param name="reset">For the `reset` statement.</param>
    /// <param name="restore">For the `restore N` statement.</param>
    public async IAsyncEnumerable<WriterEvent> WriteAsync(string prompt, Network network,
        Func<Network>? reset = null, Func<int, Network?>? restore = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var reply = new StringBuilder();
        var pending = new StringBuilder();                 // text after the last complete line
        var before = network.Clone();                      // only the LAST edits block counts
        Edits.Assembler? assembler = null;
        int statements = 0, applied = 0;

        IEnumerable<WriterEvent> Line(string line)
        {
            string trimmed = line.TrimEnd('\r');
            if (assembler is null)
            {
                if (!trimmed.TrimStart().StartsWith("```edits")) yield break;
                foreach (var (key, t) in before.Tensors) t.Data.CopyTo(network[key].Data, 0);   // a new block starts over
                (assembler, statements, applied) = (new Edits.Assembler(), 0, 0);
                yield break;
            }
            var statement = trimmed.TrimStart().StartsWith("```") ? assembler.Flush() : assembler.Feed(trimmed);
            if (trimmed.TrimStart().StartsWith("```")) assembler = null;
            if (statement is not { } s) yield break;
            var outcome = Edits.ApplyStatement(network, s.Line, s.Text, reset, restore);
            statements++;
            if (outcome.Applied) applied++;
            yield return new WriterEvent.Edit(outcome, network.NonZero);
        }

        await foreach (var update in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, prompt)],
                           cancellationToken: cancellationToken))
        {
            if (update.AdditionalProperties?.TryGetValue("thinking_tokens", out object? tokens) == true && tokens is int n)
                yield return new WriterEvent.Thinking(n);
            if (string.IsNullOrEmpty(update.Text)) continue;
            reply.Append(update.Text);
            pending.Append(update.Text);
            yield return new WriterEvent.Text(update.Text);

            string buffered = pending.ToString();
            int newline;
            while ((newline = buffered.IndexOf('\n')) >= 0)
            {
                foreach (var e in Line(buffered[..newline])) yield return e;
                buffered = buffered[(newline + 1)..];
            }
            pending.Clear().Append(buffered);
        }
        foreach (var e in Line(pending.ToString())) yield return e;
        yield return new WriterEvent.Done(reply.ToString(), statements, applied, statements - applied);
    }
}
