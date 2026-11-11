using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

namespace WeightSpace.AI;

/// <summary>
/// An IChatClient that answers with a reply recorded during the experiment, streamed in pieces.
/// It makes the demo independent of the network and of any account, and it is what lets the tests
/// prove that the recorded replies really produce the stored networks.
/// </summary>
public sealed class ReplayChatClient(string reply, int charsPerChunk = 48, TimeSpan? delayPerChunk = null) : IChatClient
{
    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default) =>
        Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, reply)) { ModelId = "replay" });

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
        ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        for (int at = 0; at < reply.Length; at += charsPerChunk)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return new ChatResponseUpdate(ChatRole.Assistant, reply.Substring(at, Math.Min(charsPerChunk, reply.Length - at)))
            {
                ModelId = "replay",
            };
            if (delayPerChunk is { } delay && delay > TimeSpan.Zero) await Task.Delay(delay, cancellationToken);
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }
}
