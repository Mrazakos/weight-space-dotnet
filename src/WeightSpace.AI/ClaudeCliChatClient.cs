using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace WeightSpace.AI;

/// <summary>
/// An IChatClient over the Claude Code command line in headless mode, with every tool switched
/// off: the prompt goes in on stdin, the reply comes back as an event stream, and nothing else
/// can happen in between. This is the setup the experiment used (its Python harness does the same
/// call), and it needs no API key, only a logged-in `claude` on the PATH.
///
/// It is a deliberately small adapter: one user turn in, one assistant message out. While the
/// model is reasoning, updates carry no text but report the running count of reasoning tokens in
/// AdditionalProperties["thinking_tokens"].
/// </summary>
public sealed class ClaudeCliChatClient(string model = "claude-fable-5-2", string effort = "medium",
    string systemPrompt = "You are a helpful assistant.") : IChatClient
{
    public string Model { get; } = model;
    public string Effort { get; } = effort;

    /// <summary>True when the last call's session reported no tools and no MCP servers.</summary>
    public bool? LastCallWasToolFree { get; private set; }

    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var text = new StringBuilder();
        await foreach (var update in GetStreamingResponseAsync(messages, options, cancellationToken))
            text.Append(update.Text);
        return new ChatResponse(new ChatMessage(ChatRole.Assistant, text.ToString())) { ModelId = Model };
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
        ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        string prompt = string.Join("\n\n", messages.Where(m => m.Role == ChatRole.User).Select(m => m.Text));
        string workDir = Path.Combine(Path.GetTempPath(), "weightspace_sessions", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(workDir);                      // an empty folder: no project context

        var start = new ProcessStartInfo("claude")
        {
            WorkingDirectory = workDir,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardInputEncoding = new UTF8Encoding(false),
        };
        foreach (string arg in new[]
                 {
                     "-p", "--model", options?.ModelId ?? Model, "--effort", Effort, "--tools", "",
                     "--strict-mcp-config", "--safe-mode", "--disable-slash-commands",
                     "--system-prompt", systemPrompt, "--output-format", "stream-json", "--verbose",
                 })
            start.ArgumentList.Add(arg);
        start.Environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "128000";       // a network is a long reply

        using var process = Process.Start(start) ?? throw new InvalidOperationException("could not start `claude`");
        try
        {
            await process.StandardInput.WriteAsync(prompt.AsMemory(), cancellationToken);
            process.StandardInput.Close();
            _ = process.StandardError.ReadToEndAsync(cancellationToken);    // keep the pipe drained

            bool toolFree = false, sawText = false;
            while (await process.StandardOutput.ReadLineAsync(cancellationToken) is { } line)
            {
                if (line.Length == 0 || line[0] != '{') continue;
                using var doc = JsonDocument.Parse(line);
                var e = doc.RootElement;
                string type = e.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
                string subtype = e.TryGetProperty("subtype", out var s) ? s.GetString() ?? "" : "";

                if (type == "system" && subtype == "init")
                {
                    toolFree = e.GetProperty("tools").GetArrayLength() == 0 && e.GetProperty("mcp_servers").GetArrayLength() == 0;
                    LastCallWasToolFree = toolFree;
                }
                else if (type == "system" && subtype == "thinking_tokens")
                {
                    yield return new ChatResponseUpdate
                    {
                        Role = ChatRole.Assistant, ModelId = Model,
                        AdditionalProperties = new() { ["thinking_tokens"] = e.GetProperty("estimated_tokens").GetInt32() },
                    };
                }
                else if (type == "assistant")
                {
                    foreach (var block in e.GetProperty("message").GetProperty("content").EnumerateArray())
                    {
                        string kind = block.GetProperty("type").GetString() ?? "";
                        if (kind is "tool_use" or "server_tool_use") LastCallWasToolFree = false;
                        if (kind != "text") continue;
                        sawText = true;
                        yield return new ChatResponseUpdate(ChatRole.Assistant, block.GetProperty("text").GetString()) { ModelId = Model };
                    }
                }
                else if (type == "result")
                {
                    if (e.TryGetProperty("is_error", out var err) && err.GetBoolean())
                        throw new InvalidOperationException("claude reported an error: " +
                            (e.TryGetProperty("result", out var r) ? r.GetString() : "unknown"));
                    if (!sawText && e.TryGetProperty("result", out var result))
                        yield return new ChatResponseUpdate(ChatRole.Assistant, result.GetString()) { ModelId = Model };
                }
            }
            await process.WaitForExitAsync(cancellationToken);
        }
        finally
        {
            if (!process.HasExited) process.Kill(entireProcessTree: true);
            try { Directory.Delete(workDir, recursive: true); } catch (IOException) { }
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }
}
