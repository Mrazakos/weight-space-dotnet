namespace WeightSpace.Core;

/// <summary>Where the repository's data lives, found by walking up from the running program.</summary>
public static class Repo
{
    private static readonly Lazy<string> RootPath = new(() =>
    {
        string? fromEnv = Environment.GetEnvironmentVariable("WEIGHTSPACE_ROOT");
        if (!string.IsNullOrEmpty(fromEnv)) return fromEnv;
        foreach (string start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
            for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
                if (File.Exists(Path.Combine(dir.FullName, "WeightSpace.sln"))) return dir.FullName;
        throw new DirectoryNotFoundException("WeightSpace.sln not found above the working directory; set WEIGHTSPACE_ROOT");
    });

    public static string Root => RootPath.Value;
    public static string Data(params string[] parts) => Path.Combine([Root, "data", .. parts]);
    public static string Experiment(params string[] parts) => Path.Combine([Root, "experiment", .. parts]);
    public static string Prompts(params string[] parts) => Path.Combine([Root, "prompts", .. parts]);

    /// <summary>The recorded reply of one turn of one run of the experiment.</summary>
    public static string Reply(string run, int turn) =>
        File.ReadAllText(Experiment("runs", run, $"turn_{turn:00}", "answer.md"));

    /// <summary>What was sent in that turn: prompt.txt for the first, message.txt afterwards.</summary>
    public static string Sent(string run, int turn) =>
        File.ReadAllText(Experiment("runs", run, $"turn_{turn:00}", turn == 0 ? "prompt.txt" : "message.txt"));
}
