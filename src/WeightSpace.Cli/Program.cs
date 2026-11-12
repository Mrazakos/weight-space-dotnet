using Microsoft.Extensions.AI;
using WeightSpace.AI;
using WeightSpace.Core;
using WeightSpace.Onnx;

// weightspace run <network>              run a stored network on the demo clip
// weightspace write replay:<run>[:turn]  rebuild a network from a recorded reply
// weightspace write claude [effort]      ask Claude (CLI, tools off) to write one now
//
// Examples
//   dotnet run --project src/WeightSpace.Cli -- run expA [--engine onnx|csharp]
//   dotnet run --project src/WeightSpace.Cli -- write replay:expA_scratch_oneshot
//   dotnet run --project src/WeightSpace.Cli -- write claude medium

System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
// --engine onnx|csharp (default onnx) may appear anywhere
string engineName = "onnx";
int ei = Array.IndexOf(args, "--engine");
if (ei >= 0 && ei + 1 < args.Length) { engineName = args[ei + 1]; args = [.. args.Where((_, i) => i != ei && i != ei + 1)]; }
var engine = Engines.Factory(engineName);
if (args.Length < 2) return Usage();
var clip = Clip.Load(Repo.Data("clip"));

switch (args[0])
{
    case "run":
        Report(Network.Load(Repo.Data("networks", args[1] + ".json")));
        return 0;

    case "write":
        string run = "expA_scratch_oneshot";
        IChatClient client;
        if (args[1].StartsWith("replay:"))
        {
            string[] parts = args[1].Split(':');
            run = parts[1];
            client = new ReplayChatClient(Repo.Reply(run, parts.Length > 2 ? int.Parse(parts[2]) : 0), 4096);
        }
        else if (args[1] == "claude")
            client = new ClaudeCliChatClient(effort: args.Length > 2 ? args[2] : "medium");
        else return Usage();

        string prompt = Repo.Sent(run, 0);
        var network = new Network { Name = "written", Description = "written just now through " + client.GetType().Name };
        Console.WriteLine($"prompt: {prompt.Length:N0} characters -> {client.GetType().Name}");
        await foreach (var e in new WeightWriter(client).WriteAsync(prompt, network))
            switch (e)
            {
                case WriterEvent.Thinking t:
                    Console.Write($"\rreasoning: {t.Tokens:N0} tokens   ");
                    break;
                case WriterEvent.Edit { Outcome.Applied: false } edit:
                    Console.WriteLine($"\nskipped line {edit.Outcome.Line}: {edit.Outcome.Text} -> {edit.Outcome.Error}");
                    break;
                case WriterEvent.Done done:
                    Console.WriteLine($"\nreply: {done.Reply.Length:N0} characters, {done.Statements} statements, {done.Applied} applied, {done.Refused} skipped");
                    break;
            }
        if (client is ClaudeCliChatClient cli) Console.WriteLine($"tool-free call: {cli.LastCallWasToolFree}");
        Report(network);
        string path = Path.Combine(Repo.Root, "written.json");
        network.Save(path);
        Console.WriteLine($"saved {path}");
        return 0;

    default:
        return Usage();
}

void Report(Network network)
{
    var result = Pipeline.Run(network, clip, engine);
    var centre = result.Labels.AsSpan(clip.Margin, clip.Frames);
    int trainedRight = Enumerable.Range(clip.Margin, clip.Frames).Count(t => clip.Trained[t] == clip.Truth[t]);
    Console.WriteLine($"{network.Name}: {network.Description}");
    Console.WriteLine($"  weights set   {network.NonZero:N0} of {Network.TotalWeights:N0}");
    Console.WriteLine($"  clip          {clip.Frames / clip.Fps} s of {clip.Source}");
    Console.WriteLine($"  frames right  {result.Accuracy(clip):P1}   (trained network: {trainedRight / (double)clip.Frames:P1})");
    Console.WriteLine($"  possessions   {ClipResult.Possessions(centre)} found, {ClipResult.Possessions(clip.Truth.AsSpan(clip.Margin, clip.Frames))} true");
    Console.WriteLine($"  inference     {result.Elapsed.TotalMilliseconds:N0} ms for 4 views x {clip.TotalFrames:N0} frames, {result.Engine}");
}

static int Usage()
{
    Console.WriteLine("usage: weightspace run <expA|expB|expB_pass01..10> [--engine onnx|csharp]\n       weightspace write replay:<run>[:turn] | claude [effort]");
    return 1;
}
