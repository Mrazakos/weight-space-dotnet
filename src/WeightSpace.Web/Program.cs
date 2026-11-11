using System.Globalization;
using WeightSpace.Web;
using WeightSpace.Web.Components;

CultureInfo.DefaultThreadCurrentCulture = CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddRazorComponents().AddInteractiveServerComponents();
builder.Services.AddSingleton<DemoData>();

var app = builder.Build();
if (!app.Environment.IsDevelopment()) app.UseExceptionHandler("/Error", createScopeForErrors: true);
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);
app.UseAntiforgery();
app.MapStaticAssets();

// What the replay canvas draws from: positions, three label tracks, probabilities, unit traces.
app.MapGet("/api/replay/{id}", (string id, DemoData data) =>
    data.Has(id) ? Results.Text(data.ReplayJson(id), "application/json") : Results.NotFound());

// The charts as files, for slides: /charts/results.svg and /charts/passes.svg (?theme=dark).
app.MapGet("/charts/results.svg", (DemoData data, string? theme) =>
    Results.Text(Charts.Results(data.Benchmark, theme == "dark"), "image/svg+xml"));
app.MapGet("/charts/passes.svg", (DemoData data, string? theme) =>
    Results.Text(Charts.Passes(data.PassScores, theme == "dark"), "image/svg+xml"));

app.MapRazorComponents<App>().AddInteractiveServerRenderMode();
app.Run();
