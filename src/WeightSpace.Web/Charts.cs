using System.Globalization;
using System.Text;

namespace WeightSpace.Web;

/// <summary>
/// The two result charts as plain SVG strings, so the same drawing serves the page and the
/// files exported for slides (/charts/results.svg, /charts/passes.svg, add ?theme=dark).
/// Colour says how the weights were made; every mark carries its value.
/// </summary>
public static class Charts
{
    private sealed record Theme(string Page, string Ink, string Ink2, string Muted, string Grid, string Axis,
        string None, string Written, string Trained, string[] Series);

    private static readonly Theme Light = new("#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
        "#b4b2aa", "#2a78d6", "#eb6834", ["#2a78d6", "#eb6834", "#1baf7a"]);
    private static readonly Theme Dark = new("#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
        "#6f6e68", "#3987e5", "#d95926", ["#3987e5", "#d95926", "#199e70"]);
    private const string Font = "system-ui, -apple-system, 'Segoe UI', sans-serif";

    private static string F(double v) => v.ToString("0.##", CultureInfo.InvariantCulture);
    private static string Esc(string s) => s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");

    /// <summary>spellF1 on the held-out benchmark: one panel per tracking source, one bar per variant.</summary>
    public static string Results(Benchmark bench, bool dark = false)
    {
        var t = dark ? Dark : Light;
        const int labelW = 300, panelW = 250, gap = 44, top = 96, rowH = 46, barH = 26;
        int rows = bench.Rows.Count, width = labelW + 3 * panelW + 2 * gap + 80, height = top + rows * rowH + 96;
        string Colour(string kind) => kind switch { "none" => t.None, "written" => t.Written, _ => t.Trained };

        var s = new StringBuilder();
        s.Append($"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}' font-family=\"{Font}\" role='img' aria-label='Benchmark spellF1 per tracking source and variant'>");
        s.Append($"<rect width='{width}' height='{height}' fill='{t.Page}'/>");
        s.Append($"<text x='24' y='40' font-size='22' font-weight='600' fill='{t.Ink}'>Held-out benchmark: possessions recovered (spellF1, higher is better)</text>");
        for (int r = 0; r < rows; r++)
            s.Append($"<text x='{labelW - 14}' y='{top + r * rowH + barH / 2 + 6}' font-size='16' text-anchor='end' fill='{t.Ink}'>{Esc(bench.Rows[r].Label)}</text>");

        for (int p = 0; p < 3; p++)
        {
            string src = Benchmark.SourceOrder[p];
            int x0 = labelW + p * (panelW + gap);
            s.Append($"<text x='{x0}' y='{top - 18}' font-size='16' fill='{t.Ink}'>{Esc(bench.SourceNames[src])}</text>");
            foreach (double tick in new[] { 0.25, 0.5, 0.75, 1.0 })
                s.Append($"<line x1='{F(x0 + tick * panelW)}' x2='{F(x0 + tick * panelW)}' y1='{top - 6}' y2='{top + rows * rowH - 8}' stroke='{t.Grid}' stroke-width='1'/>");
            foreach (double tick in new[] { 0, 0.5, 1.0 })
                s.Append($"<text x='{F(x0 + tick * panelW)}' y='{top + rows * rowH + 12}' font-size='13' text-anchor='middle' fill='{t.Muted}'>{F(tick)}</text>");
            for (int r = 0; r < rows; r++)
            {
                var row = bench.Rows[r];
                var score = row.Sources[src];
                double w = score.SpellF1 * panelW;
                int y = top + r * rowH;
                // square at the baseline, rounded at the data end
                s.Append($"<path d='M{x0} {y}h{F(Math.Max(0, w - 4))}a4 4 0 0 1 4 4v{barH - 8}a4 4 0 0 1 -4 4h-{F(Math.Max(0, w - 4))}z' fill='{Colour(row.Kind)}'>");
                s.Append($"<title>{Esc(row.Label)} — {Esc(bench.SourceNames[src])}: spellF1 {score.SpellF1:0.000}, macroF1 {score.MacroF1:0.000}, {score.Predicted} possessions found of {score.True}</title></path>");
                s.Append($"<text x='{F(x0 + w + 8)}' y='{y + barH / 2 + 6}' font-size='15' fill='{t.Ink}'>{score.SpellF1.ToString("0.000", CultureInfo.InvariantCulture)}</text>");
            }
            s.Append($"<line x1='{x0}' x2='{x0}' y1='{top - 6}' y2='{top + rows * rowH - 8}' stroke='{t.Axis}' stroke-width='1.5'/>");
        }

        int ly = height - 30, lx = labelW;
        foreach (var (kind, label) in new[] { ("none", "no network (rule baseline)"), ("written", "weights written by the LLM, no training"), ("trained", "weights trained by SGD") })
        {
            s.Append($"<rect x='{lx}' y='{ly - 13}' width='16' height='16' rx='3' fill='{Colour(kind)}'/>");
            s.Append($"<text x='{lx + 24}' y='{ly}' font-size='14' fill='{t.Ink2}'>{label}</text>");
            lx += 56 + (int)(label.Length * 7.4);
        }
        return s.Append("</svg>").ToString();
    }

    /// <summary>Experiment B: the score on the practice matches after each pass, per tracking source.</summary>
    public static string Passes(IReadOnlyList<IReadOnlyDictionary<string, double>> passes, bool dark = false)
    {
        var t = dark ? Dark : Light;
        const int width = 980, height = 490, left = 64, right = 190, top = 96, bottom = 62;
        double plotW = width - left - right, plotH = height - top - bottom;
        double X(int pass) => left + (pass - 1) / (double)(passes.Count - 1) * plotW;
        double Y(double v) => top + (1 - v) * plotH;
        var names = new Dictionary<string, string> { ["DFL"] = "DFL", ["SC"] = "SkillCorner", ["MET"] = "Metrica" };

        var s = new StringBuilder();
        s.Append($"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}' font-family=\"{Font}\" role='img' aria-label='Practice score of the hand-written network after each pass'>");
        s.Append($"<rect width='{width}' height='{height}' fill='{t.Page}'/>");
        s.Append($"<text x='24' y='38' font-size='22' font-weight='600' fill='{t.Ink}'>A hand-written network, revised after each practice pass</text>");
        foreach (double v in new[] { 0, 0.2, 0.4, 0.6, 0.8, 1.0 })
        {
            s.Append($"<line x1='{left}' x2='{F(left + plotW)}' y1='{F(Y(v))}' y2='{F(Y(v))}' stroke='{(v == 0 ? t.Axis : t.Grid)}' stroke-width='1'/>");
            s.Append($"<text x='{left - 10}' y='{F(Y(v) + 5)}' font-size='13' text-anchor='end' fill='{t.Muted}'>{F(v)}</text>");
        }
        for (int p = 1; p <= passes.Count; p++)
            s.Append($"<text x='{F(X(p))}' y='{height - bottom + 22}' font-size='13' text-anchor='middle' fill='{t.Muted}'>{p}</text>");
        s.Append($"<text x='{F(left + plotW / 2)}' y='{height - 14}' font-size='14' text-anchor='middle' fill='{t.Ink2}'>practice pass</text>");
        s.Append($"<text x='24' y='{top - 24}' font-size='14' fill='{t.Ink2}'>spellF1 on the practice matches</text>");

        for (int k = 0; k < Benchmark.SourceOrder.Length; k++)
        {
            string src = Benchmark.SourceOrder[k], colour = t.Series[k];
            var points = passes.Select((d, i) => (X: X(i + 1), Y: Y(d[src]), V: d[src])).ToList();
            s.Append($"<polyline fill='none' stroke='{colour}' stroke-width='2.5' stroke-linejoin='round' points='{string.Join(" ", points.Select(p => $"{F(p.X)},{F(p.Y)}"))}'/>");
            for (int i = 0; i < points.Count; i++)
                s.Append($"<circle cx='{F(points[i].X)}' cy='{F(points[i].Y)}' r='5' fill='{colour}' stroke='{t.Page}' stroke-width='2'><title>{names[src]}, pass {i + 1}: {points[i].V:0.000}</title></circle>");
            var last = points[^1];
            s.Append($"<text x='{F(last.X + 14)}' y='{F(last.Y + 5)}' font-size='15' fill='{t.Ink}'>{names[src]}  {last.V.ToString("0.000", CultureInfo.InvariantCulture)}</text>");
        }
        // the pass that broke the network, and the one that fixed it
        foreach (int p in new[] { 5, 6 })
            s.Append($"<line x1='{F(X(p))}' x2='{F(X(p))}' y1='{F(Y(0.93))}' y2='{F(Y(passes[p - 1]["MET"]) - 9)}' stroke='{t.Axis}' stroke-width='1' stroke-dasharray='3 3'/>");
        s.Append($"<text x='{F(X(5) - 8)}' y='{F(Y(0.93) + 4)}' font-size='13' text-anchor='end' fill='{t.Ink2}'>pass 5: a new idea breaks it</text>");
        s.Append($"<text x='{F(X(6) + 8)}' y='{F(Y(0.93) + 4)}' font-size='13' text-anchor='start' fill='{t.Ink2}'>pass 6: it finds its own bug</text>");
        return s.Append("</svg>").ToString();
    }
}
