using System.Globalization;
using System.Text.RegularExpressions;

namespace WeightSpace.Core;

/// <summary>One statement of an edits block after it was applied (or refused).</summary>
public sealed record EditOutcome(int Line, string Text, bool Applied, string? Error, string? Tensor, int[] Indices);

/// <summary>What happened to a whole reply.</summary>
public sealed class EditReport
{
    public bool FoundBlock { get; set; }
    public List<EditOutcome> Outcomes { get; } = [];
    public int Statements => Outcomes.Count;
    public int Applied => Outcomes.Count(o => o.Applied);
    public IEnumerable<EditOutcome> Errors => Outcomes.Where(o => !o.Applied);
}

/// <summary>
/// The edits language: how a model's reply sets network weights. A reply ends with a fenced block
/// opened by ```edits, one statement per line:
///
///     inp.W[3, 47] = -2.5                 one weight
///     conv.W[10, 3, :] = 0.111            a number fills a range
///     head2.W[0, 0:3] = [4, -4, 0.5]      a list, numpy assignment rules
///     gru1f.bhz[0:8] += 2                 in place: =  +=  -=  *=
///     reset                               every weight back to zero
///     restore 3                           back to the network after pass 3
///
/// Nothing in a reply is executed. The left side is matched by a regular expression, the right
/// side is read by a small parser that accepts numbers and nested lists of numbers, and anything
/// else is skipped and reported. A port of wsp/harness/edits.py from the experiment.
/// </summary>
public static partial class Edits
{
    public const double MaxAbs = 1e6;

    [GeneratedRegex(@"```edits[^\n]*\n(.*?)```", RegexOptions.Singleline)]
    private static partial Regex BlockRegex();

    [GeneratedRegex(@"^([A-Za-z_][\w.]*)\s*(?:\[(.*?)\])?\s*(\+=|-=|\*=|=)\s*(.+)$", RegexOptions.Singleline)]
    private static partial Regex StatementRegex();

    [GeneratedRegex(@"^[+-]?\d+$")]
    private static partial Regex IntRegex();

    /// <summary>Friendly tensor names: one per block, per GRU gate and per input group of the head.</summary>
    public static IReadOnlyDictionary<string, Alias> Aliases { get; } = BuildAliases();

    /// <summary>A named view into a state-dict tensor: which rows and columns of it.</summary>
    public sealed record Alias(string Key, int[] Shape, int[] Offset);

    private static Dictionary<string, Alias> BuildAliases()
    {
        const int h = Network.Hidden;
        var a = new Dictionary<string, Alias>
        {
            ["inp.W"] = new("inp.0.weight", [h, Network.Features], [0, 0]),
            ["inp.b"] = new("inp.0.bias", [h], [0]),
            ["conv.W"] = new("conv.weight", [h, h, Network.Kernel], [0, 0, 0]),
            ["conv.b"] = new("conv.bias", [h], [0]),
            ["head1.Wf"] = new("head.0.weight", [h, h], [0, 0]),
            ["head1.Wb"] = new("head.0.weight", [h, h], [0, h]),
            ["head1.Wh"] = new("head.0.weight", [h, h], [0, 2 * h]),
            ["head1.b"] = new("head.0.bias", [h], [0]),
            ["head2.W"] = new("head.2.weight", [3, h], [0, 0]),
            ["head2.b"] = new("head.2.bias", [3], [0]),
        };
        foreach (var (layer, nIn) in new[] { (0, h), (1, 2 * h) })
            foreach (var (suffix, dir) in new[] { ("", "f"), ("_reverse", "b") })
                for (int g = 0; g < 3; g++)                       // PyTorch stacks reset, update, new
                {
                    string name = $"gru{layer + 1}{dir}", gate = "rzn"[g].ToString();
                    a[$"{name}.Wi{gate}"] = new($"gru.weight_ih_l{layer}{suffix}", [h, nIn], [g * h, 0]);
                    a[$"{name}.Wh{gate}"] = new($"gru.weight_hh_l{layer}{suffix}", [h, h], [g * h, 0]);
                    a[$"{name}.bi{gate}"] = new($"gru.bias_ih_l{layer}{suffix}", [h], [g * h]);
                    a[$"{name}.bh{gate}"] = new($"gru.bias_hh_l{layer}{suffix}", [h], [g * h]);
                }
        return a;
    }

    /// <summary>The body of the last ```edits block of a reply, or null.</summary>
    public static string? LastBlock(string text)
    {
        var matches = BlockRegex().Matches(text);
        return matches.Count == 0 ? null : matches[^1].Groups[1].Value;
    }

    /// <summary>(first line, text) per statement; a statement continues while brackets are open.</summary>
    public static List<(int Line, string Text)> Statements(string block)
    {
        var result = new List<(int, string)>();
        var assembler = new Assembler();
        foreach (string raw in block.Split('\n'))
            if (assembler.Feed(raw) is { } statement) result.Add(statement);
        if (assembler.Flush() is { } rest) result.Add(rest);
        return result;
    }

    /// <summary>
    /// Turns lines into statements one at a time, so that a reply can be applied while it is
    /// still streaming in: feed each finished line, get a statement back when one is complete.
    /// </summary>
    public sealed class Assembler
    {
        private readonly List<string> _buffer = [];
        private int _start, _depth, _n;

        public (int Line, string Text)? Feed(string rawLine)
        {
            _n++;
            string line = rawLine.Split('#', 2)[0].TrimEnd();
            if (line.Trim().Length == 0 && _depth == 0) return null;
            if (_depth == 0) { _buffer.Clear(); _start = _n; }
            _buffer.Add(line.Trim());
            _depth += line.Count(c => c == '[') - line.Count(c => c == ']');
            if (_depth > 0) return null;
            _depth = 0;
            return (_start, string.Join(" ", _buffer));
        }

        /// <summary>An unfinished statement at the end of the block, if any.</summary>
        public (int Line, string Text)? Flush()
        {
            if (_depth <= 0) return null;
            _depth = 0;
            return (_start, string.Join(" ", _buffer));
        }
    }

    /// <summary>Apply the last edits block of a reply to a network, in place.</summary>
    public static EditReport Apply(Network network, string reply,
        Func<Network>? reset = null, Func<int, Network?>? restore = null)
    {
        var report = new EditReport();
        string? block = LastBlock(reply);
        report.FoundBlock = block is not null;
        if (block is null)
        {
            report.Outcomes.Add(new(0, "", false, "no ```edits block in the reply", null, []));
            return report;
        }
        foreach (var (line, text) in Statements(block))
            report.Outcomes.Add(ApplyStatement(network, line, text, reset, restore));
        return report;
    }

    /// <summary>Apply one statement. A statement that cannot be read changes nothing.</summary>
    public static EditOutcome ApplyStatement(Network network, int line, string text,
        Func<Network>? reset = null, Func<int, Network?>? restore = null)
    {
        try
        {
            string[] words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            Network? replacement = null;
            if (words is ["reset"])
                replacement = (reset ?? throw new EditException("reset is not available in this run"))();
            else if (words is ["restore", var n] && IntRegex().IsMatch(n))
                replacement = (restore ?? throw new EditException("restore is not available in this run"))(int.Parse(n))
                              ?? throw new EditException($"there is no pass {n} to restore");
            if (replacement is not null)
            {
                foreach (var (key, t) in replacement.Tensors) t.Data.CopyTo(network[key].Data, 0);
                return new(line, text, true, null, null, []);
            }

            var m = StatementRegex().Match(text);
            if (!m.Success) throw new EditException("not of the form NAME[index] = value");
            string name = m.Groups[1].Value, op = m.Groups[3].Value;
            if (!Aliases.TryGetValue(name, out var alias)) throw new EditException($"unknown tensor '{name}'");
            var target = network[alias.Key];
            var selection = Select(m.Groups[2].Success ? m.Groups[2].Value : "", alias.Shape);
            var value = Value.Parse(m.Groups[4].Value);

            int[] flat = FlatIndices(selection, alias, target.Shape);
            double[] values = Broadcast(value, selection);
            var updated = new float[flat.Length];
            for (int i = 0; i < flat.Length; i++)
            {
                double current = target.Data[flat[i]];
                double result = op switch { "=" => values[i], "+=" => current + values[i], "-=" => current - values[i], _ => current * values[i] };
                if (!double.IsFinite(result) || Math.Abs(result) > MaxAbs)
                    throw new EditException($"result is non-finite or larger than {MaxAbs:g}");
                updated[i] = (float)result;
            }
            for (int i = 0; i < flat.Length; i++) target.Data[flat[i]] = updated[i];    // all or nothing
            return new(line, text, true, null, alias.Key, flat);
        }
        catch (EditException e)
        {
            return new(line, text.Length > 200 ? text[..200] : text, false, e.Message, null, []);
        }
    }

    /// <summary>Per axis, the positions an index selects, and whether the axis survives (a range).</summary>
    private sealed record Selection(List<int[]> Positions, List<bool> Kept)
    {
        public int[] Shape => [.. Positions.Where((_, d) => Kept[d]).Select(p => p.Length)];
        public int Count => Positions.Aggregate(1, (a, p) => a * p.Length);
    }

    private static Selection Select(string index, int[] shape)
    {
        string[] parts = index.Trim().Length == 0 ? [] : [.. index.Split(',').Select(p => p.Trim())];
        if (parts.Length > shape.Length)
            throw new EditException($"{parts.Length} indices for a tensor with {shape.Length} axes");
        var positions = new List<int[]>();
        var kept = new List<bool>();
        for (int d = 0; d < shape.Length; d++)
        {
            int size = shape[d];
            if (d >= parts.Length) { positions.Add([.. Enumerable.Range(0, size)]); kept.Add(true); continue; }
            string part = parts[d];
            if (IntRegex().IsMatch(part))
            {
                int i = int.Parse(part, CultureInfo.InvariantCulture);
                if (i < -size || i >= size) throw new EditException($"index {i} out of range for axis of size {size}");
                positions.Add([i < 0 ? i + size : i]);
                kept.Add(false);
            }
            else if (part.Contains(':'))
            {
                string[] f = part.Split(':');
                if (f.Length > 3 || f.Any(x => x.Trim().Length > 0 && !IntRegex().IsMatch(x.Trim())))
                    throw new EditException($"cannot read range '{part}'");
                int?[] v = [.. f.Select(x => x.Trim().Length > 0 ? int.Parse(x.Trim(), CultureInfo.InvariantCulture) : (int?)null)];
                int? start = v[0], stop = v.Length > 1 ? v[1] : null;
                int step = (v.Length > 2 ? v[2] : null) ?? 1;
                foreach (int? bound in new[] { start, stop })
                    if (bound is int b && (b < -size || b > size))
                        throw new EditException($"range '{part}' out of bounds for axis of size {size}");
                if (step == 0) throw new EditException($"range step 0 in '{part}'");
                positions.Add(PythonSlice(start, stop, step, size));
                kept.Add(true);
            }
            else throw new EditException($"cannot read index '{part}' (integers and a:b:s ranges only)");
        }
        return new Selection(positions, kept);
    }

    private static int[] PythonSlice(int? start, int? stop, int step, int size)
    {
        var result = new List<int>();
        if (step > 0)
        {
            int s = start ?? 0, e = stop ?? size;
            if (s < 0) s += size;
            if (e < 0) e += size;
            for (int i = Math.Clamp(s, 0, size); i < Math.Clamp(e, 0, size); i += step) result.Add(i);
        }
        else
        {
            int s = start is int a ? (a < 0 ? a + size : a) : size - 1;
            int e = stop is int b ? (b < 0 ? b + size : b) : -1;
            for (int i = Math.Clamp(s, -1, size - 1); i > Math.Clamp(e, -1, size - 1); i += step) result.Add(i);
        }
        return [.. result];
    }

    /// <summary>Flat positions in the underlying tensor, in row-major order of the selection.</summary>
    private static int[] FlatIndices(Selection selection, Alias alias, int[] tensorShape)
    {
        var strides = new int[tensorShape.Length];
        for (int d = tensorShape.Length - 1, s = 1; d >= 0; d--) { strides[d] = s; s *= tensorShape[d]; }
        var flat = new int[selection.Count];
        var counter = new int[selection.Positions.Count];
        for (int n = 0; n < flat.Length; n++)
        {
            int index = 0;
            for (int d = 0; d < counter.Length; d++)
                index += (alias.Offset[d] + selection.Positions[d][counter[d]]) * strides[d];
            flat[n] = index;
            for (int d = counter.Length - 1; d >= 0 && ++counter[d] == selection.Positions[d].Length; d--) counter[d] = 0;
        }
        return flat;
    }

    /// <summary>numpy broadcasting of a parsed value onto the selected block.</summary>
    private static double[] Broadcast(Value value, Selection selection)
    {
        int[] shape = selection.Shape;
        int total = shape.Aggregate(1, (a, b) => a * b);
        if (value.Shape.Length > shape.Length)
            throw new EditException($"shape mismatch: a value of shape ({string.Join(",", value.Shape)}) for a block of shape ({string.Join(",", shape)})");
        int lead = shape.Length - value.Shape.Length;
        for (int d = 0; d < value.Shape.Length; d++)
            if (value.Shape[d] != shape[lead + d] && value.Shape[d] != 1)
                throw new EditException($"shape mismatch: a value of shape ({string.Join(",", value.Shape)}) for a block of shape ({string.Join(",", shape)})");
        var result = new double[total];
        var counter = new int[shape.Length];
        for (int n = 0; n < total; n++)
        {
            int index = 0;
            for (int d = 0; d < value.Shape.Length; d++)
                index = index * value.Shape[d] + (value.Shape[d] == 1 ? 0 : counter[lead + d]);
            result[n] = value.Data[index];
            for (int d = shape.Length - 1; d >= 0 && ++counter[d] == shape[d]; d--) counter[d] = 0;
        }
        return result;
    }

    /// <summary>A number or a rectangular nested list of numbers.</summary>
    private sealed record Value(int[] Shape, double[] Data)
    {
        public static Value Parse(string text)
        {
            int pos = 0, leafDepth = -1;
            var data = new List<double>();
            var shape = new List<int>();                      // shape[d] = length of every list at depth d
            ParseNode(text, ref pos, 0, data, shape, ref leafDepth);
            SkipSpace(text, ref pos);
            if (pos != text.Length) throw new EditException("right-hand side is not a number or a list of numbers");
            return new Value([.. shape], [.. data]);
        }

        private static void SkipSpace(string s, ref int pos)
        {
            while (pos < s.Length && char.IsWhiteSpace(s[pos])) pos++;
        }

        private static void ParseNode(string s, ref int pos, int depth, List<double> data, List<int> shape, ref int leafDepth)
        {
            const string ragged = "right-hand side is not a rectangular list of numbers";
            SkipSpace(s, ref pos);
            if (pos < s.Length && s[pos] == '[')
            {
                if (leafDepth == depth) throw new EditException(ragged);       // a list where numbers were
                pos++;
                int count = 0;
                while (true)
                {
                    SkipSpace(s, ref pos);
                    if (pos < s.Length && s[pos] == ']') { pos++; break; }
                    ParseNode(s, ref pos, depth + 1, data, shape, ref leafDepth);
                    count++;
                    SkipSpace(s, ref pos);
                    if (pos < s.Length && s[pos] == ',') pos++;
                    else if (pos >= s.Length || s[pos] != ']')
                        throw new EditException("right-hand side is not a number or a list of numbers");
                }
                if (count == 0) throw new EditException("right-hand side has an empty list");
                while (shape.Count <= depth) shape.Add(-1);
                if (shape[depth] == -1) shape[depth] = count;
                else if (shape[depth] != count) throw new EditException(ragged);
                return;
            }
            int begin = pos;
            while (pos < s.Length && (char.IsAsciiLetterOrDigit(s[pos]) || s[pos] is '+' or '-' or '.' or '_')) pos++;
            string token = s[begin..pos].Replace("_", "");
            if (!double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) || !double.IsFinite(v))
                throw new EditException("right-hand side is not a number or a list of numbers");
            if (Math.Abs(v) > MaxAbs) throw new EditException($"value larger than {MaxAbs:g}");
            if (leafDepth == -1) leafDepth = depth;
            else if (leafDepth != depth) throw new EditException(ragged);     // a number where lists were
            data.Add(v);
        }
    }

    private sealed class EditException(string message) : Exception(message);
}
