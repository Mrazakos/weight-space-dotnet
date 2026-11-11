using System.Numerics.Tensors;

namespace WeightSpace.Core;

/// <summary>What one forward pass produces for T frames.</summary>
public sealed class ForwardResult(int frames)
{
    public int Frames { get; } = frames;
    /// <summary>[T * 3] logits in the order H, A, D.</summary>
    public float[] Logits { get; } = new float[frames * 3];
    /// <summary>Layer-1 GRU states, [T * 128] each: forward ("gru1f") and backward ("gru1b").</summary>
    public float[] Gru1F { get; set; } = [];
    public float[] Gru1B { get; set; } = [];

    public float[] Trace(Probe probe)
    {
        var src = probe.Layer == "gru1b" ? Gru1B : Gru1F;
        var trace = new float[Frames];
        for (int t = 0; t < Frames; t++) trace[t] = src[t * Network.Hidden + probe.Unit];
        return trace;
    }
}

/// <summary>Something that can run the network: plain C# here, ONNX Runtime in WeightSpace.Onnx.</summary>
public interface IPossessionEngine : IDisposable
{
    string Name { get; }
    /// <summary>x is [T * 48], row-major. The recurrent state starts at zero at both ends.</summary>
    ForwardResult Forward(ReadOnlySpan<float> x);
}

/// <summary>
/// PossessionNet, the forward pass, in plain C#. The same computation as the PyTorch module the
/// weights were written for:
///
///   a = GELU(inp.W x + inp.b)                     c = conv over 9 frames of a      h = a + relu(c)
///   two layers of bidirectional GRU over h        m = GELU(head1 [qf, qb, h])      logits = head2 m
///
/// Rows are stored once in a form that suits them: a row with few nonzero weights is a short list
/// of (index, value) pairs, a full row is a span for TensorPrimitives.Dot. A unit whose weights
/// and biases are all zero outputs exactly zero in this architecture, so it is skipped; for a
/// hand-written network that leaves a few dozen units to compute.
/// </summary>
public sealed class PossessionNet : IPossessionEngine
{
    public string Name => "plain C#";
    public void Dispose() { }

    private const int H = Network.Hidden, F = Network.Features, K = Network.Kernel;

    private readonly Rows _inp, _conv, _head1, _head2;
    private readonly float[] _inpB, _convB, _head1B, _head2B;
    private readonly Gru _g1f, _g1b, _g2f, _g2b;

    public PossessionNet(Network net)
    {
        _inp = new Rows(net["inp.0.weight"].Data, H, F);
        _inpB = net["inp.0.bias"].Data;
        _conv = new Rows(net["conv.weight"].Data, H, H * K);
        _convB = net["conv.bias"].Data;
        _head1 = new Rows(net["head.0.weight"].Data, H, 3 * H);
        _head1B = net["head.0.bias"].Data;
        _head2 = new Rows(net["head.2.weight"].Data, 3, H);
        _head2B = net["head.2.bias"].Data;
        _g1f = new Gru(net, 0, "", H);
        _g1b = new Gru(net, 0, "_reverse", H);
        _g2f = new Gru(net, 1, "", 2 * H);
        _g2b = new Gru(net, 1, "_reverse", 2 * H);
    }

    /// <summary>x is [T * 48], row-major. The recurrent state starts at zero at both ends.</summary>
    public ForwardResult Forward(ReadOnlySpan<float> x)
    {
        int T = x.Length / F;
        var result = new ForwardResult(T);

        // a = GELU(inp.W x + inp.b)
        var a = new float[T * H];
        foreach (int u in _inp.Active(_inpB))
            for (int t = 0; t < T; t++)
                a[t * H + u] = Gelu(_inp.Dot(u, x.Slice(t * F, F)) + _inpB[u]);

        // h = a + relu(conv(a)): conv.W[u, i, k] reads unit i of a at frame t + k - 4, zero outside
        var h = (float[])a.Clone();
        var window = new float[H * K];                       // laid out [i * K + k], like conv.W[u]
        var convActive = _conv.Active(_convB);
        if (convActive.Length > 0)
            for (int t = 0; t < T; t++)
            {
                for (int k = 0; k < K; k++)
                {
                    int s = t + k - K / 2;
                    for (int i = 0; i < H; i++) window[i * K + k] = s >= 0 && s < T ? a[s * H + i] : 0f;
                }
                foreach (int u in convActive)
                    h[t * H + u] += MathF.Max(0f, _conv.Dot(u, window) + _convB[u]);
            }

        // layer 1, both directions
        var g1f = _g1f.Run(h, H, T, reverse: false);
        var g1b = _g1b.Run(h, H, T, reverse: true);
        result.Gru1F = g1f;
        result.Gru1B = g1b;

        // layer 2 reads [gru1f | gru1b]
        var g1 = new float[T * 2 * H];
        for (int t = 0; t < T; t++)
        {
            g1f.AsSpan(t * H, H).CopyTo(g1.AsSpan(t * 2 * H, H));
            g1b.AsSpan(t * H, H).CopyTo(g1.AsSpan(t * 2 * H + H, H));
        }
        var qf = _g2f.Run(g1, 2 * H, T, reverse: false);
        var qb = _g2b.Run(g1, 2 * H, T, reverse: true);

        // m = GELU(head1 [qf, qb, h] + b), logits = head2 m + b
        var cat = new float[3 * H];
        var m = new float[H];
        var headActive = _head1.Active(_head1B);
        for (int t = 0; t < T; t++)
        {
            qf.AsSpan(t * H, H).CopyTo(cat);
            qb.AsSpan(t * H, H).CopyTo(cat.AsSpan(H));
            h.AsSpan(t * H, H).CopyTo(cat.AsSpan(2 * H));
            foreach (int u in headActive) m[u] = Gelu(_head1.Dot(u, cat) + _head1B[u]);
            for (int c = 0; c < 3; c++) result.Logits[t * 3 + c] = _head2.Dot(c, m) + _head2B[c];
        }
        return result;
    }

    /// <summary>GELU(v) = v * Phi(v), the exact form PyTorch uses.</summary>
    public static float Gelu(float v) => (float)(0.5 * v * (1.0 + Erf(v / Math.Sqrt(2.0))));

    // Abramowitz and Stegun 7.1.26; absolute error below 1.5e-7, far inside float32 noise here.
    private static double Erf(double x)
    {
        double sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
        return sign * (1.0 - poly * Math.Exp(-x * x));
    }

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    /// <summary>The rows of one weight matrix, each stored sparse or dense, whichever is cheaper.</summary>
    private sealed class Rows
    {
        private readonly float[] _dense;
        private readonly int _cols;
        private readonly int[]?[] _idx;
        private readonly float[]?[] _val;
        private readonly bool[] _zero;

        public Rows(float[] data, int rows, int cols)
        {
            (_dense, _cols) = (data, cols);
            _idx = new int[rows][];
            _val = new float[rows][];
            _zero = new bool[rows];
            for (int r = 0; r < rows; r++)
            {
                var row = data.AsSpan(r * cols, cols);
                int nnz = 0;
                foreach (float v in row) if (v != 0f) nnz++;
                _zero[r] = nnz == 0;
                if (nnz == 0 || nnz * 4 > cols) continue;         // empty, or dense enough for SIMD
                var (idx, val) = (new int[nnz], new float[nnz]);
                for (int c = 0, n = 0; c < cols; c++)
                    if (row[c] != 0f) (idx[n], val[n++]) = (c, row[c]);
                (_idx[r], _val[r]) = (idx, val);
            }
        }

        public bool IsZero(int row) => _zero[row];

        /// <summary>Rows that can produce something: a nonzero weight or a nonzero bias.</summary>
        public int[] Active(float[] bias) =>
            [.. Enumerable.Range(0, _zero.Length).Where(r => !_zero[r] || bias[r] != 0f)];

        public float Dot(int row, ReadOnlySpan<float> x)
        {
            if (_zero[row]) return 0f;
            var idx = _idx[row];
            if (idx is null) return TensorPrimitives.Dot(_dense.AsSpan(row * _cols, _cols), x);
            var val = _val[row]!;
            float sum = 0f;
            for (int n = 0; n < idx.Length; n++) sum += val[n] * x[idx[n]];
            return sum;
        }
    }

    /// <summary>One direction of one GRU layer. PyTorch stacks the gates as reset, update, new.</summary>
    private sealed class Gru
    {
        private readonly Rows _wi, _wh;
        private readonly float[] _bi, _bh;
        private readonly int[] _active;

        public Gru(Network net, int layer, string suffix, int nIn)
        {
            _wi = new Rows(net[$"gru.weight_ih_l{layer}{suffix}"].Data, 3 * H, nIn);
            _wh = new Rows(net[$"gru.weight_hh_l{layer}{suffix}"].Data, 3 * H, H);
            _bi = net[$"gru.bias_ih_l{layer}{suffix}"].Data;
            _bh = net[$"gru.bias_hh_l{layer}{suffix}"].Data;
            // A unit with nothing set stays at 0: r = z = 1/2, n = tanh(0) = 0, s = s_prev / 2 = 0.
            _active = [.. Enumerable.Range(0, H).Where(u => Enumerable.Range(0, 3).Any(g =>
                !_wi.IsZero(g * H + u) || !_wh.IsZero(g * H + u) || _bi[g * H + u] != 0f || _bh[g * H + u] != 0f))];
        }

        public float[] Run(float[] input, int nIn, int T, bool reverse)
        {
            var output = new float[T * H];
            var (state, next) = (new float[H], new float[H]);
            for (int step = 0; step < T; step++)
            {
                int t = reverse ? T - 1 - step : step;
                var x = input.AsSpan(t * nIn, nIn);
                foreach (int u in _active)
                {
                    float r = Sigmoid(_wi.Dot(u, x) + _bi[u] + _wh.Dot(u, state) + _bh[u]);
                    float z = Sigmoid(_wi.Dot(H + u, x) + _bi[H + u] + _wh.Dot(H + u, state) + _bh[H + u]);
                    float n = MathF.Tanh(_wi.Dot(2 * H + u, x) + _bi[2 * H + u]
                                         + r * (_wh.Dot(2 * H + u, state) + _bh[2 * H + u]));
                    next[u] = (1f - z) * n + z * state[u];
                }
                (state, next) = (next, state);
                state.CopyTo(output.AsSpan(t * H, H));
            }
            return output;
        }
    }
}
