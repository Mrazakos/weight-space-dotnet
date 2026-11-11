namespace WeightSpace.Core;

/// <summary>
/// The Viterbi decoder that turns per-frame probabilities into a label sequence. Every change of
/// label costs a penalty, which is what stops the output from flickering into hundreds of fake
/// possessions. Labels: 0 = H (home), 1 = A (away), 2 = D (dead ball).
/// </summary>
public static class Decoder
{
    /// <param name="probs">[T * 3] probabilities.</param>
    /// <param name="switchHA">cost of H to A or A to H, in log-probability units (16).</param>
    /// <param name="switchAlive">cost of any change between live play and dead ball (24).</param>
    public static byte[] Viterbi(ReadOnlySpan<float> probs, double switchHA = 16, double switchAlive = 24)
    {
        int T = probs.Length / 3;
        double[,] trans =
        {
            { 0, -switchHA, -switchAlive },
            { -switchHA, 0, -switchAlive },
            { -switchAlive, -switchAlive, 0 },
        };
        static double LogP(float p) => Math.Log(Math.Clamp(p, 1e-6f, 1f));

        var score = new double[3];
        var next = new double[3];
        var back = new byte[T * 3];
        for (int j = 0; j < 3; j++) score[j] = LogP(probs[j]);
        for (int t = 1; t < T; t++)
        {
            for (int j = 0; j < 3; j++)
            {
                int best = 0;
                double bestScore = score[0] + trans[0, j];
                for (int i = 1; i < 3; i++)
                {
                    double s = score[i] + trans[i, j];
                    if (s > bestScore) (best, bestScore) = (i, s);
                }
                back[t * 3 + j] = (byte)best;
                next[j] = bestScore + LogP(probs[t * 3 + j]);
            }
            (score, next) = (next, score);
        }
        var path = new byte[T];
        int last = 0;
        for (int j = 1; j < 3; j++) if (score[j] > score[last]) last = j;
        path[T - 1] = (byte)last;
        for (int t = T - 1; t > 0; t--) path[t - 1] = back[t * 3 + path[t]];
        return path;
    }
}
