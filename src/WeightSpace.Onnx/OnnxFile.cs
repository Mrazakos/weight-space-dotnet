namespace WeightSpace.Onnx;

/// <summary>
/// Just enough protobuf to find the weight bytes in an ONNX file:
/// ModelProto.graph (7) -> GraphProto.initializer (5) -> TensorProto.name (8) and raw_data (9).
/// </summary>
internal static class OnnxFile
{
    /// <summary>Initializer name -> (offset, length) of its raw_data inside <paramref name="model"/>.</summary>
    public static Dictionary<string, (int Offset, int Length)> RawInitializers(byte[] model)
    {
        var found = new Dictionary<string, (int, int)>();
        foreach (var (field, start, end) in Fields(model, 0, model.Length))
        {
            if (field != 7) continue;                                       // graph
            foreach (var (gf, gs, ge) in Fields(model, start, end))
            {
                if (gf != 5) continue;                                      // initializer
                string? name = null;
                (int, int)? raw = null;
                foreach (var (tf, ts, te) in Fields(model, gs, ge))
                {
                    if (tf == 8) name = System.Text.Encoding.UTF8.GetString(model, ts, te - ts);
                    else if (tf == 9) raw = (ts, te - ts);
                }
                if (name is not null && raw is not null) found[name] = raw.Value;
            }
        }
        return found;
    }

    /// <summary>The length-delimited fields of the message in [start, end): (field number, payload start, payload end).</summary>
    private static IEnumerable<(int Field, int Start, int End)> Fields(byte[] b, int start, int end)
    {
        int pos = start;
        while (pos < end)
        {
            ulong tag = Varint(b, ref pos);
            int field = (int)(tag >> 3);
            switch ((int)(tag & 7))
            {
                case 0: Varint(b, ref pos); break;
                case 1: pos += 8; break;
                case 5: pos += 4; break;
                case 2:
                    int len = (int)Varint(b, ref pos);
                    yield return (field, pos, pos + len);
                    pos += len;
                    break;
                default: throw new InvalidDataException($"unsupported protobuf wire type at {pos}");
            }
        }
    }

    private static ulong Varint(byte[] b, ref int pos)
    {
        ulong v = 0;
        for (int shift = 0; ; shift += 7)
        {
            byte x = b[pos++];
            v |= (ulong)(x & 0x7f) << shift;
            if (x < 0x80) return v;
        }
    }
}
