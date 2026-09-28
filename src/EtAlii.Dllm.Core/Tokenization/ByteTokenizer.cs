using System.Text;

namespace EtAlii.Dllm.Core.Tokenization;

/// <summary>
/// Byte-level tokenizer: one token per UTF-8 byte plus an end-of-sequence token. A placeholder until the BPE
/// tokenizer from the roadmap lands, but already lossless and fully deterministic.
/// </summary>
public sealed class ByteTokenizer : ITokenizer
{
    public int VocabularySize => 257;

    public int EndOfSequence => 256;

    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var bytes = Encoding.UTF8.GetBytes(text);
        var tokens = new int[bytes.Length];
        for (var i = 0; i < bytes.Length; i++)
        {
            tokens[i] = bytes[i];
        }
        return tokens;
    }

    public string Decode(IEnumerable<int> tokens)
    {
        var bytes = tokens.Where(t => t is >= 0 and < 256).Select(t => (byte)t).ToArray();
        return Encoding.UTF8.GetString(bytes);
    }
}
