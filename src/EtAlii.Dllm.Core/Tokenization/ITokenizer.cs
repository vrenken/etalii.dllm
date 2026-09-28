namespace EtAlii.Dllm.Core.Tokenization;

public interface ITokenizer
{
    int VocabularySize { get; }

    int EndOfSequence { get; }

    IReadOnlyList<int> Encode(string text);

    string Decode(IEnumerable<int> tokens);
}
