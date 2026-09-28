using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace EtAlii.Dllm.Core.Numerics;

/// <summary>
/// Hashes the exact bit patterns of numeric data. Two runs are reproducible if and only if their fingerprints match.
/// </summary>
public static class Fingerprint
{
    public static string Of(ReadOnlySpan<float> values)
    {
        // Normalise to little endian so big endian hosts hash the same bytes.
        var bits = new int[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            var b = BitConverter.SingleToInt32Bits(values[i]);
            bits[i] = BitConverter.IsLittleEndian ? b : System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(b);
        }
        return Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(bits.AsSpan())));
    }

    public static string Of(ReadOnlySpan<int> tokens)
    {
        var copy = tokens.ToArray();
        if (!BitConverter.IsLittleEndian)
        {
            for (var i = 0; i < copy.Length; i++)
            {
                copy[i] = System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(copy[i]);
            }
        }
        return Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(copy.AsSpan())));
    }
}
