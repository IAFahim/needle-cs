namespace Needle;

internal static class Bits
{
#if NETSTANDARD2_1
    static readonly byte[] DeBruijn =
    {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9,
    };

    public static int TrailingZeros(uint value) => DeBruijn[(uint)((value & (uint)-(int)value) * 0x077CB531u) >> 27];
#else
    public static int TrailingZeros(uint value) => System.Numerics.BitOperations.TrailingZeroCount(value);
#endif
}
