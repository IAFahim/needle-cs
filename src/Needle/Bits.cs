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

    public static int PopCount(uint value)
    {
        value -= (value >> 1) & 0x55555555u;
        value = (value & 0x33333333u) + ((value >> 2) & 0x33333333u);
        return (int)(((value + (value >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24);
    }
#else
    public static int TrailingZeros(uint value) => System.Numerics.BitOperations.TrailingZeroCount(value);

    public static int PopCount(uint value) => System.Numerics.BitOperations.PopCount(value);
#endif
}
