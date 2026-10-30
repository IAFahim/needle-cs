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

    public static int Log2(ulong value)
    {
        var high = (uint)(value >> 32);
        if (high != 0) return 32 + Log2Lower(high);
        return Log2Lower((uint)value);
    }

    static int Log2Lower(uint value)
    {
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        return PopCount(value) - 1;
    }

#else
    public static int TrailingZeros(uint value) => System.Numerics.BitOperations.TrailingZeroCount(value);

    public static int PopCount(uint value) => System.Numerics.BitOperations.PopCount(value);

    public static int Log2(ulong value) => System.Numerics.BitOperations.Log2(value);

#endif
}
