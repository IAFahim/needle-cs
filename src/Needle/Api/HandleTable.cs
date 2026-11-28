using System;

namespace Needle;

/// <summary>
/// Slot plus generation table behind the public handle structs. A handle that outlives its object, or a
/// <c>default</c> handle, fails cleanly instead of touching freed memory.
/// </summary>
internal static class HandleTable<T> where T : class
{
    struct Entry
    {
        public T? Value;
        public int Generation;
    }

    static readonly object Gate = new();
    static Entry[] entries = new Entry[8];
    static int[] free = new int[8];
    static int freeCount;
    static int used;

    public static void Add(T value, out int slot, out int generation)
    {
        lock (Gate)
        {
            if (freeCount > 0) slot = free[--freeCount];
            else
            {
                if (used == entries.Length)
                {
                    var grown = new Entry[entries.Length * 2];
                    Array.Copy(entries, grown, entries.Length);
                    entries = grown;
                }
                slot = used++;
            }
            ref var entry = ref entries[slot];
            if (entry.Generation <= 0) entry.Generation = 1;
            entry.Value = value;
            generation = entry.Generation;
        }
    }

    public static bool TryGet(int slot, int generation, out T value)
    {
        var table = entries;
        if (generation > 0 && (uint)slot < (uint)table.Length)
        {
            ref var entry = ref table[slot];
            var found = entry.Value;
            if (entry.Generation == generation && found != null)
            {
                value = found;
                return true;
            }
        }
        value = null!;
        return false;
    }

    public static T Get(int slot, int generation, string typeName)
    {
        if (TryGet(slot, generation, out var value)) return value;
        throw new ObjectDisposedException(typeName, generation == 0
            ? $"this {typeName} was never opened; it is a default value"
            : $"this {typeName} has been disposed");
    }

    public static T? Remove(int slot, int generation)
    {
        lock (Gate)
        {
            if (generation <= 0 || (uint)slot >= (uint)entries.Length) return null;
            ref var entry = ref entries[slot];
            if (entry.Generation != generation || entry.Value == null) return null;
            var value = entry.Value;
            entry.Value = null;
            entry.Generation = entry.Generation == int.MaxValue ? 1 : entry.Generation + 1;
            if (freeCount == free.Length) Array.Resize(ref free, free.Length * 2);
            free[freeCount++] = slot;
            return value;
        }
    }
}
