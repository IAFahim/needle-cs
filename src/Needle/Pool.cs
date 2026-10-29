using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace Needle;

internal sealed unsafe class Pool : IDisposable
{
    const int SpinsBeforeParking = 40_000;
    const int CacheLine = 64;

    [StructLayout(LayoutKind.Explicit, Size = 4 * CacheLine)]
    struct Control
    {
        [FieldOffset(0)] public long Ticket;
        [FieldOffset(CacheLine)] public int Done;
        [FieldOffset(2 * CacheLine)] public delegate*<void*, int, void> Function;
        [FieldOffset(2 * CacheLine + 8)] public void* Context;
        [FieldOffset(2 * CacheLine + 16)] public int Count;
        [FieldOffset(3 * CacheLine)] public int Sleeping;
        [FieldOffset(3 * CacheLine + 4)] public int Stopping;
    }

    readonly Control* control;
    readonly Thread[] workers;
    readonly int[] cores;
    readonly SemaphoreSlim wake = new(0);

    public int Threads => workers.Length + 1;

    public Pool(int threads)
    {
        control = (Control*)Mem.AlignedAlloc((nuint)sizeof(Control), CacheLine);
        Mem.Clear(control, (nuint)sizeof(Control));
        cores = Libc.FastCores(threads);
        if (cores.Length > 0) Libc.PinToCpu(cores[0]);
        workers = new Thread[Math.Max(0, threads - 1)];
        for (var i = 0; i < workers.Length; i++)
        {
            var slot = i + 1;
            workers[i] = new Thread(() => Work(slot)) { IsBackground = true, Name = $"needle-{i}" };
            workers[i].Start();
        }
    }

    public void Run(delegate*<void*, int, void> function, void* context, int count)
    {
        if (workers.Length == 0 || count <= 1)
        {
            for (var i = 0; i < count; i++) function(context, i);
            return;
        }
        var generation = (Volatile.Read(ref control->Ticket) >> 32) + 1;
        Interlocked.Exchange(ref control->Ticket, (generation << 32) | int.MaxValue);
        control->Function = function;
        control->Context = context;
        control->Count = count;
        Volatile.Write(ref control->Done, 0);
        Volatile.Write(ref control->Ticket, generation << 32);
        if (Volatile.Read(ref control->Sleeping) > 0) wake.Release(Volatile.Read(ref control->Sleeping));
        Drain();
        var spin = new SpinWait();
        while (Volatile.Read(ref control->Done) < count)
#if NETSTANDARD2_1
            spin.SpinOnce();
#else
            spin.SpinOnce(-1);
#endif
    }

    bool Drain()
    {
        var claimed = false;
        for (;;)
        {
            var ticket = Volatile.Read(ref control->Ticket);
            var index = (int)ticket;
            if (index >= control->Count) return claimed;
            if (Interlocked.CompareExchange(ref control->Ticket, ticket + 1, ticket) != ticket) continue;
            control->Function(control->Context, index);
            Interlocked.Increment(ref control->Done);
            claimed = true;
        }
    }

    void Work(int slot)
    {
        if (cores.Length > 0) Libc.PinToCpu(cores[slot % cores.Length]);
        var idle = 0;
        while (Volatile.Read(ref control->Stopping) == 0)
        {
            if (Drain()) { idle = 0; continue; }
            if (++idle < SpinsBeforeParking) { Thread.SpinWait(4); continue; }
            Interlocked.Increment(ref control->Sleeping);
            if ((int)Volatile.Read(ref control->Ticket) >= control->Count && Volatile.Read(ref control->Stopping) == 0) wake.Wait();
            Interlocked.Decrement(ref control->Sleeping);
            idle = 0;
        }
    }

    public void Dispose()
    {
        Volatile.Write(ref control->Stopping, 1);
        if (workers.Length > 0) wake.Release(workers.Length);
        foreach (var worker in workers) worker.Join();
        Mem.AlignedFree(control);
    }
}
