using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;

namespace Needle;

public readonly struct VoiceHostOptions
{
    public string WhistleModelPath { get; init; }
    public string NeedleModelPath { get; init; }
    public string ToolsJson { get; init; }
    public NeedleOptions? Session { get; init; }
    public int WhistleThreads { get; init; }
    public string? Language { get; init; }
    public int ReplyCapacity { get; init; }
    public int WorkCapacity { get; init; }
}

public struct VoiceWork
{
    public int Slot;
    public int Length;
    public ulong Utterance;
}

public struct VoiceReply
{
    public ulong Utterance;
    public ulong Ticket;
    public Reply Reply;
    public CallsUtf8 Calls;
    public TextUtf8 Reasoning;
    public TextUtf8 Response;
    public TextUtf8 Transcript;
    public int TranscriptLength;
    public bool TranscriptTruncated;
    public bool QueryTruncated;
}

// Audio in, reply out. The audio thread appends PCM with WriteAudio; the game thread
// snapshots an utterance with TryEndUtterance and polls TryReceive for voice replies.
// Text queries go through the hub NeedleChannel (VoiceHost.Channel); every reply the
// hub produces is forwarded here, voice replies with their transcript attached and
// anything else through TryReceiveDirect, so the game never races the worker.
public readonly unsafe struct VoiceChannel
{
    internal const int AudioCap = Whistle.MaxSeconds * Wav.SampleRate;
    const int SlotCount = 2;
    const int MaxTranscript = 2048;

    internal readonly byte* block;
    readonly long* sleeping;
    readonly long* sealedFlag;
    readonly long* writers;
    readonly long* head;
    readonly long* counter;
    readonly long* slotBusy;
    readonly int* slotLength;
    readonly RingOps.Header* work;
    readonly RingOps.Header* replies;
    readonly RingOps.Header* direct;
    readonly float* fifo;
    readonly float* slots;

    internal VoiceChannel(byte* block, long* sleeping, long* sealedFlag, long* writers, long* head, long* counter, long* slotBusy, int* slotLength, RingOps.Header* work, RingOps.Header* replies, RingOps.Header* direct, float* fifo, float* slots)
    {
        this.block = block;
        this.sleeping = sleeping;
        this.sealedFlag = sealedFlag;
        this.writers = writers;
        this.head = head;
        this.counter = counter;
        this.slotBusy = slotBusy;
        this.slotLength = slotLength;
        this.work = work;
        this.replies = replies;
        this.direct = direct;
        this.fifo = fifo;
        this.slots = slots;
    }

    internal static VoiceChannel Allocate(int workCapacity, int replyCapacity)
    {
        var workCells = PowerOfTwo(Math.Max(1, workCapacity));
        var replyCells = PowerOfTwo(Math.Max(1, replyCapacity));
        var at = 128;
        var workHeader = at; at += Align(sizeof(RingOps.Header));
        var workCellOffset = at; at += workCells * sizeof(RingOps.Cell<VoiceWork>);
        var replyHeader = Align(at); at += Align(sizeof(RingOps.Header));
        var replyCellOffset = at; at += replyCells * sizeof(RingOps.Cell<VoiceReply>);
        var directHeader = Align(at); at += Align(sizeof(RingOps.Header));
        var directCellOffset = at; at += replyCells * sizeof(RingOps.Cell<ChannelReply>);
        var fifoOffset = Align(at); at += AudioCap * sizeof(float);
        var slotOffset = Align(at); at += SlotCount * AudioCap * sizeof(float);
        var total = Align(at) + 128;
        var block = (byte*)Mem.AlignedAlloc((nuint)total, 64);
        Mem.Clear(block, (nuint)total);
        var workPtr = (RingOps.Header*)(block + workHeader);
        var replyPtr = (RingOps.Header*)(block + replyHeader);
        var directPtr = (RingOps.Header*)(block + directHeader);
        RingOps.Initialize<VoiceWork>(workPtr, workCells);
        RingOps.Initialize<VoiceReply>(replyPtr, replyCells);
        RingOps.Initialize<ChannelReply>(directPtr, replyCells);
        return new VoiceChannel(block,
            (long*)(block + 0), (long*)(block + 8), (long*)(block + 16), (long*)(block + 24),
            (long*)(block + 32), (long*)(block + 40), (int*)(block + 56),
            workPtr, replyPtr, directPtr,
            (float*)(block + fifoOffset), (float*)(block + slotOffset));
    }

    public void Free()
    {
        if (block != null) Mem.AlignedFree(block);
    }

    internal void SetWake(delegate*<void*, void> wake, void* state)
    {
        Volatile.Write(ref *(nint*)(block + 64), (nint)(void*)wake);
        Volatile.Write(ref *(nint*)(block + 72), (nint)state);
    }

    internal void SetSleeping(long value) => Volatile.Write(ref *sleeping, value);

    // Audio thread, lock-free. Drops the chunk (false) while a snapshot is in flight
    // or the FIFO is full.
    public bool WriteAudio(ReadOnlySpan<float> pcm)
    {
        if (pcm.Length == 0) return true;
        if (Volatile.Read(ref *sealedFlag) != 0) return false;
        Interlocked.Increment(ref *writers);
        try
        {
            if (Volatile.Read(ref *sealedFlag) != 0) return false;
            var current = Volatile.Read(ref *head);
            if (current + pcm.Length > AudioCap) return false;
            pcm.CopyTo(new Span<float>(fifo + current, AudioCap - (int)current));
            Volatile.Write(ref *head, current + pcm.Length);
            return true;
        }
        finally { Interlocked.Decrement(ref *writers); }
    }

    // Game thread: snapshots the FIFO into a free utterance slot and queues it for the
    // worker. Returns false when there is nothing to transcribe or no slot is free.
    public bool TryEndUtterance(out ulong utterance)
    {
        utterance = 0;
        if (Volatile.Read(ref *head) == 0) return false;
        if (Interlocked.CompareExchange(ref *sealedFlag, 1, 0) != 0) return false;
        var spin = new SpinWait();
        while (Volatile.Read(ref *writers) != 0) spin.SpinOnce();
        var slot = -1;
        for (var i = 0; i < SlotCount; i++)
            if (Interlocked.CompareExchange(ref slotBusy[i], 1, 0) == 0) { slot = i; break; }
        var samples = (int)Math.Min(Volatile.Read(ref *head), AudioCap);
        if (slot < 0 || samples == 0)
        {
            if (slot >= 0) Volatile.Write(ref slotBusy[slot], 0);
            Volatile.Write(ref *sealedFlag, 0);
            return false;
        }
        new ReadOnlySpan<float>(fifo, samples).CopyTo(new Span<float>(slots + (long)slot * AudioCap, samples));
        slotLength[slot] = samples;
        Volatile.Write(ref *head, 0);
        Volatile.Write(ref *sealedFlag, 0);
        utterance = (ulong)Interlocked.Increment(ref *counter);
        var job = new VoiceWork { Slot = slot, Length = samples, Utterance = utterance };
        if (!RingOps.TryEnqueue(work, in job, out _))
        {
            Volatile.Write(ref slotBusy[slot], 0);
            return false;
        }
        SignalWake();
        return true;
    }

    public bool TryReceive(out VoiceReply reply) => RingOps.TryDequeue(replies, out reply, out _);

    public bool TryReceiveDirect(out ChannelReply reply) => RingOps.TryDequeue(direct, out reply, out _);

    public bool HasCompletedReply => RingOps.HasItems<VoiceReply>(replies);

    public bool HasDirectReply => RingOps.HasItems<ChannelReply>(direct);

    internal bool TryTakeWork(out VoiceWork job, out long position) => RingOps.TryDequeue(work, out job, out position);

    internal ReadOnlySpan<float> SlotPcm(int slot, int length) => new(slots + (long)slot * AudioCap, length);

    internal void FreeSlot(int slot) => Volatile.Write(ref slotBusy[slot], 0);

    internal bool TryPublishReply(in VoiceReply reply) => RingOps.TryEnqueue(replies, in reply, out _);

    internal bool TryPublishDirect(in ChannelReply reply) => RingOps.TryEnqueue(direct, in reply, out _);

    void SignalWake()
    {
        var function = Volatile.Read(ref *(nint*)(block + 64));
        if (function == IntPtr.Zero) return;
        if (Interlocked.CompareExchange(ref *sleeping, 0, 0) == 0) return;
        var state = Volatile.Read(ref *(nint*)(block + 72));
        ((delegate*<void*, void>)(void*)function)((void*)state);
    }

    static int PowerOfTwo(int requested)
    {
        var capacity = 1;
        while (capacity < requested) capacity <<= 1;
        return capacity;
    }

    static int Align(int value) => (value + 7) & ~7;
}

public sealed unsafe class VoiceHost : IDisposable
{
    readonly object gate = new();
    Model? whistleModel;
    Whistle? whistle;
    NeedleHost? hub;
    NeedleChannel hubChannel;
    VoiceChannel channel;
    byte[]? textScratch;
    WordSlice[]? wordScratch;
    List<string>? keywords;
    string? transcribeLanguage;
    ulong[]? pendingUtterance;
    ulong[]? pendingTicket;
    byte[]? pendingText;
    int[]? pendingLength;
    int[]? pendingTotal;
    int pendingHead;
    int pendingCount;
    ThreadPolicy policy;
    Thread? worker;
    GCHandle self;
    int stopping;
    int started;
    int disposed;

    static int PendingCapacity => 64;

    public VoiceChannel Start(in VoiceHostOptions options)
    {
        if (Interlocked.Exchange(ref started, 1) != 0 || Volatile.Read(ref disposed) != 0)
            throw new InvalidOperationException("VoiceHost was already started");
        try
        {
            if (string.IsNullOrEmpty(options.WhistleModelPath)) throw new ArgumentException("whistle model path is required");
            if (string.IsNullOrEmpty(options.NeedleModelPath)) throw new ArgumentException("needle model path is required");
            var sessionOptions = options.Session ?? new NeedleOptions();
            policy = sessionOptions.ResolvedThreading;
            whistleModel = Model.Open(options.WhistleModelPath);
            whistle = new Whistle(whistleModel, Math.Max(1, options.WhistleThreads == 0 ? 4 : options.WhistleThreads));
            textScratch = new byte[whistle.MaxTranscriptTokens * 64];
            wordScratch = new WordSlice[whistle.MaxTranscriptTokens + 1];
            keywords = VoiceSession.EnumKeywords(options.ToolsJson ?? "");
            transcribeLanguage = options.Language;
            hub = new NeedleHost();
            hubChannel = hub.Start(new HostOptions
            {
                ModelPath = options.NeedleModelPath,
                ToolsJson = options.ToolsJson ?? "",
                Session = sessionOptions,
                RequestCapacity = NeedleChannel.DefaultCapacity,
                ReplyCapacity = options.ReplyCapacity == 0 ? NeedleChannel.DefaultCapacity : options.ReplyCapacity,
            });
            channel = VoiceChannel.Allocate(
                options.WorkCapacity == 0 ? 8 : options.WorkCapacity,
                options.ReplyCapacity == 0 ? NeedleChannel.DefaultCapacity : options.ReplyCapacity);
            pendingUtterance = new ulong[PendingCapacity];
            pendingTicket = new ulong[PendingCapacity];
            pendingText = new byte[PendingCapacity * 2048];
            pendingLength = new int[PendingCapacity];
            pendingTotal = new int[PendingCapacity];
            self = GCHandle.Alloc(this);
            channel.SetWake(&WakeTrampoline, (void*)GCHandle.ToIntPtr(self));
            worker = new Thread(Loop) { IsBackground = true, Name = "needle-voice" };
            worker.Start();
            return channel;
        }
        catch
        {
            ReleaseNative();
            Interlocked.Exchange(ref started, 0);
            throw;
        }
    }

    // Hub channel for direct text queries. Submit here, receive via
    // VoiceChannel.TryReceiveDirect (voice replies arrive via TryReceive).
    public NeedleChannel Channel => hubChannel;

    bool TryProcessOne()
    {
        if (channel.TryTakeWork(out var job, out _))
        {
            ProcessJob(job);
            return true;
        }
        return false;
    }

    void ProcessJob(VoiceWork job)
    {
        var speech = whistle!.TranscribeUtf8(channel.SlotPcm(job.Slot, job.Length), textScratch!, wordScratch!, transcribeLanguage, keywords);
        channel.FreeSlot(job.Slot);
        var length = Math.Min(speech.Utf8Length, ChannelLimits.QueryCapacity);
        var ticket = 0ul;
        var spin = new SpinWait();
        while (!hubChannel.TrySubmitQuery(new ReadOnlySpan<byte>(textScratch!, 0, length), out ticket))
#if NETSTANDARD2_1
            spin.SpinOnce();
#else
            spin.SpinOnce(-1);
#endif
        var stored = Math.Min(speech.Utf8Length, 2048);
        var slot = (pendingHead + pendingCount) % PendingCapacity;
        if (pendingCount < PendingCapacity)
        {
            pendingUtterance![slot] = job.Utterance;
            pendingTicket![slot] = ticket;
            new ReadOnlySpan<byte>(textScratch!, 0, stored).CopyTo(pendingText!.AsSpan(slot * 2048));
            pendingLength![slot] = stored;
            pendingTotal![slot] = speech.Utf8Length;
            pendingCount++;
        }
        DrainReplies();
    }

    void DrainReplies()
    {
        while (hubChannel.TryReceive(out var reply)) Publish(reply);
    }

    void Publish(ChannelReply reply)
    {
        for (var i = 0; i < pendingCount; i++)
        {
            var slot = (pendingHead + i) % PendingCapacity;
            if (pendingTicket![slot] != reply.Ticket) continue;
            VoiceReply voice = default;
            voice.Utterance = pendingUtterance![slot];
            voice.Ticket = reply.Ticket;
            voice.Reply = reply.Reply;
            voice.Calls = reply.Calls;
            voice.Reasoning = reply.Reasoning;
            voice.Response = reply.Response;
            new ReadOnlySpan<byte>(pendingText!, slot * 2048, pendingLength![slot]).CopyTo(Utf8.AsSpan(ref voice.Transcript, pendingLength[slot]));
            voice.TranscriptLength = pendingLength[slot];
            voice.TranscriptTruncated = pendingLength[slot] < pendingTotal![slot];
            voice.QueryTruncated = pendingTotal[slot] > ChannelLimits.QueryCapacity;
            var spin = new SpinWait();
            while (!channel.TryPublishReply(in voice))
#if NETSTANDARD2_1
                spin.SpinOnce();
#else
                spin.SpinOnce(-1);
#endif
            pendingHead = (slot + 1) % PendingCapacity;
            pendingCount -= i + 1;
            return;
        }
        var forward = new SpinWait();
        while (!channel.TryPublishDirect(in reply))
#if NETSTANDARD2_1
            forward.SpinOnce();
#else
            forward.SpinOnce(-1);
#endif
    }

    void Loop()
    {
        OsThreads.ConfigureCurrentThread(policy.Priority, policy.Placement, -1);
        while (Volatile.Read(ref stopping) == 0)
            if (!TryProcessOne())
            {
                DrainReplies();
                if (!channel.HasCompletedReply) Park();
            }
    }

    void Park()
    {
        lock (gate)
        {
            channel.SetSleeping(1);
            var idle = 0;
            while (Volatile.Read(ref stopping) == 0 && !channel.HasCompletedReply)
            {
                Monitor.Wait(gate, 5);
                DrainReplies();
                if (++idle > 600) break;
            }
            channel.SetSleeping(0);
        }
    }

    void Pulse()
    {
        lock (gate) Monitor.PulseAll(gate);
    }

    static void WakeTrampoline(void* state) =>
        ((VoiceHost)GCHandle.FromIntPtr((nint)state).Target!).Pulse();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        Volatile.Write(ref stopping, 1);
        Pulse();
        worker?.Join();
        ReleaseNative();
        if (self.IsAllocated) self.Free();
    }

    void ReleaseNative()
    {
        if (channel.block != null)
        {
            channel.Free();
            channel = default;
        }
        hub?.Dispose();
        hub = null;
        whistle?.Dispose();
        whistle = null;
        whistleModel?.Dispose();
        whistleModel = null;
    }
}
