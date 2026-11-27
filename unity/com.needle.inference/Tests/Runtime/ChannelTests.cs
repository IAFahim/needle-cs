using System;
using System.Text;
using NUnit.Framework;

namespace Needle.Unity.Tests
{
    // The unmanaged channel the game thread and Burst jobs talk through, without a model: the test
    // plays the worker side itself.
    public sealed unsafe class ChannelTests
    {
        internal static void Publish(NeedleChannel channel, ulong ticket, ReplyStatus status, string calls, float confidence = 0.9f)
        {
            var reply = new ChannelReply { Ticket = ticket };
            int length = calls.Length == 0 ? 0 : Encoding.UTF8.GetBytes(calls.AsSpan(), Utf8.AsSpan(ref reply.Calls, ChannelLimits.CallsCapacity));
            int callCount = calls.Length == 0 ? 0 : 1;
            reply.Reply = new Reply(status, FailureReason.None, ReplyFlags.None, confidence, confidence, confidence,
                callCount, length, 0, 0, 0, 0, 0, 0, 0);
            Assert.IsTrue(channel.TryPublishReply(in reply), "reply ring is full");
        }

        internal static string QueryText(in ChannelRequest request)
        {
            var copy = request;
            return Encoding.UTF8.GetString(Utf8.AsSpan(ref copy.Utf8, copy.Utf8Length));
        }

        [Test]
        public void QueryRoundTripsThroughTheRings()
        {
            var channel = NeedleChannel.Allocate(8, 8);
            try
            {
                Assert.IsFalse(channel.HasPendingRequest);
                byte[] query = Encoding.UTF8.GetBytes("set a timer for 12 minutes");
                Assert.IsTrue(channel.TrySubmitQuery(query, out ulong ticket));
                Assert.IsTrue(channel.HasPendingRequest);

                Assert.IsTrue(channel.TryClaim(out ChannelRequest request, out ulong claimed));
                Assert.AreEqual(ticket, claimed);
                Assert.AreEqual(ChannelRequestKind.Query, request.Kind);
                Assert.AreEqual("set a timer for 12 minutes", QueryText(in request));
                Assert.IsFalse(channel.HasPendingRequest);

                Publish(channel, claimed, ReplyStatus.Called, "{\"name\":\"set_timer\",\"arguments\":{\"minutes\":12}}");
                Assert.IsTrue(channel.HasCompletedReply);
                Assert.IsTrue(channel.TryReceive(out ChannelReply reply));
                Assert.AreEqual(ticket, reply.Ticket);
                Assert.AreEqual(ReplyStatus.Called, reply.Reply.Status);
                StringAssert.Contains("set_timer", Encoding.UTF8.GetString(Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length)));
                Assert.IsFalse(channel.TryReceive(out _));
            }
            finally
            {
                channel.Free();
            }
        }

        [Test]
        public void TicketsAreMonotonicAndCancellable()
        {
            var channel = NeedleChannel.Allocate(8, 8);
            try
            {
                byte[] query = Encoding.UTF8.GetBytes("stop");
                Assert.IsTrue(channel.TrySubmitQuery(query, out ulong first));
                Assert.IsTrue(channel.TrySubmitQuery(query, out ulong second));
                Assert.Greater(second, first);
                Assert.IsFalse(channel.IsCancelled(second));
                channel.Cancel(second);
                Assert.IsTrue(channel.IsCancelled(second));
                Assert.IsFalse(channel.IsCancelled(first));
            }
            finally
            {
                channel.Free();
            }
        }

        [Test]
        public void RejectsQueriesLongerThanTheSlot()
        {
            var channel = NeedleChannel.Allocate(8, 8);
            try
            {
                byte[] tooLong = new byte[ChannelLimits.QueryCapacity + 1];
                Assert.IsFalse(channel.TrySubmitQuery(tooLong, out _));
                Assert.IsFalse(channel.HasPendingRequest);
            }
            finally
            {
                channel.Free();
            }
        }

        [Test]
        public void FullRequestRingRefusesInsteadOfBlocking()
        {
            var channel = NeedleChannel.Allocate(2, 2);
            try
            {
                byte[] query = Encoding.UTF8.GetBytes("hold");
                int accepted = 0;
                for (int i = 0; i < 16; i++)
                    if (channel.TrySubmitQuery(query, out _)) accepted++;
                Assert.Greater(accepted, 0);
                Assert.Less(accepted, 16, "a two-slot ring accepted sixteen requests");
            }
            finally
            {
                channel.Free();
            }
        }
    }
}
