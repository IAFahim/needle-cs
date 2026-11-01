using Needle;
using Unity.Collections;
using Unity.Entities;

namespace Needle
{
    public struct NeedleAsk : IComponentData
    {
        public FixedString512Bytes Query;
    }
    public struct NeedlePending : IComponentData, IEnableableComponent { }
    public struct NeedleOutcome : IComponentData
    {
        public ReplyStatus Status;
        public float Confidence;
        public int CallCount;
        public CallsUtf8 Calls;
    }
    public struct NeedleChannelHandle : IComponentData
    {
        public NeedleChannel Channel;
        public nint Identity;
    }
}
