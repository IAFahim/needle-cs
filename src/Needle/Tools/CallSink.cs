using System;

namespace Needle.Tools;

public interface IGrammarSink
{
    void ToolSelected(int tool);
    void PropertySelected(int property);
    void StringByte(byte value);
    void NumberByte(byte value);
    void EnumSelected(int literal);
    void BoolSelected(bool value);
    void NullSelected();
    void ValueCompleted();
    void CallCompleted();
}

public struct NoSink : IGrammarSink
{
    public void ToolSelected(int tool)
    {
    }

    public void PropertySelected(int property)
    {
    }

    public void StringByte(byte value)
    {
    }

    public void NumberByte(byte value)
    {
    }

    public void EnumSelected(int literal)
    {
    }

    public void BoolSelected(bool value)
    {
    }

    public void NullSelected()
    {
    }

    public void ValueCompleted()
    {
    }

    public void CallCompleted()
    {
    }
}

// When argument repair rewrites a turn's calls (NeedleOptions.Repair), the session replays the repaired calls
// into a copy of the sink taken before the turn and hands that copy back. A sink therefore has to be a
// self-contained value, as the generated sinks are: state held behind a reference would see both passes.
public interface ICallSink<TCommand> : IGrammarSink
{
    int CallCount { get; }
    bool TryTakeCall(out TCommand command);
}
