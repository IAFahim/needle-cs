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

public interface ICallSink<TCommand> : IGrammarSink
{
    int CallCount { get; }
    bool TryTakeCall(out TCommand command);
}
