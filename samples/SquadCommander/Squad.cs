using System;
using Needle.Tools;

namespace Needle.Samples.SquadCommander;

public enum Item { Potion, Sword, Shield, Arrow, Gold }
public enum Formation { Line, Wedge, Circle, Column }

// The whole contract between the language model and your game. The source generator turns this
// into a JSON schema, a grammar that makes malformed calls impossible, and a blittable command
// struct, so nothing below the game thread ever parses JSON or allocates.
[NeedleTools]
public partial interface ISquadCommands
{
    // Short, concrete descriptions matter: they are all the model knows about your game.
    [Tool("Move the selected unit to a map position.")] void MoveTo([Param("Map x coordinate")] float x, [Param("Map y coordinate")] float y);
    [Tool("Attack a target.")] void Attack([Param("Name of the enemy or building")] Utf8Text64 target);
    [Tool("Give items to a party member.")] void GiveItem(Item item, [Param("How many")] int count, [Param("Party member name")] Utf8Text32 to);
    [Tool("Change the squad formation.")] void SetFormation(Formation formation = Formation.Line);
    [Tool("Follow a character.")] void Follow([Param("Who to follow")] Utf8Text32 target);
    [Tool("Stop all current actions.")] void Stop();
}

/// <summary>Plain game state. Nothing here knows about the language model.</summary>
public sealed class Squad
{
    public float X, Y, TargetX, TargetY;
    public bool Moving;
    public Formation Formation = Formation.Line;
    public Utf8Text64 Engaging;
    public Utf8Text32 Following;
    public readonly int[] Inventory = { 5, 2, 2, 40, 100 };

    public void Tick(float dt)
    {
        if (!Moving) return;
        var dx = TargetX - X;
        var dy = TargetY - Y;
        var distance = MathF.Sqrt(dx * dx + dy * dy);
        var step = 6f * dt;
        if (distance <= step) { X = TargetX; Y = TargetY; Moving = false; return; }
        X += dx / distance * step;
        Y += dy / distance * step;
    }

    static string Text(ReadOnlySpan<byte> utf8) => System.Text.Encoding.UTF8.GetString(utf8);

    // Formatting allocates, so call it from tooling or between frames, never from the hot path.
    public override string ToString() =>
        $"at ({X:0},{Y:0}) {(Moving ? "moving" : "idle")} formation={Formation} engaging='{Text(Engaging.Span)}' following='{Text(Following.Span)}'";

    public static string Describe(in SquadCommand command) => command.Kind switch
    {
        SquadCommandKind.MoveTo => $"move to ({command.MoveTo.X:0}, {command.MoveTo.Y:0})",
        SquadCommandKind.Attack => $"attack {Text(command.Attack.Target.Span)}",
        SquadCommandKind.GiveItem => $"give {command.GiveItem.Count} {command.GiveItem.Item} to {Text(command.GiveItem.To.Span)}",
        SquadCommandKind.SetFormation => $"formation {command.SetFormation.Formation}",
        SquadCommandKind.Follow => $"follow {Text(command.Follow.Target.Span)}",
        SquadCommandKind.Stop => "stop",
        _ => command.Kind.ToString(),
    };
}

/// <summary>
/// Applies decoded commands on the game thread. A struct handler keeps dispatch free of virtual
/// calls, boxing and allocation: it only copies fixed-size values into game state.
/// </summary>
public struct SquadHandler : ISquadCommands
{
    public Squad Squad;

    public void MoveTo(float x, float y)
    {
        Squad.TargetX = x; Squad.TargetY = y; Squad.Moving = true; Squad.Following = default;
    }

    public void Attack(Utf8Text64 target) => Squad.Engaging = target;

    public void GiveItem(Item item, int count, Utf8Text32 to) => Squad.Inventory[(int)item] -= count;

    public void SetFormation(Formation formation) => Squad.Formation = formation;

    public void Follow(Utf8Text32 target)
    {
        Squad.Following = target;
        Squad.Moving = false;
    }

    public void Stop()
    {
        Squad.Moving = false; Squad.Engaging = default; Squad.Following = default;
    }
}
