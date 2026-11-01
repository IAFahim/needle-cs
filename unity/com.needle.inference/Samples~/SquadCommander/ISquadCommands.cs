using Needle.Tools;

namespace Needle.Samples.SquadCommander
{
    public enum Item { Potion, Sword, Shield, Arrow, Gold }
    public enum Formation { Line, Wedge, Circle, Column }

    // The whole contract between the language model and your game. SquadCommands.g.cs is what the
    // Needle.Generators source generator produces from this interface: a JSON schema, a grammar
    // that makes malformed calls impossible, and a blittable command struct. Regenerate it when
    // you change the interface (see samples/SquadCommander in the repository).
    [NeedleTools]
    public partial interface ISquadCommands
    {
        [Tool("Move the selected unit to a map position.")] void MoveTo([Param("Map x coordinate")] float x, [Param("Map y coordinate")] float y);
        [Tool("Attack a target.")] void Attack([Param("Name of the enemy or building")] Utf8Text64 target);
        [Tool("Give items to a party member.")] void GiveItem(Item item, [Param("How many")] int count, [Param("Party member name")] Utf8Text32 to);
        [Tool("Change the squad formation.")] void SetFormation(Formation formation = Formation.Line);
        [Tool("Follow a character.")] void Follow([Param("Who to follow")] Utf8Text32 target);
        [Tool("Stop all current actions.")] void Stop();
    }
}
