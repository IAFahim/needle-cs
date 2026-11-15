using Needle.Tools;

namespace Needle.Samples.Playground
{
    public enum Item { Potion, Sword, Shield, Arrow, Gold }
    public enum Formation { Line, Wedge, Circle, Column }

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
