using Needle.Tools;

namespace Needle.Samples;

public enum Item { Potion, Sword, Shield, Arrow, Gold }
public enum Formation { Line, Wedge, Circle, Column }

[NeedleTools]
public partial interface INpcCommands
{
    [Tool("Move the selected unit to a map position.")] void MoveTo(float x, float y);
    [Tool("Attack a target entity.")] void Attack(int targetId);
    [Tool("Give items to a party member.")] void GiveItem(Item item, int count);
    [Tool("Change the squad formation.")] void SetFormation(Formation formation = Formation.Line);
}
