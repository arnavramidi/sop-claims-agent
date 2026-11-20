"""What one turn may still spend on the larger model. The engine makes one per turn and hands the same one to every
state step and to the reply's composition, so the counts are kept in this one place."""

from dataclasses import dataclass


@dataclass
class TurnBudget:
    larger_settles_left: int = 1  # an unsure pick the larger model may settle
    larger_rewrites_left: int = 1  # a failed reply the larger model may rewrite
    summary_picks_left: int = 1  # the fast model's pick of one listed claim by its summary
