from dataclasses import dataclass
from typing import Iterable, Self

from schale import literal, localization
from schale.cache_control import cache_collection
from schale.format_reward import format_reward
from schale.schema.group import GroupEntry
from schale.schema.stages import (
    Stage,
)
from schale.schema.stages import Reward


_stage_cache: dict[int, Stage] | None = None
_group_cache: dict[int, GroupEntry] = {}


@dataclass(slots=True)
class RewardMultipliers:
    """Multipliers for the rewards of a stage."""

    campaign_normal: float = 1.0
    campaign_hard: float = 1.0
    bounty: float = 1.0
    week: float = 1.0
    school: float = 1.0


@dataclass(slots=True)
class RewardExpectation:
    """Expected value for a single reward."""

    type: literal.DropType
    id: int
    expected_amount: float
    chance: float
    reward_type: literal.RewardCondition | None
    source_group_id: int | None = None

    def __str__(self) -> str:
        base = format_reward(self.id, self.type)
        fields = [
            base,
            f"exp={self.expected_amount:.2f}",
            f"chance={self.chance * 100:.2f}%",
        ]
        if self.reward_type is not None:
            fields.append(f"condition={self.reward_type}")
        if self.source_group_id is not None:
            fields.append(f"group={self.source_group_id}")
        return " | ".join(fields)


@dataclass(slots=True)
class StageRewards:
    """A collection of rewards for a stage."""

    stage: Stage
    rewards: list[RewardExpectation]

    def __str__(self) -> str:
        def format_entry_cost(stage: Stage) -> Iterable[str]:
            for entry_cost in stage.root.EntryCost:
                cost_id, cost_amount = entry_cost
                match cost_id:
                    case 5:
                        cost_name = "AP"
                    case 22:
                        cost_name = localization.BOUNTY_TICKET_NAMES[
                            localization.USER_LANG
                        ]
                    case 23:
                        cost_name = localization.SCRIMMAGE_TICKET_NAMES[
                            localization.USER_LANG
                        ]
                    case _:
                        raise ValueError(
                            f"Unknown entry cost id: {cost_id} for stage {stage.root}"
                        )

                yield f"{cost_name} x{cost_amount}"

        costs = "\n".join(f"  - {cost}" for cost in format_entry_cost(self.stage))
        rewards = "\n".join(f"  + {reward}" for reward in self.rewards)
        return f"{self.stage.root}\n{costs}\n{rewards}"

    @classmethod
    def from_stages(
        cls,
        stage_ids: Iterable[int] | None = None,
        *,
        server: literal.Server = "Global",
        reward_conditions: Iterable[literal.RewardCondition] | None = None,
        multipliers: RewardMultipliers = RewardMultipliers(),
    ) -> list[Self]:
        """Calculate the rewards for a stage.

        Args:
            stage_ids: The ids of the stages to calculate the rewards for.
            server: The server to calculate the rewards for.
            reward_conditions: The conditions that must be met to receive the rewards. If None, no conditions are applied.
            multipliers: The multipliers for the rewards.

        Returns:
            A list of StageRewards objects for each stage.
        """

        allowed_conditions = frozenset(reward_conditions or ())
        summaries: list[Self] = []

        stage_cache = cache_collection.stages
        for stage_id in stage_ids or stage_cache.keys():
            stage = stage_cache[stage_id]

            server_data = stage.root.ServerData
            if server_data and server in server_data:
                resolved_rewards = list(server_data[server].Rewards)
            else:
                resolved_rewards = list(stage.root.Rewards)

            match stage.root.Category:
                case "Campaign":
                    stage_multiplier = (
                        multipliers.campaign_normal
                        if stage.root.Difficulty == 0
                        else multipliers.campaign_hard
                    )
                case "Bounty":
                    stage_multiplier = multipliers.bounty
                case "WeekDungeon":
                    stage_multiplier = multipliers.week
                case "SchoolDungeon":
                    stage_multiplier = multipliers.school
                case _:  # pyright: ignore[reportUnnecessaryComparison]
                    raise ValueError(f"Unknown stage category: {stage.root.Category}")

            rewards: list[RewardExpectation] = []
            for reward in resolved_rewards:
                if (
                    reward.RewardType is not None
                    and reward.RewardType not in allowed_conditions
                ):
                    continue

                if reward.Type != "GachaGroup":
                    expansion = [
                        RewardExpectation(
                            type=reward.Type,
                            id=reward.Id,
                            expected_amount=_expected_amount(reward),
                            chance=_reward_chance(reward),
                            reward_type=reward.RewardType,
                        )
                    ]
                else:
                    expansion = _expand_gachagroup_reward(reward, server)

                for expanded in expansion:
                    if expanded.reward_type is None:
                        expanded.expected_amount *= stage_multiplier
                    if expanded.chance <= 0:
                        continue
                    rewards.append(expanded)
            summaries.append(cls(stage=stage, rewards=rewards))

        return summaries


def _reward_chance(reward: Reward) -> float:
    return float(reward.Chance) if reward.Chance is not None else 1.0


def _expected_amount(reward: Reward) -> float:
    if reward.AmountMin is not None and reward.AmountMax is not None:
        base_amount = (reward.AmountMin + reward.AmountMax) / 2.0
    elif reward.Amount is not None:
        base_amount = float(reward.Amount)
    else:
        base_amount = 1.0
    return base_amount * _reward_chance(reward)


def _expand_gachagroup_reward(
    reward: Reward, server: literal.Server
) -> list[RewardExpectation]:
    group_entry = cache_collection.groups[reward.Id]
    if server == "Global" and group_entry.ItemsGlobal:
        items = list(group_entry.ItemsGlobal)
    elif server == "Cn" and group_entry.ItemsCn:
        items = list(group_entry.ItemsCn)
    else:
        items = list(group_entry.Items)

    if not items:
        return []
    expected_rolls = _expected_amount(reward)
    if expected_rolls == 0:
        return []
    group_results: list[RewardExpectation] = []
    stage_chance = _reward_chance(reward)
    for item in items:
        avg_amount = (item.AmountMin + item.AmountMax) / 2.0
        expected_amount = expected_rolls * avg_amount * item.Chance
        group_results.append(
            RewardExpectation(
                type=item.Type,
                id=item.Id,
                expected_amount=expected_amount,
                chance=min(1.0, stage_chance * item.Chance),
                reward_type=reward.RewardType,
                source_group_id=reward.Id,
            )
        )
    return group_results


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    for stage_rewards in StageRewards.from_stages(
        server="Global",
        stage_ids=[1011101],
        reward_conditions=("FirstClear", "ThreeStar"),
        multipliers=RewardMultipliers(campaign_normal=1.2, campaign_hard=1.5),
    ):
        print("")
        print(stage_rewards)
