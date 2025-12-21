from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, RootModel, model_validator

from schale import literal
from schale import localization

_EntryCost = tuple[int, int]
_StarCondition = tuple[int, int]
_ChallengeCondition = tuple[Literal["Turns"], int]

S = TypeVar("S", bound=literal.StageCategory)


class Reward(BaseModel):
    Type: literal.Reward
    Id: int
    Amount: Optional[int] = None
    AmountMin: Optional[int] = None
    AmountMax: Optional[int] = None
    Chance: Optional[float] = None
    RewardType: Optional[literal.RewardType] = None


class Formation(BaseModel):
    Id: int
    LevelMinion: int
    GradeMinion: int
    EnemyList: list[int]
    MapIcon: Optional[str] = None
    MoveType: Optional[str] = None
    UnitGrade: Optional[literal.UnitGrade] = None


class HexaMapTile(BaseModel):
    Pos: tuple[int, int]  # [x, y]
    Start: Optional[bool] = None
    Formation: Optional[int] = None


class ServerDataRewards(BaseModel):
    Rewards: list[Reward]


class BaseStage(BaseModel, Generic[S]):
    Id: int
    Category: S
    EntryCost: list[_EntryCost]
    StarCondition: _StarCondition
    Rewards: list[Reward]

    Terrain: literal.Terrain
    Level: int
    ArmorTypes: list[int]
    ServerData: Optional[dict[literal.Server, ServerDataRewards]] = None
    Formations: list[Formation]

    @model_validator(mode="before")
    @classmethod
    def forbid_unknown_keys(cls, data: dict[str, object]) -> dict[str, object]:
        if unexpected_keys := set(data.keys()) - cls.model_fields.keys():
            raise ValueError(
                f"Model {cls.__name__} has unexpected keys: {unexpected_keys}"
            )
        return data


class CampaignStage(BaseStage[Literal["Campaign"]]):
    Name: str
    Area: int
    Stage: int | Literal["A"]
    Difficulty: int

    ChallengeCondition: list[_ChallengeCondition]
    HexaMap: Optional[list[HexaMapTile]] = None

    def __str__(self) -> str:
        return f"[{'Normal' if self.Difficulty == 0 else 'Hard'} {self.Area}-{self.Stage}] {self.Name} (#{self.Id})"


class BountyStage(BaseStage[Literal["Bounty"]]):
    Type: literal.BountyStageType
    Stage: int

    def __str__(self) -> str:
        type_name = localization.BOUNTY_TRANSLATIONS[self.Type][localization.USER_LANG]
        stage_letter = chr(self.Stage + 64)
        return f"[{localization.CATEGORY_TRANSLATIONS['Bounty'][localization.USER_LANG]} - {type_name}] {type_name} {stage_letter} (#{self.Id})"


class WeekDungeonStage(BaseStage[Literal["WeekDungeon"]]):
    Type: literal.WeekDungeonStageType
    Stage: int

    def __str__(self) -> str:
        type_name = localization.WEEK_DUNGEON_TRANSLATIONS[self.Type][
            localization.USER_LANG
        ]
        stage_letter = chr(self.Stage + 64)
        return f"[{localization.CATEGORY_TRANSLATIONS['WeekDungeon'][localization.USER_LANG]} - {type_name}] {type_name} {stage_letter} (#{self.Id})"


class SchoolDungeonStage(BaseStage[Literal["SchoolDungeon"]]):
    Type: literal.SchoolDungeonStageType
    Stage: int

    def __str__(self) -> str:
        type_name = localization.SCHOOL_DUNGEON_TRANSLATIONS[self.Type][
            localization.USER_LANG
        ]
        stage_letter = chr(self.Stage + 64)
        return f"[{localization.CATEGORY_TRANSLATIONS['SchoolDungeon'][localization.USER_LANG]} - {type_name}] {type_name} {stage_letter} (#{self.Id})"


Stage = RootModel[CampaignStage | BountyStage | WeekDungeonStage | SchoolDungeonStage]
