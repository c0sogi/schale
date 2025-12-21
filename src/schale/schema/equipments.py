from typing import Optional, Self

from pydantic import BaseModel, model_validator

from schale import literal
from schale import localization

_TriBool = tuple[bool, bool, bool]
_StatRange = tuple[int, int]
_RecipeRequirement = tuple[int, int]


class ShopEntry(BaseModel):
    ShopCategory: literal.ShopCategory
    Released: _TriBool
    Amount: int
    CostType: literal.ShopCostType
    CostId: int
    CostAmount: int


class Equipment(BaseModel):
    Id: int
    Category: literal.EquipmentCategory
    Rarity: literal.Rarity
    Tier: int
    Icon: str
    Shops: list[ShopEntry]
    Name: str
    Desc: str
    IsReleased: _TriBool
    MaxLevel: int
    StatType: list[str]
    StatValue: list[_StatRange]
    LevelUpFeedExp: Optional[int] = None
    Recipe: Optional[list[_RecipeRequirement]] = None
    RecipeCost: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def forbid_unknown_keys(cls, data: dict[str, object]) -> dict[str, object]:
        if unexpected_keys := set(data.keys()) - cls.model_fields.keys():
            raise ValueError(
                f"Model {cls.__name__} has unexpected keys: {unexpected_keys}"
            )
        return data

    @model_validator(mode="after")
    def validate_stat_alignment(self) -> Self:
        if len(self.StatType) != len(self.StatValue):
            raise ValueError(
                f"StatType/StatValue length mismatch: {len(self.StatType)} "
                f"vs {len(self.StatValue)}"
            )
        if (self.Recipe is None) != (self.RecipeCost is None):
            raise ValueError("Recipe and RecipeCost must be provided together")
        return self

    def __str__(self) -> str:
        tier_name = f"Tier {self.Tier} " if self.Tier > 0 else ""
        return f"[{tier_name}{localization.EQUIPMENT_TRANSLATIONS[self.Category][localization.USER_LANG]}] {self.Name} (#{self.Id})"
