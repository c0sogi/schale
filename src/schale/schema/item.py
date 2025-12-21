from typing import Generic, Literal, Optional, TypeVar
from pydantic import BaseModel, RootModel, model_validator

from schale import literal
from schale import localization


_BonusAmount = tuple[int, int]
_TriBool = tuple[bool, bool, bool]

C = TypeVar("C", bound=literal.ItemCategory)


class ServerBonusData(BaseModel):
    Global: Optional[list[_BonusAmount]] = None
    Cn: Optional[list[_BonusAmount]] = None
    Jp: Optional[list[_BonusAmount]] = None


class ShiftingCraftRecipeData(BaseModel):
    Released: _TriBool
    RequireItem: tuple[int, int]
    RequireGold: int
    IngredientTag: list[str]
    IngredientExp: int


class ConsumableChoice(BaseModel):
    Type: literal.ConsumableChoiceType
    Id: int
    AmountMin: int
    AmountMax: int


class BaseItem(BaseModel, Generic[C]):
    Id: int
    IsReleased: _TriBool
    Category: C
    Rarity: literal.Rarity
    Quality: int
    Tags: list[str]
    Craftable: _TriBool
    StageDrop: _TriBool
    Shop: _TriBool
    Icon: str
    Name: str
    Desc: str

    @model_validator(mode="before")
    @classmethod
    def forbid_unknown_keys(cls, data: dict[str, object]) -> dict[str, object]:
        if unexpected_keys := set(data.keys()) - cls.model_fields.keys():
            raise ValueError(
                f"Model {cls.__name__} has unexpected keys: {unexpected_keys}"
            )
        return data

    def __str__(self) -> str:
        return f"[{localization.ITEM_TRANSLATIONS[self.Category][localization.USER_LANG]}] {' '.join(self.Name.splitlines()).strip()} (#{self.Id})"


class MaterialItem(BaseItem[Literal["Material"]]):
    CraftQuality: Optional[int] = None
    ShiftingCraftQuality: Optional[int] = None
    ShiftingCraftQualityCn: Optional[int] = None
    ShiftingCraftQualityGlobal: Optional[int] = None
    ShiftingCraftRecipe: Optional[ShiftingCraftRecipeData] = None
    SubCategory: Optional[str] = None


class CoinItem(BaseItem[Literal["Coin"]]):
    CraftQuality: Optional[int] = None
    EventBonus: Optional[ServerBonusData] = None
    EventBonusRerun: Optional[ServerBonusData] = None
    EventId: Optional[int] = None


class CharacterExpGrowthItem(BaseItem[Literal["CharacterExpGrowth"]]):
    CraftQuality: int
    ExpValue: int


class FavorItem(BaseItem[Literal["Favor"]]):
    CraftQuality: Optional[int] = None
    ExpValue: int
    ShiftingCraftQuality: Optional[int] = None


class SecretStoneItem(BaseItem[Literal["SecretStone"]]):
    pass


class CollectibleItem(BaseItem[Literal["Collectible"]]):
    EventId: Optional[int] = None


class ConsumableItem(BaseItem[Literal["Consumable"]]):
    ConsumeType: str
    GachaGroupId: Optional[int] = None
    Items: Optional[list[ConsumableChoice]] = None
    ItemsCn: Optional[list[ConsumableChoice]] = None
    ItemsGlobal: Optional[list[ConsumableChoice]] = None
    ShiftingCraftQuality: Optional[int] = None
    ShiftingCraftRecipe: Optional[ShiftingCraftRecipeData] = None


Item = RootModel[
    MaterialItem
    | CoinItem
    | CharacterExpGrowthItem
    | FavorItem
    | SecretStoneItem
    | CollectibleItem
    | ConsumableItem
]
