from typing import Optional

from pydantic import BaseModel, model_validator

from schale import literal
from schale.schema.item import ShiftingCraftRecipeData

_TriBool = tuple[bool, bool, bool]
_Size = tuple[int, int]
_TemplateEntry = tuple[int, int]


class Furniture(BaseModel):
    Id: int
    IsReleased: _TriBool
    Rarity: literal.Rarity
    Icon: str
    Craftable: _TriBool
    ComfortBonus: int
    Category: literal.FurnitureCategory
    Size: _Size
    Tags: list[str]
    CraftQuality: int
    ShiftingCraftQuality: Optional[int] = None
    ShiftingCraftRecipe: Optional[ShiftingCraftRecipeData] = None
    SubCategory: literal.FurnitureSubCategory
    SetGroupId: int
    Name: str
    Desc: str
    Interaction: _TriBool
    Templates: Optional[list[_TemplateEntry]] = None

    @model_validator(mode="before")
    @classmethod
    def forbid_unknown_keys(cls, data: dict[str, object]) -> dict[str, object]:
        if unexpected_keys := set(data.keys()) - cls.model_fields.keys():
            raise ValueError(
                f"Model {cls.__name__} has unexpected keys: {unexpected_keys}"
            )
        return data

    def __str__(self) -> str:
        return f"[{self.Category}] {self.Name} (#{self.Id})"
