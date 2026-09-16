from typing import Optional

from pydantic import BaseModel, model_validator

from schale import literal


class GroupItem(BaseModel):
    Type: literal.DropType
    Id: int
    Chance: float
    AmountMin: int
    AmountMax: int


class GroupEntry(BaseModel):
    Items: list[GroupItem]
    ItemsCn: Optional[list[GroupItem]] = None
    ItemsGlobal: Optional[list[GroupItem]] = None
    Recursive: Optional[bool] = None
    RewardAll: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def forbid_unknown_keys(cls, data: dict[str, object]) -> dict[str, object]:
        if unexpected_keys := set(data.keys()) - cls.model_fields.keys():
            raise ValueError(
                f"Model {cls.__name__} has unexpected keys: {unexpected_keys}"
            )
        return data

    def get_group_items(self, server: literal.Server) -> list[GroupItem]:
        if server == "Global" and self.ItemsGlobal:
            return list(self.ItemsGlobal)
        if server == "Cn" and self.ItemsCn:
            return list(self.ItemsCn)
        return list(self.Items)
