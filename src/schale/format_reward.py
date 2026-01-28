from functools import cache

from schale import literal, localization
from schale.cache_control import cache_collection
from schale.schema.equipments import Equipment
from schale.schema.furniture import Furniture
from schale.schema.item import Item


@cache
def format_reward(reward_id: int, reward_type: literal.DropType) -> str:
    match reward_type:
        case "Currency":
            if reward_id == 1:
                return localization.CREDIT_NAMES[localization.USER_LANG]
            elif reward_id == 3:
                return localization.PYROXENE_NAMES[localization.USER_LANG]
            else:
                raise ValueError(f"Unknown currency id: {reward_id}")
        case "Equipment":
            return str(Equipment.model_validate(cache_collection.equipments[reward_id]))
        case "Furniture":
            return str(Furniture.model_validate(cache_collection.furnitures[reward_id]))
        case "Item":
            return str(Item.model_validate(cache_collection.items[reward_id]).root)
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(f"Unknown reward type: {reward_type}")
