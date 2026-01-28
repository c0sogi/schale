from typing import Literal


Server = Literal["Global", "Cn", "Jp"]
ConsumableChoiceType = Literal["Item", "Equipment", "Furniture"]
DropType = Literal["Currency", "Equipment", "Furniture", "Item"]
FurnitureCategory = Literal["Interiors", "Furnitures", "Decorations"]
FurnitureSubCategory = Literal[
    "Background",
    "Bed",
    "Chair",
    "Closet",
    "Floor",
    "FloorDecoration",
    "FurnitureEtc",
    "HomeAppliance",
    "Prop",
    "Table",
    "Trophy",
    "WallDecoration",
    "Wallpaper",
]
Rarity = Literal["N", "R", "SR", "SSR"]
EquipmentCategory = Literal[
    "Exp",
    "WeaponExpGrowthA",
    "WeaponExpGrowthB",
    "WeaponExpGrowthC",
    "WeaponExpGrowthZ",
    "Hat",
    "Gloves",
    "Shoes",
    "Bag",
    "Badge",
    "Hairpin",
    "Charm",
    "Necklace",
    "Watch",
]




ShopCategory = Literal["General", "MasterCoin", "SecretStoneGrowth"]
ShopCostType = Literal["Currency", "Item"]
ItemCategory = Literal[
    "Material",
    "Coin",
    "CharacterExpGrowth",
    "Favor",
    "SecretStone",
    "Collectible",
    "Consumable",
]
RewardCondition = Literal["FirstClear", "ThreeStar"]




Terrain = Literal["Indoor", "Outdoor", "Street"]
UnitGrade = Literal["Grade1", "Grade2", "Grade3", "Boss"]
RewardType = Literal["FirstClear", "ThreeStar"]
Reward = Literal["Currency", "GachaGroup", "Equipment", "Item"]


StageCategory = Literal["Campaign", "Bounty", "WeekDungeon", "SchoolDungeon"]

BountyStageType = Literal["ChaserA", "ChaserB", "ChaserC"]
WeekDungeonStageType = Literal["FindGift", "Blood"]
SchoolDungeonStageType = Literal["SchoolA", "SchoolB", "SchoolC"]


# Student-related types
StatType = Literal[
    "AttackPower",
    "MaxHP",
    "DefensePower",
    "HealPower",
    "AccuracyPoint",
    "DodgePoint",
    "CriticalPoint",
    "CriticalDamageRate",
    "StabilityPoint",
    "Range",
    "AmmoCount",
    "AmmoCost",
    "RegenCost",
]

ArmorType = Literal["LightArmor", "HeavyArmor", "Unarmed", "ElasticArmor"]
BulletType = Literal["Explosion", "Pierce", "Mystic", "Sonic"]
WeaponType = Literal["SG", "SMG", "AR", "GL", "HG", "RL", "SR", "RG", "MG", "MT", "FT"]
School = Literal[
    "Abydos",
    "Arius",
    "ETC",
    "Gehenna",
    "Hyakkiyako",
    "Millennium",
    "RedWinter",
    "Shanhaijing",
    "SRT",
    "Trinity",
    "Valkyrie",
]
SquadType = Literal["Main", "Support"]
TacticRole = Literal[
    "DamageDealer",
    "Tanker",
    "Healer",
    "Support",
    "Vehicle",
]
Position = Literal["Front", "Middle", "Back"]
