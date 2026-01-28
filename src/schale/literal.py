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

# Student PathNames (all 194 students from SchaleDB)
# This enables IDE autocomplete and type safety when querying students
StudentPathName = Literal[
    "Airi", "Airi_Band", "Akane", "Akane_Bunny",
    "Akari", "Akari_NewYear", "Ako", "Ako_Dress",
    "Aris", "Aris_Maid", "Aru", "Aru_Dress",
    "Aru_NewYear", "Asuna", "Asuna_Bunny", "Atsuko",
    "Atsuko_Swimsuit", "Ayane", "Ayane_Swimsuit", "Azusa",
    "Azusa_Swimsuit", "Cherino", "Cherino_HotSpring", "Chihiro",
    "Chinatsu", "Chinatsu_HotSpring", "Chise", "Chise_Swimsuit",
    "Eimi", "Eimi_Swimsuit", "Fubuki", "Fubuki_Swimsuit",
    "Fuuka", "Fuuka_NewYear", "Hanae", "Hanae_Christmas",
    "Hanako", "Hanako_Swimsuit", "Hare", "Hare_Camp",
    "Haruka", "Haruka_NewYear", "Haruna", "Haruna_NewYear",
    "Haruna_Track", "Hasumi", "Hasumi_Track", "Hatsune_Miku",
    "Hibiki", "Hibiki_Cheerleader", "Hifumi", "Hifumi_Swimsuit",
    "Himari", "Hina", "Hina_Dress", "Hina_Swimsuit",
    "Hinata", "Hinata_Swimsuit", "Hiyori", "Hiyori_Swimsuit",
    "Hoshino", "Hoshino_Battle_Dealer", "Hoshino_Battle_Tank", "Hoshino_Swimsuit",
    "Ibuki", "Ichika", "Iori", "Iori_Swimsuit",
    "Iroha", "Izumi", "Izumi_Swimsuit", "Izuna",
    "Izuna_Swimsuit", "Junko", "Junko_NewYear", "Juri",
    "Kaede", "Kaho", "Kanna", "Kanna_Swimsuit",
    "Karin", "Karin_Bunny", "Kasumi", "Kayoko",
    "Kayoko_Dress", "Kayoko_NewYear", "Kazusa", "Kazusa_Band",
    "Kikyou", "Kirara", "Kirino", "Kirino_Swimsuit",
    "Koharu", "Koharu_Swimsuit", "Kokona", "Kotama",
    "Kotama_Camp", "Kotori", "Kotori_Cheerleader", "Koyuki",
    "Maki", "Makoto", "Mari", "Mari_Track",
    "Marina", "Mashiro", "Mashiro_Swimsuit", "Megu",
    "Meru", "Michiru", "Midori", "Midori_Maid",
    "Mika", "Mimori", "Mimori_Swimsuit", "Mina",
    "Mine", "Minori", "Misaka_Mikoto", "Misaki",
    "Miyako", "Miyako_Swimsuit", "Miyu", "Miyu_Swimsuit",
    "Moe", "Moe_Swimsuit", "Momiji", "Momoi",
    "Momoi_Maid", "Mutsuki", "Mutsuki_NewYear", "Nagisa",
    "Natsu", "Neru", "Neru_Bunny", "Noa",
    "Nodoka", "Nodoka_HotSpring", "Nonomi", "Nonomi_Swimsuit",
    "Pina", "Reisa", "Renge", "Rumi",
    "Saki", "Saki_Swimsuit", "Sakurako", "Saori",
    "Saori_Swimsuit", "Saten_Ruiko", "Saya", "Saya_Casual",
    "Sena", "Serika", "Serika_NewYear", "Serika_Swimsuit",
    "Serina", "Serina_Christmas", "Shigure", "Shigure_HotSpring",
    "Shimiko", "Shiroko", "Shiroko_Cycling", "Shiroko_Swimsuit",
    "Shiroko_Terror", "Shizuko", "Shizuko_Swimsuit", "Shokuhou_Misaki",
    "Shun", "Shun_Small", "Sumire", "Suzumi",
    "Toki", "Toki_Bunny", "Tomoe", "Tsubaki",
    "Tsubaki_Guide", "Tsukuyo", "Tsurugi", "Tsurugi_Swimsuit",
    "Ui", "Ui_Swimsuit", "Umika", "Utaha",
    "Utaha_Cheerleader", "Wakamo", "Wakamo_Swimsuit", "Yoshimi",
    "Yoshimi_Band", "Yukari", "Yuuka", "Yuuka_Track",
    "Yuzu", "Yuzu_Maid",
]
