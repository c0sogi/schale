from typing import Optional
from pydantic import BaseModel, Field

from schale import literal


_TriBool = tuple[bool, bool, bool]
_FavorStatValue = list[list[int]]  # 7 entries, each with 2 values


class GearData(BaseModel):
    """Equipment (Gear) data for students"""

    Released: _TriBool
    StatType: list[literal.StatType]
    StatValue: list[list[int]]


class WeaponData(BaseModel):
    """Weapon data for students"""

    Name: str
    Desc: str
    StatLevelUpType: str
    MaxLevel: int
    AttackPower1: int
    AttackPower100: int
    MaxHP1: int
    MaxHP100: int
    HealPower1: int
    HealPower100: int


class SkillEffect(BaseModel):
    """Skill effect details"""

    Type: str
    # Other fields can be added as needed


class Skill(BaseModel):
    """Student skill data"""

    SkillType: str
    Name: Optional[str] = None
    Desc: Optional[str] = None
    Parameters: Optional[list[list[str]]] = None
    Cost: Optional[list[int]] = None
    Duration: Optional[int] = None
    Range: Optional[int] = None
    Icon: Optional[str] = None
    Effects: list[SkillEffect]


class Student(BaseModel):
    """
    Student (Character) data from SchaleDB.

    Contains complete student information including stats, skills,
    favor/bond data, and profile information.
    """

    # === Core Identity ===
    Id: int
    DevName: str
    Name: str
    PathName: str

    # === Release Status ===
    IsReleased: _TriBool  # [Global, CN, JP]
    IsLimited: int = 0
    DefaultOrder: int = 0

    # === Profile Information ===
    School: literal.School
    Club: str
    StarGrade: int = Field(ge=1, le=3)
    SquadType: literal.SquadType
    TacticRole: literal.TacticRole

    # === Combat Attributes ===
    Position: literal.Position
    BulletType: literal.BulletType
    ArmorType: literal.ArmorType
    WeaponType: literal.WeaponType
    Cover: bool = True

    # === Adaptation (Terrain bonuses) ===
    StreetBattleAdaptation: int = Field(ge=0, le=4)
    OutdoorBattleAdaptation: int = Field(ge=0, le=4)
    IndoorBattleAdaptation: int = Field(ge=0, le=4)

    # === Base Stats (Level 1) ===
    MaxHP1: int
    AttackPower1: int
    DefensePower1: int
    HealPower1: int
    AccuracyPoint: int
    DodgePoint: int
    CriticalPoint: int
    CriticalDamageRate: int
    StabilityPoint: int
    Range: int
    AmmoCount: int
    AmmoCost: int
    RegenCost: int

    # === Max Stats (Level 100) ===
    MaxHP100: int
    AttackPower100: int
    DefensePower100: int
    HealPower100: int

    # === Optional Stats ===
    DefensePenetration1: Optional[int] = None
    DefensePenetration100: Optional[int] = None

    # === Favor/Bond System ===
    FavorStatType: list[literal.StatType]
    FavorStatValue: _FavorStatValue
    FavorAlts: Optional[list[int]] = None  # Alternative versions sharing bond
    FavorItemTags: Optional[list[str]] = None
    FavorItemUniqueTags: Optional[list[str]] = None

    # === Equipment ===
    Equipment: list[literal.EquipmentCategory]
    Gear: Optional[GearData] = None

    # === Skills ===
    Skills: list[Skill]
    SkillExMaterial: Optional[list[list[int]]] = None
    SkillExMaterialAmount: Optional[list[list[int]]] = None
    SkillMaterial: Optional[list[list[int]]] = None
    SkillMaterialAmount: Optional[list[list[int]]] = None

    # === Weapon ===
    WeaponImg: str
    Weapon: Optional[WeaponData] = None

    # === Character Profile ===
    FamilyName: Optional[str] = None
    PersonalName: Optional[str] = None
    SchoolYear: Optional[str] = None
    CharacterAge: Optional[str] = None
    Birthday: Optional[str] = None
    BirthDay: Optional[str] = None  # Short format (e.g., "3/12")
    CharHeightMetric: Optional[str] = None
    CharHeightImperial: Optional[str] = None
    Hobby: Optional[str] = None
    Designer: Optional[str] = None
    Illustrator: Optional[str] = None
    CharacterVoice: Optional[str] = None
    ProfileIntroduction: Optional[str] = None
    CharacterSSRNew: Optional[str] = None  # Recruitment quote

    # === Visual Assets ===
    CollectionBG: Optional[str] = None

    # === Memory Lobby ===
    MemoryLobby: Optional[list[int]] = None
    MemoryLobbyBGM: Optional[str] = None

    # === Furniture Interaction ===
    FurnitureInteraction: Optional[list[list[int]]] = None

    # === Summons (for summoner characters) ===
    Summons: list[int] = Field(default_factory=list)

    # === Related Data ===
    LinkedCharacterId: Optional[int] = None  # For linked characters
    StyleId: Optional[int] = None  # Character style variant
    TSAId: Optional[str] = None  # TSA-related ID

    def __str__(self) -> str:
        return f"{self.Name} ({self.School}) - {self.Id}"

    def get_bond_stats(self, bond_level: int) -> dict[str, int]:
        """
        Calculate bond stat bonuses for a given bond level (1-50).

        Based on SchaleDB's getBondStats algorithm:
        - FavorStatValue has 7 entries representing incremental stat gains
        - Index 0: Levels 1-4   (each level adds this value)
        - Index 1: Levels 5-9
        - Index 2: Levels 10-14
        - Index 3: Levels 15-19
        - Index 4: Levels 20-29
        - Index 5: Levels 30-39
        - Index 6: Levels 40-49

        Args:
            bond_level: Bond level (1-50)

        Returns:
            Dictionary mapping stat type to total bonus value.
            Example: {"AttackPower": 217, "MaxHP": 890}

        Example:
            >>> student = Student(...)
            >>> student.get_bond_stats(50)
            {"AttackPower": 217, "MaxHP": 890}
        """
        if bond_level < 1:
            bond_level = 1
        if bond_level > 50:
            bond_level = 50

        stat1_total = 0
        stat2_total = 0

        for level in range(1, bond_level):
            if level < 20:
                # Levels 1-19: Use indices 0-3
                index = level // 5
                stat1_total += self.FavorStatValue[index][0]
                stat2_total += self.FavorStatValue[index][1]
            elif level < 50:
                # Levels 20-49: Use indices 4-6
                index = 2 + level // 10
                stat1_total += self.FavorStatValue[index][0]
                stat2_total += self.FavorStatValue[index][1]

        return {
            self.FavorStatType[0]: stat1_total,
            self.FavorStatType[1]: stat2_total,
        }

    def get_total_stats(self, level: int = 1, bond_level: int = 1) -> dict[str, int]:
        """
        Calculate total stats including base stats and bond bonuses.

        Args:
            level: Character level (1-100)
            bond_level: Bond level (1-50)

        Returns:
            Dictionary with all stats including bond bonuses.
        """
        if level < 1:
            level = 1
        if level > 100:
            level = 100

        # Linear interpolation for base stats
        level_factor = (level - 1) / 99

        base_stats = {
            "MaxHP": int(self.MaxHP1 + (self.MaxHP100 - self.MaxHP1) * level_factor),
            "AttackPower": int(
                self.AttackPower1 + (self.AttackPower100 - self.AttackPower1) * level_factor
            ),
            "DefensePower": int(
                self.DefensePower1 + (self.DefensePower100 - self.DefensePower1) * level_factor
            ),
            "HealPower": int(
                self.HealPower1 + (self.HealPower100 - self.HealPower1) * level_factor
            ),
            "AccuracyPoint": self.AccuracyPoint,
            "DodgePoint": self.DodgePoint,
            "CriticalPoint": self.CriticalPoint,
            "CriticalDamageRate": self.CriticalDamageRate,
            "StabilityPoint": self.StabilityPoint,
            "Range": self.Range,
            "AmmoCount": self.AmmoCount,
            "AmmoCost": self.AmmoCost,
            "RegenCost": self.RegenCost,
        }

        # Add bond bonuses
        bond_bonuses = self.get_bond_stats(bond_level)
        for stat_type, bonus in bond_bonuses.items():
            if stat_type in base_stats:
                base_stats[stat_type] += bonus

        return base_stats
