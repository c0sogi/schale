"""
Example script demonstrating bond stat calculation functionality.

This script shows how to:
1. Access cached student data
2. Calculate bond stat bonuses
3. Get total stats with bond bonuses
"""

from schale.schema.student import Student


def create_example_student() -> Student:
    """Create an example student for demonstration."""
    hina_data = {
        "Id": 10086,
        "DevName": "Hina_Dress",
        "Name": "Hina (Dress)",
        "PathName": "Hina_Dress",
        "IsReleased": [True, True, True],
        "DefaultOrder": 0,
        "School": "Gehenna",
        "Club": "Disciplinary Committee",
        "StarGrade": 3,
        "SquadType": "Main",
        "TacticRole": "DamageDealer",
        "Position": "Back",
        "BulletType": "Explosion",
        "ArmorType": "LightArmor",
        "WeaponType": "SR",
        "Cover": True,
        "StreetBattleAdaptation": 4,
        "OutdoorBattleAdaptation": 2,
        "IndoorBattleAdaptation": 0,
        "MaxHP1": 2236,
        "MaxHP100": 19390,
        "AttackPower1": 369,
        "AttackPower100": 3690,
        "DefensePower1": 19,
        "DefensePower100": 119,
        "HealPower1": 1408,
        "HealPower100": 4225,
        "AccuracyPoint": 905,
        "DodgePoint": 201,
        "CriticalPoint": 201,
        "CriticalDamageRate": 20000,
        "StabilityPoint": 1988,
        "Range": 750,
        "AmmoCount": 5,
        "AmmoCost": 1,
        "RegenCost": 700,
        "FavorStatType": ["AttackPower", "MaxHP"],
        "FavorStatValue": [
            [3, 0],  # Index 0: Levels 1-4
            [5, 0],  # Index 1: Levels 5-9
            [7, 43],  # Index 2: Levels 10-14
            [9, 51],  # Index 3: Levels 15-19
            [2, 8],  # Index 4: Levels 20-29
            [3, 13],  # Index 5: Levels 30-39
            [5, 21],  # Index 6: Levels 40-49
        ],
        "Equipment": ["Hat", "Hairpin", "Watch"],
        "Skills": [],
        "WeaponImg": "weapon_icon_10086",
    }
    return Student.model_validate(hina_data)


def main():
    """Demonstrate bond stat calculations."""
    print("=== Bond Stat Calculation Example ===\n")

    # Create example student
    student = create_example_student()
    print(f"Student: {student.Name}")
    print(f"School: {student.School}")
    print(f"Bond Stat Types: {', '.join(student.FavorStatType)}")
    print()

    # Show bond stats at different levels
    print("Bond Stats at Different Levels:")
    print("-" * 60)
    print(f"{'Level':<10} {'AttackPower':>15} {'MaxHP':>15}")
    print("-" * 60)

    milestone_levels = [1, 5, 10, 15, 20, 30, 40, 50]
    for level in milestone_levels:
        bond_stats = student.get_bond_stats(level)
        print(
            f"{level:<10} {'+'+str(bond_stats['AttackPower']):>15} {'+'+str(bond_stats['MaxHP']):>15}"
        )

    print()

    # Show total stats at different character/bond levels
    print("\nTotal Stats (Level 1 character with different bond levels):")
    print("-" * 60)
    print(f"{'Bond Lv':<10} {'AttackPower':>15} {'MaxHP':>15}")
    print("-" * 60)

    for bond_level in [1, 20, 50]:
        total_stats = student.get_total_stats(level=1, bond_level=bond_level)
        print(
            f"{bond_level:<10} {total_stats['AttackPower']:>15} {total_stats['MaxHP']:>15}"
        )

    print()

    # Show total stats at max level with max bond
    print("\nTotal Stats at Max Level (100) with Max Bond (50):")
    print("-" * 60)
    max_stats = student.get_total_stats(level=100, bond_level=50)
    print(f"AttackPower: {max_stats['AttackPower']}")
    print(f"MaxHP: {max_stats['MaxHP']}")
    print(f"DefensePower: {max_stats['DefensePower']}")
    print(f"CriticalPoint: {max_stats['CriticalPoint']}")


if __name__ == "__main__":
    main()
