"""
Bond EXP system and optimization examples.

This script demonstrates:
1. Bond progress tracking with EXP addition
2. Bond distribution optimization across FavorAlts students
3. Maximizing specific stats through optimal bond leveling
"""

from schale.bond_progress import BondProgress
from schale.bond_optimizer import (
    optimize_bond_distribution_greedy,
    create_bond_pool_with_alts,
    BondPool,
)
from schale.schema.student import Student


def create_example_students() -> dict[int, Student]:
    """Create example Hina variants for demonstration."""
    students_data = {
        10004: {
            "Id": 10004,
            "DevName": "Hina",
            "Name": "Hina",
            "PathName": "Hina",
            "IsReleased": [True, True, True],
            "School": "Gehenna",
            "Club": "Disciplinary Committee",
            "StarGrade": 3,
            "SquadType": "Main",
            "TacticRole": "Tanker",
            "Position": "Middle",
            "BulletType": "Explosion",
            "ArmorType": "HeavyArmor",
            "WeaponType": "SG",
            "Cover": True,
            "StreetBattleAdaptation": 4,
            "OutdoorBattleAdaptation": 2,
            "IndoorBattleAdaptation": 0,
            "MaxHP1": 3062,
            "MaxHP100": 26557,
            "AttackPower1": 244,
            "AttackPower100": 2441,
            "DefensePower1": 41,
            "DefensePower100": 253,
            "HealPower1": 1408,
            "HealPower100": 4225,
            "AccuracyPoint": 693,
            "DodgePoint": 534,
            "CriticalPoint": 137,
            "CriticalDamageRate": 20000,
            "StabilityPoint": 0,
            "Range": 400,
            "AmmoCount": 6,
            "AmmoCost": 2,
            "RegenCost": 700,
            "FavorStatType": ["AttackPower", "MaxHP"],
            "FavorStatValue": [
                [3, 38],
                [5, 63],
                [7, 88],
                [9, 106],
                [2, 18],
                [3, 29],
                [5, 47],
            ],
            "FavorAlts": [10022, 10086],
            "Equipment": ["Hat", "Gloves", "Shoes"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10004",
        },
        10022: {
            "Id": 10022,
            "DevName": "Hina_Swimsuit",
            "Name": "Hina (Swimsuit)",
            "PathName": "Hina_Swimsuit",
            "IsReleased": [True, True, True],
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
                [3, 0],
                [5, 0],
                [7, 43],
                [9, 51],
                [2, 8],
                [3, 13],
                [5, 21],
            ],
            "FavorAlts": [10004, 10086],
            "Equipment": ["Hat", "Hairpin", "Watch"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10022",
        },
        10086: {
            "Id": 10086,
            "DevName": "Hina_Dress",
            "Name": "Hina (Dress)",
            "PathName": "Hina_Dress",
            "IsReleased": [True, True, True],
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
                [3, 0],
                [5, 0],
                [7, 43],
                [9, 51],
                [2, 8],
                [3, 13],
                [5, 21],
            ],
            "FavorAlts": [10004, 10022],
            "Equipment": ["Hat", "Hairpin", "Watch"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10086",
        },
    }

    return {id: Student.model_validate(data) for id, data in students_data.items()}


def main():
    """Demonstrate bond EXP system and optimization."""
    print("=" * 70)
    print("Bond EXP System and Optimization Examples")
    print("=" * 70)

    # ========================================================================
    # Part 1: Bond Progress Tracking
    # ========================================================================
    print("\n[Part 1] Bond Progress Tracking with EXP Addition")
    print("-" * 70)

    # Start at level 1
    progress = BondProgress(level=1, current_exp=0)
    print(f"Starting: {progress}")

    # Add 500 EXP
    progress = progress + 500
    print(f"After +500 EXP: {progress}")

    # Add more EXP
    progress = progress + 2000
    print(f"After +2000 EXP: {progress}")

    # Check total EXP
    print(f"\nTotal accumulated EXP: {progress.total_exp}")
    print(f"EXP needed for next level: {progress.exp_to_next_level}")

    # Create from total EXP
    progress2 = BondProgress.from_total_exp(10000)
    print(f"\nCreated from 10000 total EXP: {progress2}")

    # ========================================================================
    # Part 2: Bond Optimization for Single Student
    # ========================================================================
    print("\n\n[Part 2] Bond Optimization - Single Student")
    print("-" * 70)

    students = create_example_students()
    hina = students[10004]

    # Current state: Level 20
    current = BondProgress(level=20, current_exp=0)
    available_exp = 15000

    print(f"Student: {hina.Name}")
    print(f"Current bond: Level {current.level}")
    print(f"Available EXP: {available_exp:,}")
    print(f"\nFavorStatType: {', '.join(hina.FavorStatType)}")

    # Optimize for AttackPower
    result = optimize_bond_distribution_greedy(
        [hina], current, available_exp, "AttackPower"
    )

    print(f"\nOptimization Result (AttackPower):")
    print(f"  Final level: {result.final_level}")
    print(f"  Level ups: {result.level_ups}")
    print(f"  AttackPower gain: +{result.stat_gain}")
    print(f"  EXP used: {result.exp_used:,}")
    print(f"  EXP remaining: {result.exp_remaining:,}")

    # ========================================================================
    # Part 3: Bond Pool Optimization (FavorAlts)
    # ========================================================================
    print("\n\n[Part 3] Bond Pool Optimization - FavorAlts Students")
    print("-" * 70)

    # All Hina variants share bond levels
    hina = students[10004]
    hina_swimsuit = students[10022]
    hina_dress = students[10086]

    all_hina = [hina, hina_swimsuit, hina_dress]

    print("Students in bond pool:")
    for student in all_hina:
        print(f"  - {student.Name} ({student.TacticRole})")

    # Create bond pool
    pool = create_bond_pool_with_alts(hina, students, BondProgress(level=10))

    print(f"\nBond pool size: {len(pool.students)} students")
    print(f"Shared bond level: {pool.shared_progress.level}")

    # Optimize for AttackPower
    print("\n--- Optimizing for AttackPower ---")
    result_atk = pool.optimize("AttackPower", 10000)

    print(f"Best student: ID {result_atk.target_student_id}")
    best_student_atk = students[result_atk.target_student_id]
    print(f"  Name: {best_student_atk.Name}")
    print(f"  Final level: {result_atk.final_level}")
    print(f"  AttackPower gain: +{result_atk.stat_gain}")

    # Optimize for MaxHP
    print("\n--- Optimizing for MaxHP ---")
    result_hp = pool.optimize("MaxHP", 10000)

    print(f"Best student: ID {result_hp.target_student_id}")
    best_student_hp = students[result_hp.target_student_id]
    print(f"  Name: {best_student_hp.Name}")
    print(f"  Final level: {result_hp.final_level}")
    print(f"  MaxHP gain: +{result_hp.stat_gain}")

    # ========================================================================
    # Part 4: Comparing Student Bond Gains
    # ========================================================================
    print("\n\n[Part 4] Comparing Bond Stat Gains Across Students")
    print("-" * 70)

    print(f"{'Student':<20} {'From Level':>12} {'To Level':>10} {'Atk Gain':>10} {'HP Gain':>10}")
    print("-" * 70)

    from_level = 20
    to_level = 30

    for student in all_hina:
        atk_stats_from = student.get_bond_stats(from_level)
        atk_stats_to = student.get_bond_stats(to_level)

        atk_gain = atk_stats_to["AttackPower"] - atk_stats_from["AttackPower"]
        hp_gain = atk_stats_to["MaxHP"] - atk_stats_from["MaxHP"]

        print(f"{student.Name:<20} {from_level:>12} {to_level:>10} {'+'+str(atk_gain):>10} {'+'+str(hp_gain):>10}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key Features:

1. **BondProgress Class**:
   - Immutable bond tracking (level + EXP)
   - Support for + and - operators
   - Automatic level calculation from total EXP

2. **EXP Addition**:
   progress = BondProgress(level=10, current_exp=100)
   new_progress = progress + 5000  # Add 5000 EXP

3. **Bond Pool Optimization**:
   - Find which student benefits most from bond leveling
   - Maximize specific stats (AttackPower, MaxHP, etc.)
   - Works with FavorAlts students sharing bond

4. **Usage with Real Data**:
   from schale import cache_collection, get_student_by_path_name
   from schale.bond_optimizer import create_bond_pool_with_alts
   from schale.bond_progress import BondProgress

   hina = get_student_by_path_name("Hina")
   pool = create_bond_pool_with_alts(
       hina,
       cache_collection.students,
       BondProgress(level=20)
   )
   result = pool.optimize("AttackPower", 10000)
"""
    )


if __name__ == "__main__":
    main()
