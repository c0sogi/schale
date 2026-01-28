"""
Bond EXP system and optimization examples with independent student levels.

This script demonstrates:
1. Bond progress tracking with EXP addition
2. Bond distribution optimization across FavorAlts students
3. Each student has independent bond level, but stat bonuses apply based on max level
4. Maximizing specific stats through optimal bond leveling
"""

from schale.bond_progress import BondProgress
from schale.bond_optimizer import (
    BondPool,
    create_bond_pool_with_alts,
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
            "FavorStatValue": [[3, 0], [5, 0], [7, 43], [9, 51], [2, 8], [3, 13], [5, 21]],
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
            "FavorStatValue": [[3, 0], [5, 0], [7, 43], [9, 51], [2, 8], [3, 13], [5, 21]],
            "FavorAlts": [10004, 10022],
            "Equipment": ["Hat", "Hairpin", "Watch"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10086",
        },
    }

    return {id: Student.model_validate(data) for id, data in students_data.items()}


def main():
    """Demonstrate bond EXP system and optimization with independent levels."""
    print("=" * 70)
    print("Bond EXP System and Optimization Examples")
    print("=" * 70)

    # ========================================================================
    # Part 1: Bond Progress Tracking (unchanged)
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

    # ========================================================================
    # Part 2: FavorAlts System - Independent Levels
    # ========================================================================
    print("\n\n[Part 2] FavorAlts System - Independent Bond Levels")
    print("-" * 70)

    students = create_example_students()
    hina = students[10004]
    hina_swimsuit = students[10022]
    hina_dress = students[10086]

    print("Key Concept:")
    print("  - Each student has INDEPENDENT bond level")
    print("  - Stat bonuses apply based on MAXIMUM level among FavorAlts")
    print("  - Example: Hina(20), Hina_Swimsuit(15), Hina_Dress(30)")
    print("    → All three get level 30 stat bonuses!")

    # ========================================================================
    # Part 3: Bond Pool Optimization - Independent Levels
    # ========================================================================
    print("\n\n[Part 3] Bond Pool Optimization - Independent Levels")
    print("-" * 70)

    # Create initial state with different levels
    initial_progress = {
        10004: BondProgress(level=20, current_exp=0),  # Hina
        10022: BondProgress(level=15, current_exp=0),  # Hina (Swimsuit)
        10086: BondProgress(level=25, current_exp=0),  # Hina (Dress)
    }

    all_students_dict = {s.Id: s for s in [hina, hina_swimsuit, hina_dress]}
    pool = create_bond_pool_with_alts(
        hina, all_students_dict, initial_progress=initial_progress
    )

    print("Initial State:")
    print(f"  {'Student':<22} {'Level':>7} {'Current Stat Bonus':>20}")
    print("  " + "-" * 50)
    for student in [hina, hina_swimsuit, hina_dress]:
        level = initial_progress[student.Id].level
        print(f"  {student.Name:<22} {level:>7}    (see below)")

    print(f"\n  Current MAX level: {pool.max_level}")
    print(f"  → All students get level {pool.max_level} stat bonuses:")
    print(f"     AttackPower: +{pool.get_stat_bonus('AttackPower')}")
    print(f"     MaxHP: +{pool.get_stat_bonus('MaxHP')}")

    # Optimize for AttackPower
    print("\n--- Optimizing for AttackPower with 10,000 EXP ---")
    result_atk = pool.optimize("AttackPower", 10000)

    print(f"\nOptimization Result:")
    print(f"  New MAX level: {result_atk.max_level} (was {pool.max_level})")
    print(f"  AttackPower gain: +{result_atk.stat_gain}")
    print(f"  EXP used: {result_atk.exp_used:,} / {10000:,}")
    print(f"  EXP remaining: {result_atk.exp_remaining:,}")

    print(f"\n  EXP Allocation:")
    for student_id, allocated_exp in result_atk.exp_allocation.items():
        student_name = all_students_dict[student_id].Name
        final_level = result_atk.final_levels[student_id]
        initial_level = initial_progress[student_id].level
        print(
            f"    {student_name:<22} Level {initial_level:>2} → {final_level:>2}  "
            f"({allocated_exp:>5} EXP)"
        )

    # Optimize for MaxHP
    print("\n--- Optimizing for MaxHP with 10,000 EXP ---")
    result_hp = pool.optimize("MaxHP", 10000)

    print(f"\nOptimization Result:")
    print(f"  New MAX level: {result_hp.max_level} (was {pool.max_level})")
    print(f"  MaxHP gain: +{result_hp.stat_gain}")
    print(f"  EXP used: {result_hp.exp_used:,} / {10000:,}")
    print(f"  EXP remaining: {result_hp.exp_remaining:,}")

    print(f"\n  EXP Allocation:")
    for student_id, allocated_exp in result_hp.exp_allocation.items():
        student_name = all_students_dict[student_id].Name
        final_level = result_hp.final_levels[student_id]
        initial_level = initial_progress[student_id].level
        print(
            f"    {student_name:<22} Level {initial_level:>2} → {final_level:>2}  "
            f"({allocated_exp:>5} EXP)"
        )

    # ========================================================================
    # Part 4: Different Starting Scenarios
    # ========================================================================
    print("\n\n[Part 4] Scenario Analysis - When to Level Each Student")
    print("-" * 70)

    # Scenario 1: All at same level
    print("\nScenario 1: All students at level 10")
    pool1 = create_bond_pool_with_alts(
        hina,
        all_students_dict,
        initial_progress={
            10004: BondProgress(10, 0),
            10022: BondProgress(10, 0),
            10086: BondProgress(10, 0),
        },
    )
    result1 = pool1.optimize("AttackPower", 5000)
    print(f"  Strategy: Level up one student to increase max level")
    print(f"  Result: Max level {pool1.max_level} → {result1.max_level}")
    print(f"  Allocation: ", end="")
    for sid, exp in result1.exp_allocation.items():
        if exp > 0:
            print(f"{all_students_dict[sid].Name} gets {exp} EXP")

    # Scenario 2: One already high
    print("\nScenario 2: One student already at level 30, others at 10")
    pool2 = create_bond_pool_with_alts(
        hina,
        all_students_dict,
        initial_progress={
            10004: BondProgress(30, 0),
            10022: BondProgress(10, 0),
            10086: BondProgress(10, 0),
        },
    )
    result2 = pool2.optimize("AttackPower", 5000)
    print(f"  Strategy: Continue leveling the highest to push max level further")
    print(f"  Result: Max level {pool2.max_level} → {result2.max_level}")
    print(f"  Allocation: ", end="")
    for sid, exp in result2.exp_allocation.items():
        if exp > 0:
            print(f"{all_students_dict[sid].Name} gets {exp} EXP")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        """
Key Features:

1. **Independent Bond Levels**:
   - Each student has their own bond level
   - Hina: Level 20, Hina(Swimsuit): Level 15, Hina(Dress): Level 25

2. **Shared Stat Bonuses**:
   - All FavorAlts get stat bonuses based on MAXIMUM level
   - In above example, all three get level 25 bonuses

3. **Independent EXP Distribution**:
   - Can freely allocate EXP to each student
   - Example: Give 60 EXP to Hina, 30 to Swimsuit, 10 to Dress

4. **Optimization Strategy**:
   - Greedy algorithm: Always level up the student with max level
   - This ensures the stat bonus level increases as quickly as possible
   - Efficient use of limited bond items

5. **Usage with Real Data**:
   from schale import cache_collection, get_student_by_path_name
   from schale.bond_optimizer import create_bond_pool_with_alts
   from schale.bond_progress import BondProgress

   hina = get_student_by_path_name("Hina")
   pool = create_bond_pool_with_alts(
       hina,
       cache_collection.students,
       initial_progress={
           10004: BondProgress(level=20, current_exp=0),
           10022: BondProgress(level=15, current_exp=0),
           10086: BondProgress(level=25, current_exp=0),
       }
   )
   result = pool.optimize("AttackPower", 10000)

   # See how to allocate EXP
   for student_id, exp in result.exp_allocation.items():
       print(f"Give {exp} EXP to student {student_id}")
"""
    )


if __name__ == "__main__":
    main()
