"""
Bond EXP system and optimization examples with independent student levels.

This script demonstrates:
1. Bond progress tracking with EXP addition
2. Bond distribution optimization across FavorAlts students
3. Each student has independent bond level, but stat bonuses apply based on max level
4. Maximizing specific stats through optimal bond leveling
"""

from schale import get_student_by_path_name
from schale.bond_progress import BondProgress
from schale.bond_optimizer import create_bond_pool_with_alts


def main():
    """Demonstrate bond EXP system and optimization with independent levels."""
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

    # ========================================================================
    # Part 2: FavorAlts System - Independent Levels
    # ========================================================================
    print("\n\n[Part 2] FavorAlts System - Independent Bond Levels")
    print("-" * 70)

    print("Key Concept:")
    print("  - Each student has INDEPENDENT bond level")
    print("  - Stat bonuses apply based on MAXIMUM level among FavorAlts")
    print("  - Example: Hina(20), Hina_Swimsuit(15), Hina_Dress(30)")
    print("    → All three get level 30 stat bonuses!")

    # ========================================================================
    # Part 3: Bond Pool Optimization - Simplified API
    # ========================================================================
    print("\n\n[Part 3] Bond Pool Optimization - Simplified API")
    print("-" * 70)

    # NO MORE BOILERPLATE! Just use PathName strings
    print("Creating bond pool with simple API:")
    print('  pool = create_bond_pool_with_alts(')
    print('      "Hina",')
    print('      initial_levels={')
    print('          "Hina": 20,')
    print('          "Hina_Swimsuit": 15,')
    print('          "Hina_Dress": 25,')
    print('      }')
    print('  )')

    # Create bond pool with improved API
    pool = create_bond_pool_with_alts(
        "Hina",  # Just use PathName! No need to fetch student object
        initial_levels={
            "Hina": 20,  # No more int IDs!
            "Hina_Swimsuit": 15,
            "Hina_Dress": 25,
        }
        # No need to pass cache_collection.students anymore!
    )

    print(f"\nInitial State:")
    print(f"  Students in pool: {len(pool.students)}")
    for student in pool.students:
        level = pool.student_progress[student.Id].level
        print(f"    - {student.Name:<22} Level {level}")

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
    for student in pool.students:
        allocated_exp = result_atk.exp_allocation.get(student.Id, 0)
        initial_level = pool.student_progress[student.Id].level
        final_level = result_atk.final_levels[student.Id]
        print(
            f"    {student.Name:<22} Level {initial_level:>2} → {final_level:>2}  "
            f"({allocated_exp:>5} EXP)"
        )

    # Optimize for MaxHP
    print("\n--- Optimizing for MaxHP with 10,000 EXP ---")
    result_hp = pool.optimize("MaxHP", 10000)

    print(f"\nOptimization Result:")
    print(f"  New MAX level: {result_hp.max_level} (was {pool.max_level})")
    print(f"  MaxHP gain: +{result_hp.stat_gain}")
    print(f"  EXP used: {result_hp.exp_used:,} / {10000:,}")

    # ========================================================================
    # Part 4: Different Starting Scenarios
    # ========================================================================
    print("\n\n[Part 4] Scenario Analysis - When to Level Each Student")
    print("-" * 70)

    # Scenario 1: All at same level
    print("\nScenario 1: All students at level 10")
    pool1 = create_bond_pool_with_alts(
        "Hina",
        initial_levels={
            "Hina": 10,
            "Hina_Swimsuit": 10,
            "Hina_Dress": 10,
        },
    )
    result1 = pool1.optimize("AttackPower", 5000)
    print(f"  Strategy: Level up one student to increase max level")
    print(f"  Result: Max level {pool1.max_level} → {result1.max_level}")
    print(f"  Allocation:")
    for student in pool1.students:
        exp = result1.exp_allocation.get(student.Id, 0)
        if exp > 0:
            print(f"    - {student.Name} gets {exp} EXP")

    # Scenario 2: One already high
    print("\nScenario 2: One student already at level 30, others at 10")
    pool2 = create_bond_pool_with_alts(
        "Hina",
        initial_levels={
            "Hina": 30,
            "Hina_Swimsuit": 10,
            "Hina_Dress": 10,
        },
    )
    result2 = pool2.optimize("AttackPower", 5000)
    print(f"  Strategy: Continue leveling the highest to push max level further")
    print(f"  Result: Max level {pool2.max_level} → {result2.max_level}")
    print(f"  Allocation:")
    for student in pool2.students:
        exp = result2.exp_allocation.get(student.Id, 0)
        if exp > 0:
            print(f"    - {student.Name} gets {exp} EXP")

    # ========================================================================
    # Part 5: Partial Level Specification
    # ========================================================================
    print("\n\n[Part 5] Partial Level Specification")
    print("-" * 70)

    # You don't need to specify all students - missing ones default to level 1
    print("You can specify only some students, others default to level 1:")
    pool3 = create_bond_pool_with_alts(
        "Hina",
        initial_levels={
            "Hina": 30,
            # Swimsuit and Dress will default to level 1
        },
    )
    print(f"  Pool max level: {pool3.max_level}")
    for student in pool3.students:
        level = pool3.student_progress[student.Id].level
        print(f"    - {student.Name:<22} Level {level}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        """
Key API Improvements:

1. **No More Boilerplate**:
   Before: create_bond_pool_with_alts(hina, cache_collection.students, ...)
   After:  create_bond_pool_with_alts("Hina", ...)

2. **Use PathNames Instead of IDs**:
   Before: initial_progress={10004: BondProgress(20, 0), 10022: ...}
   After:  initial_levels={"Hina": 20, "Hina_Swimsuit": 15}

3. **Simple Level Numbers**:
   Before: BondProgress(level=20, current_exp=0)
   After:  Just use integer: 20

4. **String or Object**:
   You can pass either a PathName string or a Student object
   create_bond_pool_with_alts("Hina", ...) OR
   create_bond_pool_with_alts(hina_object, ...)

Usage Example:

from schale.bond_optimizer import create_bond_pool_with_alts

# Super simple API!
pool = create_bond_pool_with_alts(
    "Hina",
    initial_levels={
        "Hina": 20,
        "Hina_Swimsuit": 15,
        "Hina_Dress": 25,
    }
)

# Optimize
result = pool.optimize("AttackPower", 10000)

# See allocation
for student in pool.students:
    exp = result.exp_allocation.get(student.Id, 0)
    level = result.final_levels[student.Id]
    print(f"{student.Name}: Level {level}, got {exp} EXP")
"""
    )


if __name__ == "__main__":
    main()
