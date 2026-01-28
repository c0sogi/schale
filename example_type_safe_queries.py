"""
Type-safe student query examples.

This example demonstrates how the StudentPathName Literal type provides:
1. IDE autocomplete for all 194 student names
2. Compile-time type checking
3. Protection against typos and invalid names
"""

from schale import (
    get_student_by_path_name,
    get_students_by_school,
    get_students_by_role,
    get_student_variants,
    search_students_by_name,
    get_students_by_weapon,
    filter_students,
)


def main():
    """Demonstrate type-safe student queries."""

    print("=" * 70)
    print("Type-Safe Student Queries")
    print("=" * 70)

    # ========================================================================
    # 1. Type-safe student lookup by PathName
    # ========================================================================
    print("\n[1] Type-safe PathName lookup")
    print("-" * 70)
    print("✓ IDE autocompletes after typing 'Hina_'")
    print("✓ Type checker verifies PathName exists at compile time")
    print()

    # These PathNames have full IDE autocomplete support
    hina_swimsuit = get_student_by_path_name("Hina_Swimsuit")
    hina_dress = get_student_by_path_name("Hina_Dress")

    print(f"Found: {hina_swimsuit.Name} (ID: {hina_swimsuit.Id})")
    print(f"Found: {hina_dress.Name} (ID: {hina_dress.Id})")

    # Type error at compile time! IDE will show error
    # student = get_student_by_path_name("Invalid_Name")  # Type error!

    # Safe fallback with default
    maybe_student = get_student_by_path_name("Hina_Swimsuit", default=None)
    if maybe_student:
        print(f"\nWith default: {maybe_student.Name}")

    # ========================================================================
    # 2. Find all variants of a student
    # ========================================================================
    print("\n[2] Find all student variants")
    print("-" * 70)

    hina = get_student_by_path_name("Hina")
    variants = get_student_variants(hina)

    print(f"Base student: {hina.Name}")
    print(f"Total variants (including base): {len(variants)}")
    for variant in variants:
        print(f"  - {variant.Name} (PathName: {variant.PathName})")

    # ========================================================================
    # 3. Type-safe school filtering
    # ========================================================================
    print("\n[3] Filter by School (type-safe)")
    print("-" * 70)
    print("✓ IDE autocompletes school names")
    print()

    # School parameter has autocomplete for all valid schools
    gehenna_students = get_students_by_school("Gehenna")
    trinity_students = get_students_by_school("Trinity")

    print(f"Gehenna students: {len(gehenna_students)}")
    print(f"  First 3: {', '.join(s.Name for s in gehenna_students[:3])}")
    print(f"\nTrinity students: {len(trinity_students)}")
    print(f"  First 3: {', '.join(s.Name for s in trinity_students[:3])}")

    # ========================================================================
    # 4. Type-safe role filtering
    # ========================================================================
    print("\n[4] Filter by TacticRole (type-safe)")
    print("-" * 70)
    print("✓ IDE autocompletes: DamageDealer, Tanker, Healer, Support, Vehicle")
    print()

    tanks = get_students_by_role("Tanker")
    dealers = get_students_by_role("DamageDealer")

    print(f"Tankers: {len(tanks)}")
    print(f"  Examples: {', '.join(s.Name for s in tanks[:3])}")
    print(f"\nDamage Dealers: {len(dealers)}")
    print(f"  Examples: {', '.join(s.Name for s in dealers[:3])}")

    # ========================================================================
    # 5. Type-safe weapon filtering
    # ========================================================================
    print("\n[5] Filter by WeaponType (type-safe)")
    print("-" * 70)
    print("✓ IDE autocompletes: SG, SMG, AR, GL, HG, RL, SR, RG, MG, MT, FT")
    print()

    sr_users = get_students_by_weapon("SR")
    sg_users = get_students_by_weapon("SG")

    print(f"SR (Sniper Rifle) users: {len(sr_users)}")
    print(f"  Examples: {', '.join(s.Name for s in sr_users[:3])}")
    print(f"\nSG (Shotgun) users: {len(sg_users)}")
    print(f"  Examples: {', '.join(s.Name for s in sg_users[:3])}")

    # ========================================================================
    # 6. Complex type-safe filtering
    # ========================================================================
    print("\n[6] Multi-criteria filtering (all type-safe)")
    print("-" * 70)
    print("Query: Gehenna + DamageDealer + SR + Back position + 3-star")
    print()

    filtered = filter_students(
        school="Gehenna",  # Autocomplete
        role="DamageDealer",  # Autocomplete
        weapon="SR",  # Autocomplete
        position="Back",  # Autocomplete
        star_grade=3,
    )

    print(f"Found {len(filtered)} students:")
    for student in filtered:
        print(f"  - {student.Name}")
        print(f"    School: {student.School}, Role: {student.TacticRole}")
        print(f"    Weapon: {student.WeaponType}, Position: {student.Position}")

    # ========================================================================
    # 7. Search by name (partial match)
    # ========================================================================
    print("\n[7] Search by name (case-insensitive)")
    print("-" * 70)

    # Find all swimsuit versions
    swimsuit_students = search_students_by_name("Swimsuit")
    print(f"Swimsuit versions: {len(swimsuit_students)}")
    print(f"  First 5: {', '.join(s.Name for s in swimsuit_students[:5])}")

    # Find all Aru variants
    aru_students = search_students_by_name("Aru")
    print(f"\nAru variants: {len(aru_students)}")
    for student in aru_students:
        print(f"  - {student.Name} (PathName: {student.PathName})")

    # ========================================================================
    # 8. Bond stats with type-safe student lookup
    # ========================================================================
    print("\n[8] Bond stats calculation (type-safe)")
    print("-" * 70)

    # Type-safe student lookup
    student = get_student_by_path_name("Hina_Dress")

    # Calculate bond stats
    bond_stats = student.get_bond_stats(50)
    total_stats = student.get_total_stats(level=100, bond_level=50)

    print(f"Student: {student.Name}")
    print(f"\nBond stats at level 50:")
    for stat_type, value in bond_stats.items():
        print(f"  {stat_type}: +{value}")

    print(f"\nTotal stats (Lv100 + Bond50):")
    print(f"  AttackPower: {total_stats['AttackPower']}")
    print(f"  MaxHP: {total_stats['MaxHP']}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Benefits of Type-Safe Queries:")
    print("=" * 70)
    print("""
1. ✓ IDE Autocomplete
   - All 194 student PathNames
   - All valid School, Role, Weapon, Position values

2. ✓ Compile-time Type Checking
   - Catch typos before running code
   - Invalid values show errors in IDE

3. ✓ Better Documentation
   - Function signatures show exactly what values are valid
   - No need to guess or check documentation

4. ✓ Refactoring Safety
   - If enum values change, type checker finds all affected code
   - No silent runtime errors

Example usage:
   from schale import get_student_by_path_name

   # IDE autocompletes "Hina_Swimsuit" after typing "Hina_"
   student = get_student_by_path_name("Hina_Swimsuit")

   # Type error if typo!
   # student = get_student_by_path_name("Hina_Swmsuit")  # Error!
"""
    )


if __name__ == "__main__":
    # Note: This example requires actual student data from SchaleDB
    # If the server is unavailable, you'll get a RuntimeError
    try:
        main()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nNote: This example requires downloading student data from SchaleDB.")
        print("If the server is unavailable, the example cannot run.")
        print("\nTo test type safety, you can still:")
        print("1. Use your IDE's autocomplete (Ctrl+Space / Cmd+Space)")
        print("2. Hover over function calls to see type hints")
        print("3. Try typing invalid values to see type errors")
