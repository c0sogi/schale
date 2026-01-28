"""
학생 데이터 조회 예제 (타입 안전 버전)

SchaleDB에서 학생 정보를 가져오고 필터링하는 다양한 방법을 보여줍니다.
IDE 자동완성과 타입 체킹을 활용한 타입 안전한 방식을 사용합니다.

참고: 최신 타입 안전 예제는 example_type_safe_queries.py를 참조하세요.
"""

from schale.schema.student import Student
from schale import (
    get_student_by_path_name,
    get_students_by_school,
    get_students_by_role,
    filter_students,
)


def create_example_students() -> dict[int, Student]:
    """예제용 히나 버전들 생성 (실제로는 cache_collection.students 사용)"""
    students_data = [
        {
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
            "FavorStatValue": [[3, 38], [5, 63], [7, 88], [9, 106], [2, 18], [3, 29], [5, 47]],
            "FavorAlts": [10022, 10086],  # Swimsuit, Dress
            "Equipment": ["Hat", "Gloves", "Shoes"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10004",
        },
        {
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
            "FavorAlts": [10004, 10086],  # Original, Dress
            "Equipment": ["Hat", "Hairpin", "Watch"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10022",
        },
        {
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
            "FavorAlts": [10004, 10022],  # Original, Swimsuit
            "Equipment": ["Hat", "Hairpin", "Watch"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10086",
        },
    ]

    return {s["Id"]: Student.model_validate(s) for s in students_data}


def main():
    """학생 데이터 조회 예제"""

    # 실제 사용 시:
    # from schale import cache_collection
    # students = cache_collection.students

    # 예제용 데이터
    students = create_example_students()

    print("=" * 70)
    print("학생 데이터 조회 방법")
    print("=" * 70)

    # 방법 1: ID로 직접 조회
    print("\n[1] ID로 직접 조회")
    print("-" * 70)
    hina_dress = students[10086]
    print(f"ID 10086: {hina_dress.Name}")
    print(f"  - PathName: {hina_dress.PathName}")
    print(f"  - School: {hina_dress.School}")
    print(f"  - TacticRole: {hina_dress.TacticRole}")

    # 방법 2: Name으로 검색
    print("\n[2] Name으로 검색 (부분 일치)")
    print("-" * 70)
    hina_students = [s for s in students.values() if "Hina" in s.Name]
    for student in hina_students:
        print(f"  - {student.Name} (ID: {student.Id})")

    # 방법 3: PathName으로 검색 (정확한 매칭)
    print("\n[3] PathName으로 검색 (정확한 매칭)")
    print("-" * 70)
    target_path = "Hina_Swimsuit"
    result = next((s for s in students.values() if s.PathName == target_path), None)
    if result:
        print(f"  PathName '{target_path}': {result.Name} (ID: {result.Id})")

    # 방법 4: DevName으로 검색
    print("\n[4] DevName으로 검색")
    print("-" * 70)
    dev_name = "Hina_Dress"
    result = next((s for s in students.values() if s.DevName == dev_name), None)
    if result:
        print(f"  DevName '{dev_name}': {result.Name} (ID: {result.Id})")

    # 방법 5: 같은 캐릭터의 다른 버전 찾기 (FavorAlts 활용)
    print("\n[5] 같은 캐릭터의 다른 버전 찾기 (FavorAlts)")
    print("-" * 70)
    base_student = students[10004]  # Hina (Original)
    print(f"기준 학생: {base_student.Name} (ID: {base_student.Id})")
    print(f"다른 버전 ID들: {base_student.FavorAlts}")

    if base_student.FavorAlts:
        print("\n다른 버전들:")
        for alt_id in base_student.FavorAlts:
            if alt_id in students:
                alt_student = students[alt_id]
                print(f"  - {alt_student.Name} (ID: {alt_id})")

    # 방법 6: 학교별로 필터링
    print("\n[6] 학교별 필터링")
    print("-" * 70)
    gehenna_students = [s for s in students.values() if s.School == "Gehenna"]
    print(f"Gehenna 학생 수: {len(gehenna_students)}")
    for student in gehenna_students[:3]:  # 처음 3명만
        print(f"  - {student.Name}")

    # 방법 7: TacticRole로 필터링
    print("\n[7] TacticRole로 필터링")
    print("-" * 70)
    dealers = [s for s in students.values() if s.TacticRole == "DamageDealer"]
    print(f"DamageDealer 역할 학생 수: {len(dealers)}")
    for student in dealers:
        print(f"  - {student.Name} ({student.WeaponType})")

    # 방법 8: 복합 조건 검색
    print("\n[8] 복합 조건 검색 (Gehenna + DamageDealer + 3성)")
    print("-" * 70)
    filtered = [
        s
        for s in students.values()
        if s.School == "Gehenna" and s.TacticRole == "DamageDealer" and s.StarGrade == 3
    ]
    for student in filtered:
        print(f"  - {student.Name} (Position: {student.Position})")

    # 방법 9: 모든 히나 버전 비교
    print("\n[9] 히나 버전 비교")
    print("-" * 70)
    print(f"{'Name':<20} {'Role':<15} {'Armor':<12} {'Weapon':<8}")
    print("-" * 70)
    for student in hina_students:
        print(
            f"{student.Name:<20} {student.TacticRole:<15} "
            f"{student.ArmorType:<12} {student.WeaponType:<8}"
        )

    # 방법 10: PathName 패턴으로 검색
    print("\n[10] PathName 패턴 검색 (특정 버전만)")
    print("-" * 70)
    swimsuit_students = [s for s in students.values() if "Swimsuit" in s.PathName]
    print(f"수영복 버전 학생 수: {len(swimsuit_students)}")
    for student in swimsuit_students:
        print(f"  - {student.Name} (PathName: {student.PathName})")

    print("\n" + "=" * 70)
    print("실제 사용 예제:")
    print("=" * 70)
    print(
        """
from schale import cache_collection

# 모든 학생 가져오기
students = cache_collection.students

# ID로 조회
hina_dress = students[10086]

# Name으로 검색
hina_versions = [s for s in students.values() if "Hina" in s.Name]

# PathName으로 조회
hina_swimsuit = next(
    (s for s in students.values() if s.PathName == "Hina_Swimsuit"),
    None
)

# 학교와 역할로 필터링
gehenna_tanks = [
    s for s in students.values()
    if s.School == "Gehenna" and s.TacticRole == "Tanker"
]
"""
    )


if __name__ == "__main__":
    main()
