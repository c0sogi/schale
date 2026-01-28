"""
Type-safe utilities for querying student data.

Provides helper functions with full type safety and IDE autocomplete support.
"""

from typing import Optional, overload

from schale import literal
from schale.cache_control import cache_collection
from schale.schema.student import Student


@overload
def get_student_by_path_name(path_name: literal.StudentPathName) -> Student: ...


@overload
def get_student_by_path_name(
    path_name: literal.StudentPathName, *, default: None = None
) -> Optional[Student]: ...


@overload
def get_student_by_path_name(
    path_name: literal.StudentPathName, *, default: Student
) -> Student: ...


def get_student_by_path_name(
    path_name: literal.StudentPathName, *, default: Optional[Student] = ...
) -> Optional[Student]:
    """
    Get a student by PathName with full type safety.

    This function provides IDE autocomplete for student PathNames and
    ensures type safety at compile time.

    Args:
        path_name: The PathName of the student (e.g., "Hina_Swimsuit").
                   IDE will autocomplete all 194 valid student names.
        default: Value to return if student not found. If not provided,
                raises KeyError when student not found.

    Returns:
        Student object if found, otherwise returns default value.

    Raises:
        KeyError: If student not found and no default provided.

    Examples:
        >>> # IDE autocompletes after typing "Hina_"
        >>> student = get_student_by_path_name("Hina_Swimsuit")
        >>> print(student.Name)
        Hina (Swimsuit)

        >>> # With default value
        >>> student = get_student_by_path_name("NonexistentStudent", default=None)
        >>> if student is None:
        ...     print("Student not found")

        >>> # Raises KeyError if not found
        >>> student = get_student_by_path_name("InvalidName")  # KeyError
    """
    students = cache_collection.students

    # Find student by PathName
    result = next((s for s in students.values() if s.PathName == path_name), None)

    if result is None:
        if default is ...:
            raise KeyError(f"Student with PathName '{path_name}' not found")
        return default

    return result


def get_students_by_school(school: literal.School) -> list[Student]:
    """
    Get all students from a specific school.

    Args:
        school: School name. IDE will autocomplete valid school names.

    Returns:
        List of students belonging to the school.

    Example:
        >>> gehenna_students = get_students_by_school("Gehenna")
        >>> for student in gehenna_students:
        ...     print(student.Name)
    """
    students = cache_collection.students
    return [s for s in students.values() if s.School == school]


def get_students_by_role(role: literal.TacticRole) -> list[Student]:
    """
    Get all students with a specific tactic role.

    Args:
        role: Tactic role. IDE will autocomplete valid roles.

    Returns:
        List of students with the specified role.

    Example:
        >>> tanks = get_students_by_role("Tanker")
        >>> dealers = get_students_by_role("DamageDealer")
    """
    students = cache_collection.students
    return [s for s in students.values() if s.TacticRole == role]


def get_student_variants(student: Student) -> list[Student]:
    """
    Get all variants of a student (including the student itself).

    Uses the FavorAlts field to find alternative versions of the same character.

    Args:
        student: The student to find variants for.

    Returns:
        List containing the student and all their variants.

    Example:
        >>> hina = get_student_by_path_name("Hina")
        >>> variants = get_student_variants(hina)
        >>> for variant in variants:
        ...     print(variant.Name)  # Hina, Hina (Swimsuit), Hina (Dress)
    """
    students = cache_collection.students
    result = [student]

    if student.FavorAlts:
        for alt_id in student.FavorAlts:
            if alt_id in students:
                result.append(students[alt_id])

    return result


def search_students_by_name(name_query: str) -> list[Student]:
    """
    Search students by name (case-insensitive partial match).

    Args:
        name_query: Search query to match against student names.

    Returns:
        List of students whose names contain the query string.

    Example:
        >>> # Find all Hina variants
        >>> hina_students = search_students_by_name("Hina")

        >>> # Find swimsuit versions
        >>> swimsuit_students = search_students_by_name("Swimsuit")
    """
    students = cache_collection.students
    query_lower = name_query.lower()
    return [s for s in students.values() if query_lower in s.Name.lower()]


def get_students_by_weapon(weapon_type: literal.WeaponType) -> list[Student]:
    """
    Get all students using a specific weapon type.

    Args:
        weapon_type: Weapon type. IDE will autocomplete valid weapon types.

    Returns:
        List of students using the specified weapon.

    Example:
        >>> sr_users = get_students_by_weapon("SR")
        >>> sg_users = get_students_by_weapon("SG")
    """
    students = cache_collection.students
    return [s for s in students.values() if s.WeaponType == weapon_type]


def filter_students(
    *,
    school: Optional[literal.School] = None,
    role: Optional[literal.TacticRole] = None,
    weapon: Optional[literal.WeaponType] = None,
    position: Optional[literal.Position] = None,
    star_grade: Optional[int] = None,
) -> list[Student]:
    """
    Filter students by multiple criteria.

    All parameters are optional. Only non-None parameters are used for filtering.

    Args:
        school: Filter by school.
        role: Filter by tactic role.
        weapon: Filter by weapon type.
        position: Filter by combat position.
        star_grade: Filter by star grade (1-3).

    Returns:
        List of students matching all specified criteria.

    Example:
        >>> # 3-star Gehenna DamageDealer with SR weapon
        >>> students = filter_students(
        ...     school="Gehenna",
        ...     role="DamageDealer",
        ...     weapon="SR",
        ...     star_grade=3
        ... )
    """
    students = list(cache_collection.students.values())

    if school is not None:
        students = [s for s in students if s.School == school]

    if role is not None:
        students = [s for s in students if s.TacticRole == role]

    if weapon is not None:
        students = [s for s in students if s.WeaponType == weapon]

    if position is not None:
        students = [s for s in students if s.Position == position]

    if star_grade is not None:
        students = [s for s in students if s.StarGrade == star_grade]

    return students
