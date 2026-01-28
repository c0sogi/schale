"""
Bond distribution optimization for FavorAlts student pools.

This module provides algorithms to optimize bond item distribution across students
who share bond levels (via FavorAlts), maximizing specific stats.
"""

from dataclasses import dataclass
from typing import Optional

from schale.bond_progress import BondProgress
from schale.bond_exp_table import MAX_BOND_LEVEL, get_exp_for_next_level
from schale.schema.student import Student


@dataclass
class BondOptimizationResult:
    """
    Result of bond optimization.

    Attributes:
        target_student_id: ID of the student that maximizes the target stat
        final_level: Final bond level after optimization
        final_exp: Remaining EXP in final level
        exp_used: Total EXP consumed
        exp_remaining: EXP left over
        stat_gain: Total gain in target stat
        level_ups: Number of levels gained
    """

    target_student_id: int
    final_level: int
    final_exp: int
    exp_used: int
    exp_remaining: int
    stat_gain: int
    level_ups: int


def calculate_stat_gain_for_level(
    student: Student, from_level: int, to_level: int, stat_type: str
) -> int:
    """
    Calculate stat gain for a student when leveling from one bond level to another.

    Args:
        student: Student to calculate for
        from_level: Starting bond level
        to_level: Target bond level
        stat_type: Stat to calculate (e.g., "AttackPower", "MaxHP")

    Returns:
        Total stat gain from from_level to to_level

    Example:
        >>> gain = calculate_stat_gain_for_level(student, 10, 20, "AttackPower")
    """
    if from_level >= to_level:
        return 0

    stats_from = student.get_bond_stats(from_level)
    stats_to = student.get_bond_stats(to_level)

    return stats_to.get(stat_type, 0) - stats_from.get(stat_type, 0)


def find_best_student_for_next_level(
    students: list[Student], current_level: int, target_stat: str
) -> Optional[Student]:
    """
    Find which student gains the most of target stat when leveling up.

    Since FavorAlts students share bond levels, we want to know which student
    benefits most from the next level-up.

    Args:
        students: List of students in the bond pool
        current_level: Current shared bond level
        target_stat: Stat to maximize

    Returns:
        Student that gains the most target stat, or None if at max level

    Example:
        >>> best = find_best_student_for_next_level([hina, hina_swimsuit], 10, "AttackPower")
    """
    if current_level >= MAX_BOND_LEVEL:
        return None

    best_student = None
    best_gain = -1

    for student in students:
        gain = calculate_stat_gain_for_level(
            student, current_level, current_level + 1, target_stat
        )

        if gain > best_gain:
            best_gain = gain
            best_student = student

    return best_student


def optimize_bond_distribution_greedy(
    students: list[Student],
    current_progress: BondProgress,
    available_exp: int,
    target_stat: str,
) -> BondOptimizationResult:
    """
    Optimize bond EXP distribution using greedy algorithm.

    Since all FavorAlts students share bond levels, this finds which student
    benefits most from the available EXP and calculates the optimal final level.

    Strategy:
    - Level up as high as possible with available EXP
    - Track which student gains the most from each level
    - Return the student with maximum total stat gain

    Args:
        students: List of students sharing bond (FavorAlts pool)
        current_progress: Current shared bond progress
        available_exp: Total EXP available to spend
        target_stat: Stat to maximize (e.g., "AttackPower", "MaxHP")

    Returns:
        BondOptimizationResult with optimal distribution

    Example:
        >>> result = optimize_bond_distribution_greedy(
        ...     [hina, hina_swimsuit, hina_dress],
        ...     BondProgress(level=20, current_exp=0),
        ...     10000,
        ...     "AttackPower"
        ... )
        >>> print(f"Best student: {result.target_student_id}")
        >>> print(f"Final level: {result.final_level}")
        >>> print(f"Stat gain: {result.stat_gain}")
    """
    if not students:
        raise ValueError("Student list cannot be empty")

    if target_stat not in students[0].FavorStatType:
        # Check if any student has this stat
        has_stat = any(target_stat in s.FavorStatType for s in students)
        if not has_stat:
            raise ValueError(
                f"Target stat '{target_stat}' not found in any student's FavorStatType. "
                f"Available stats: {students[0].FavorStatType}"
            )

    # Calculate how high we can level with available EXP
    remaining_exp = current_progress.current_exp + available_exp
    exp_used = 0
    current_level = current_progress.level
    level_ups = 0

    # Level up as much as possible
    while current_level < MAX_BOND_LEVEL:
        exp_needed = get_exp_for_next_level(current_level)

        if remaining_exp >= exp_needed:
            remaining_exp -= exp_needed
            exp_used += exp_needed
            current_level += 1
            level_ups += 1
        else:
            # Not enough EXP for next level
            break

    final_level = current_level
    final_exp = remaining_exp

    # Find which student benefits most from this leveling
    best_student = None
    best_total_gain = -1

    for student in students:
        total_gain = calculate_stat_gain_for_level(
            student, current_progress.level, final_level, target_stat
        )

        if total_gain > best_total_gain:
            best_total_gain = total_gain
            best_student = student

    if best_student is None:
        # No level ups possible or all students at 0 gain
        best_student = students[0]
        best_total_gain = 0

    return BondOptimizationResult(
        target_student_id=best_student.Id,
        final_level=final_level,
        final_exp=final_exp,
        exp_used=exp_used,
        exp_remaining=available_exp - exp_used,
        stat_gain=best_total_gain,
        level_ups=level_ups,
    )


@dataclass
class BondPool:
    """
    Represents a pool of students sharing bond levels via FavorAlts.

    Attributes:
        students: List of students in the bond pool
        shared_progress: Current shared bond progress

    Example:
        >>> pool = BondPool.from_student(hina, BondProgress(level=20))
        >>> result = pool.optimize("AttackPower", 10000)
    """

    students: list[Student]
    shared_progress: BondProgress

    @classmethod
    def from_student(
        cls, student: Student, bond_progress: Optional[BondProgress] = None
    ) -> "BondPool":
        """
        Create a bond pool from a student and their FavorAlts.

        Args:
            student: Any student in the bond pool
            bond_progress: Shared bond progress (default: level 1)

        Returns:
            BondPool containing the student and all their FavorAlts

        Example:
            >>> from schale import get_student_by_path_name
            >>> hina = get_student_by_path_name("Hina")
            >>> pool = BondPool.from_student(hina, BondProgress(level=20))
            >>> print(f"Pool has {len(pool.students)} students")
        """
        if bond_progress is None:
            bond_progress = BondProgress(level=1, current_exp=0)

        # For now, just include the student itself
        # To include FavorAlts, we'd need access to cache_collection
        # This can be extended by the user if needed
        students = [student]

        return cls(students=students, shared_progress=bond_progress)

    def optimize(
        self, target_stat: str, available_exp: int, strategy: str = "greedy"
    ) -> BondOptimizationResult:
        """
        Optimize bond distribution to maximize target stat.

        Args:
            target_stat: Stat to maximize (must be in FavorStatType)
            available_exp: Total EXP available to distribute
            strategy: Optimization strategy ("greedy" only for now)

        Returns:
            BondOptimizationResult with optimal distribution

        Raises:
            ValueError: If strategy is unknown or target_stat invalid

        Example:
            >>> result = pool.optimize("AttackPower", 10000)
            >>> print(f"Optimal level: {result.final_level}")
            >>> print(f"Stat gain: +{result.stat_gain} {target_stat}")
        """
        if strategy != "greedy":
            raise ValueError(f"Unknown optimization strategy: {strategy}")

        return optimize_bond_distribution_greedy(
            self.students, self.shared_progress, available_exp, target_stat
        )


def create_bond_pool_with_alts(
    base_student: Student, all_students: dict[int, Student], bond_progress: BondProgress
) -> BondPool:
    """
    Create a bond pool including a student and all their FavorAlts.

    Args:
        base_student: Starting student
        all_students: Dict of all available students (from cache_collection.students)
        bond_progress: Shared bond progress

    Returns:
        BondPool with all students sharing bond

    Example:
        >>> from schale import cache_collection, get_student_by_path_name
        >>> hina = get_student_by_path_name("Hina")
        >>> pool = create_bond_pool_with_alts(
        ...     hina,
        ...     cache_collection.students,
        ...     BondProgress(level=20)
        ... )
        >>> # Pool now includes Hina, Hina (Swimsuit), Hina (Dress)
    """
    pool_students = [base_student]

    # Add all FavorAlts
    if base_student.FavorAlts:
        for alt_id in base_student.FavorAlts:
            if alt_id in all_students:
                pool_students.append(all_students[alt_id])

    return BondPool(students=pool_students, shared_progress=bond_progress)
