"""
Bond distribution optimization for FavorAlts student pools.

This module provides algorithms to optimize bond item distribution across students
who share stat bonuses via FavorAlts. Each student has independent bond levels,
but stat bonuses apply based on the maximum level among FavorAlts.

Key Concepts:
- Each student has their own bond level (e.g., Hina: 20, Hina_Swimsuit: 15)
- Stat bonuses apply to all FavorAlts based on the MAXIMUM level (e.g., all get level 20 bonus)
- EXP can be distributed independently to each student
- Optimization goal: Distribute total EXP to maximize target stat bonus
"""

from dataclasses import dataclass
from typing import Optional

from schale.bond_progress import BondProgress
from schale.bond_exp_table import MAX_BOND_LEVEL, get_exp_for_next_level
from schale.schema.student import Student


@dataclass
class BondOptimizationResult:
    """
    Result of bond optimization for independent student levels.

    Attributes:
        exp_allocation: Dict mapping student_id -> EXP to allocate to that student
        final_levels: Dict mapping student_id -> final bond level after allocation
        max_level: Maximum level among all students (determines stat bonus)
        exp_used: Total EXP consumed
        exp_remaining: EXP left over
        stat_gain: Total gain in target stat from current max level to new max level
    """

    exp_allocation: dict[int, int]
    final_levels: dict[int, int]
    max_level: int
    exp_used: int
    exp_remaining: int
    stat_gain: int


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


def optimize_bond_distribution_greedy(
    students: list[Student],
    current_levels: dict[int, BondProgress],
    available_exp: int,
    target_stat: str,
) -> BondOptimizationResult:
    """
    Optimize bond EXP distribution using greedy algorithm.

    Each student has independent bond levels, but stat bonuses apply based on the
    maximum level among FavorAlts. This algorithm distributes EXP to maximize
    the maximum level, thereby maximizing stat bonuses.

    Strategy:
    - Always level up the student(s) with current maximum level
    - This ensures the stat bonus level increases as quickly as possible
    - Ties are broken by choosing the student needing least EXP to next level

    Args:
        students: List of students sharing stat bonuses (FavorAlts pool)
        current_levels: Dict mapping student_id -> current BondProgress
        available_exp: Total EXP available to distribute
        target_stat: Stat to maximize (e.g., "AttackPower", "MaxHP")

    Returns:
        BondOptimizationResult with optimal EXP allocation

    Example:
        >>> result = optimize_bond_distribution_greedy(
        ...     [hina, hina_swimsuit, hina_dress],
        ...     {10004: BondProgress(20, 0), 10022: BondProgress(15, 0), 10086: BondProgress(30, 0)},
        ...     10000,
        ...     "AttackPower"
        ... )
        >>> print(f"Max level: {result.max_level}")
        >>> print(f"Stat gain: {result.stat_gain}")
        >>> print(f"Allocations: {result.exp_allocation}")
    """
    if not students:
        raise ValueError("Student list cannot be empty")

    if not current_levels:
        raise ValueError("Current levels dict cannot be empty")

    # Verify all students are in current_levels
    for student in students:
        if student.Id not in current_levels:
            raise ValueError(f"Student {student.Id} not found in current_levels")

    if target_stat not in students[0].FavorStatType:
        # Check if any student has this stat
        has_stat = any(target_stat in s.FavorStatType for s in students)
        if not has_stat:
            raise ValueError(
                f"Target stat '{target_stat}' not found in any student's FavorStatType. "
                f"Available stats: {students[0].FavorStatType}"
            )

    # Initialize tracking
    current_max_level = max(p.level for p in current_levels.values())
    working_progress = {sid: progress for sid, progress in current_levels.items()}
    exp_allocation = {student.Id: 0 for student in students}
    remaining_exp = available_exp

    # Greedy allocation: always level up the student(s) with max level
    while remaining_exp > 0:
        # Find current max level
        current_max = max(p.level for p in working_progress.values())

        # If all students at max level, stop
        if current_max >= MAX_BOND_LEVEL:
            break

        # Find students at max level that can still level up
        candidates = [
            student
            for student in students
            if working_progress[student.Id].level == current_max
            and working_progress[student.Id].level < MAX_BOND_LEVEL
        ]

        if not candidates:
            break

        # Among candidates, choose the one needing least EXP to next level
        best_student = None
        min_exp_needed = float("inf")

        for student in candidates:
            progress = working_progress[student.Id]
            exp_needed = progress.exp_to_next_level

            if exp_needed < min_exp_needed:
                min_exp_needed = exp_needed
                best_student = student

        if best_student is None or min_exp_needed > remaining_exp:
            # Can't afford any more level-ups
            break

        # Allocate EXP to level up this student
        student_id = best_student.Id
        exp_allocation[student_id] += min_exp_needed
        remaining_exp -= min_exp_needed
        working_progress[student_id] = working_progress[student_id] + min_exp_needed

    # Calculate final results
    final_levels = {sid: progress.level for sid, progress in working_progress.items()}
    new_max_level = max(final_levels.values())
    exp_used = available_exp - remaining_exp

    # Calculate stat gain based on max level increase
    # Use any student since all FavorAlts have same stat bonus structure
    reference_student = students[0]
    stat_gain = calculate_stat_gain_for_level(
        reference_student, current_max_level, new_max_level, target_stat
    )

    return BondOptimizationResult(
        exp_allocation=exp_allocation,
        final_levels=final_levels,
        max_level=new_max_level,
        exp_used=exp_used,
        exp_remaining=remaining_exp,
        stat_gain=stat_gain,
    )


@dataclass
class BondPool:
    """
    Represents a pool of students sharing stat bonuses via FavorAlts.

    Each student has independent bond levels, but all receive stat bonuses
    based on the maximum level among the pool.

    Attributes:
        students: List of students in the bond pool
        student_progress: Dict mapping student_id -> current BondProgress

    Example:
        >>> pool = BondPool(
        ...     students=[hina, hina_swimsuit],
        ...     student_progress={
        ...         10004: BondProgress(20, 0),
        ...         10022: BondProgress(15, 0)
        ...     }
        ... )
        >>> result = pool.optimize("AttackPower", 10000)
    """

    students: list[Student]
    student_progress: dict[int, BondProgress]

    def __post_init__(self):
        """Validate that all students have progress entries."""
        for student in self.students:
            if student.Id not in self.student_progress:
                raise ValueError(
                    f"Student {student.Id} ({student.Name}) not found in student_progress"
                )

    @property
    def max_level(self) -> int:
        """Get the maximum bond level among all students (determines stat bonus)."""
        return max(progress.level for progress in self.student_progress.values())

    def get_stat_bonus(self, stat_type: str) -> int:
        """
        Get current stat bonus based on maximum level.

        Args:
            stat_type: Stat to query (e.g., "AttackPower", "MaxHP")

        Returns:
            Stat bonus value at current max level
        """
        max_level = self.max_level
        # All FavorAlts have same stat bonus structure, use any student
        reference_student = self.students[0]
        return reference_student.get_bond_stats(max_level).get(stat_type, 0)

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
            BondOptimizationResult with optimal EXP allocation

        Raises:
            ValueError: If strategy is unknown or target_stat invalid

        Example:
            >>> result = pool.optimize("AttackPower", 10000)
            >>> print(f"Max level: {result.max_level}")
            >>> print(f"Stat gain: +{result.stat_gain} {target_stat}")
            >>> for student_id, exp in result.exp_allocation.items():
            ...     print(f"Student {student_id}: +{exp} EXP")
        """
        if strategy != "greedy":
            raise ValueError(f"Unknown optimization strategy: {strategy}")

        return optimize_bond_distribution_greedy(
            self.students, self.student_progress, available_exp, target_stat
        )


def create_bond_pool_with_alts(
    base_student: Student,
    all_students: dict[int, Student],
    initial_progress: Optional[dict[int, BondProgress]] = None,
) -> BondPool:
    """
    Create a bond pool including a student and all their FavorAlts.

    Args:
        base_student: Starting student
        all_students: Dict of all available students (from cache_collection.students)
        initial_progress: Optional dict of initial progress for each student.
                         If None, all students start at level 1.

    Returns:
        BondPool with all students sharing stat bonuses

    Example:
        >>> from schale import cache_collection, get_student_by_path_name
        >>> from schale.bond_progress import BondProgress
        >>> hina = get_student_by_path_name("Hina")
        >>> pool = create_bond_pool_with_alts(
        ...     hina,
        ...     cache_collection.students,
        ...     initial_progress={
        ...         10004: BondProgress(20, 0),
        ...         10022: BondProgress(15, 0),
        ...         10086: BondProgress(30, 0),
        ...     }
        ... )
        >>> # Pool now includes Hina (20), Hina_Swimsuit (15), Hina_Dress (30)
        >>> # Max level is 30, so all get level 30 stat bonuses
    """
    pool_students = [base_student]

    # Add all FavorAlts
    if base_student.FavorAlts:
        for alt_id in base_student.FavorAlts:
            if alt_id in all_students:
                pool_students.append(all_students[alt_id])

    # Initialize progress
    if initial_progress is None:
        student_progress = {
            student.Id: BondProgress(level=1, current_exp=0)
            for student in pool_students
        }
    else:
        # Use provided progress, fill missing with level 1
        student_progress = {}
        for student in pool_students:
            if student.Id in initial_progress:
                student_progress[student.Id] = initial_progress[student.Id]
            else:
                student_progress[student.Id] = BondProgress(level=1, current_exp=0)

    return BondPool(students=pool_students, student_progress=student_progress)
