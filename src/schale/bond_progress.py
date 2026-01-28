"""
Bond progress tracking and management.

This module provides the BondProgress class for tracking bond level and experience,
with support for adding experience via the __add__ operator.
"""

from dataclasses import dataclass

from schale.bond_exp_table import (
    MIN_BOND_LEVEL,
    MAX_BOND_LEVEL,
    get_cumulative_exp_for_level,
    get_exp_for_next_level,
    calculate_level_from_total_exp,
)


@dataclass(frozen=True)
class BondProgress:
    """
    Immutable representation of bond level progress.

    Tracks the current bond level and experience progress within that level.
    Supports adding experience via the + and += operators.

    Attributes:
        level: Current bond level (1-50)
        current_exp: Experience points within the current level
                     (0 to exp_required_for_next_level)

    Examples:
        >>> # Create a bond progress at level 10 with 300 EXP
        >>> progress = BondProgress(level=10, current_exp=300)

        >>> # Add 200 more EXP
        >>> new_progress = progress + 200
        >>> print(new_progress.level, new_progress.current_exp)

        >>> # Create from total cumulative EXP
        >>> progress2 = BondProgress.from_total_exp(5000)
        >>> print(progress2.level, progress2.current_exp)

        >>> # Check total EXP
        >>> total = progress.total_exp
    """

    level: int = 1
    current_exp: int = 0

    def __post_init__(self) -> None:
        """Validate bond progress parameters."""
        if not MIN_BOND_LEVEL <= self.level <= MAX_BOND_LEVEL:
            raise ValueError(
                f"Bond level must be between {MIN_BOND_LEVEL} and {MAX_BOND_LEVEL}, "
                f"got {self.level}"
            )

        if self.current_exp < 0:
            raise ValueError(f"Current EXP cannot be negative, got {self.current_exp}")

        # Validate that current_exp doesn't exceed what's needed for next level
        if self.level < MAX_BOND_LEVEL:
            max_exp_for_level = get_exp_for_next_level(self.level)
            if self.current_exp > max_exp_for_level:
                raise ValueError(
                    f"Current EXP ({self.current_exp}) exceeds maximum for level {self.level} "
                    f"({max_exp_for_level}). Use add_exp() or from_total_exp() instead."
                )

    @classmethod
    def from_total_exp(cls, total_exp: int) -> "BondProgress":
        """
        Create BondProgress from total cumulative experience.

        Automatically calculates the correct level and remaining EXP.

        Args:
            total_exp: Total cumulative EXP earned (0+)

        Returns:
            BondProgress instance at the appropriate level

        Example:
            >>> progress = BondProgress.from_total_exp(2700)
            >>> print(f"Level {progress.level}, EXP {progress.current_exp}")
            Level 10, EXP 0

            >>> progress = BondProgress.from_total_exp(3000)
            >>> print(f"Level {progress.level}, EXP {progress.current_exp}")
            Level 10, EXP 300
        """
        level, current_exp = calculate_level_from_total_exp(total_exp)
        return cls(level=level, current_exp=current_exp)

    @property
    def total_exp(self) -> int:
        """
        Calculate total cumulative EXP represented by this progress.

        Returns:
            Total EXP from level 1 to current position

        Example:
            >>> progress = BondProgress(level=10, current_exp=300)
            >>> progress.total_exp
            3000
        """
        base_exp = get_cumulative_exp_for_level(self.level)
        return base_exp + self.current_exp

    @property
    def exp_to_next_level(self) -> int:
        """
        Get EXP needed to reach the next level.

        Returns:
            EXP remaining until next level, or 0 if at max level

        Example:
            >>> progress = BondProgress(level=10, current_exp=300)
            >>> progress.exp_to_next_level
            250  # Assuming level 10->11 needs 550 total, 300 already gained
        """
        if self.level >= MAX_BOND_LEVEL:
            return 0

        total_needed = get_exp_for_next_level(self.level)
        return total_needed - self.current_exp

    @property
    def is_max_level(self) -> bool:
        """Check if at maximum bond level."""
        return self.level >= MAX_BOND_LEVEL

    def add_exp(self, exp: int) -> "BondProgress":
        """
        Add experience and return new BondProgress.

        This method is immutable - it returns a new instance rather than
        modifying the current one.

        Args:
            exp: Amount of EXP to add (can be negative)

        Returns:
            New BondProgress instance with updated level and EXP

        Example:
            >>> progress = BondProgress(level=1, current_exp=0)
            >>> new_progress = progress.add_exp(150)
            >>> print(f"Level {new_progress.level}, EXP {new_progress.current_exp}")
            Level 2, EXP 50
        """
        new_total_exp = max(0, self.total_exp + exp)
        return BondProgress.from_total_exp(new_total_exp)

    def __add__(self, exp: int) -> "BondProgress":
        """
        Add experience using the + operator.

        Args:
            exp: Amount of EXP to add

        Returns:
            New BondProgress instance

        Example:
            >>> progress = BondProgress(level=10, current_exp=100)
            >>> new_progress = progress + 500
            >>> new_progress.level
            11
        """
        if not isinstance(exp, int):
            return NotImplemented
        return self.add_exp(exp)

    def __sub__(self, exp: int) -> "BondProgress":
        """
        Subtract experience using the - operator.

        Args:
            exp: Amount of EXP to subtract

        Returns:
            New BondProgress instance (minimum level 1, exp 0)

        Example:
            >>> progress = BondProgress(level=10, current_exp=100)
            >>> new_progress = progress - 200
            >>> new_progress.level
            9
        """
        if not isinstance(exp, int):
            return NotImplemented
        return self.add_exp(-exp)

    def __str__(self) -> str:
        """String representation of bond progress."""
        if self.is_max_level:
            return f"Bond Level {self.level} (MAX)"
        return f"Bond Level {self.level} ({self.current_exp}/{self.exp_to_next_level + self.current_exp} EXP)"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"BondProgress(level={self.level}, current_exp={self.current_exp}, total_exp={self.total_exp})"
