"""
Bond (Affection) EXP table for Blue Archive.

This module contains the experience requirements for each bond level from 1 to 50.

Note: These values are based on community data and typical progression curves.
They should be validated against actual game data when available.

Sources:
- Blue Archive Wiki: https://bluearchive.wiki/wiki/Affection
- Community calculators: http://sensei.lol/bondEXPCalc.html
"""

# Cumulative EXP required to reach each bond level
# Index 0 = Level 1 (0 EXP), Index 1 = Level 2, etc.
# These values represent the TOTAL EXP needed from level 1 to reach that level
BOND_LEVEL_CUMULATIVE_EXP: list[int] = [
    0,  # Level 1 (starting level)
    100,  # Level 2
    250,  # Level 3
    450,  # Level 4
    700,  # Level 5
    1000,  # Level 6
    1350,  # Level 7
    1750,  # Level 8
    2200,  # Level 9
    2700,  # Level 10
    3250,  # Level 11
    3900,  # Level 12
    4650,  # Level 13
    5500,  # Level 14
    6450,  # Level 15
    7500,  # Level 16
    8700,  # Level 17
    10050,  # Level 18
    11550,  # Level 19
    13200,  # Level 20
    15000,  # Level 21
    17000,  # Level 22
    19200,  # Level 23
    21600,  # Level 24
    24200,  # Level 25
    27000,  # Level 26
    30000,  # Level 27
    33300,  # Level 28
    36900,  # Level 29
    40800,  # Level 30
    45000,  # Level 31
    49600,  # Level 32
    54600,  # Level 33
    60000,  # Level 34
    65800,  # Level 35
    72000,  # Level 36
    78700,  # Level 37
    85900,  # Level 38
    93600,  # Level 39
    101800,  # Level 40
    110500,  # Level 41
    120000,  # Level 42
    130200,  # Level 43
    141100,  # Level 44
    152700,  # Level 45
    165000,  # Level 46
    178200,  # Level 47
    192300,  # Level 48
    207300,  # Level 49
    223200,  # Level 50 (max level with stat bonuses)
]

# Constants
MIN_BOND_LEVEL = 1
MAX_BOND_LEVEL = 50
MAX_TOTAL_EXP = BOND_LEVEL_CUMULATIVE_EXP[MAX_BOND_LEVEL - 1]


def get_cumulative_exp_for_level(level: int) -> int:
    """
    Get the cumulative EXP required to reach a specific bond level.

    Args:
        level: Bond level (1-50)

    Returns:
        Total cumulative EXP needed to reach this level from level 1.

    Raises:
        ValueError: If level is out of valid range (1-50)

    Example:
        >>> get_cumulative_exp_for_level(10)
        2700
        >>> get_cumulative_exp_for_level(50)
        223200
    """
    if not MIN_BOND_LEVEL <= level <= MAX_BOND_LEVEL:
        raise ValueError(
            f"Bond level must be between {MIN_BOND_LEVEL} and {MAX_BOND_LEVEL}, got {level}"
        )

    return BOND_LEVEL_CUMULATIVE_EXP[level - 1]


def get_exp_for_next_level(current_level: int) -> int:
    """
    Get the EXP required to go from current level to the next level.

    Args:
        current_level: Current bond level (1-49)

    Returns:
        EXP needed to reach the next level.

    Raises:
        ValueError: If already at max level or level is invalid

    Example:
        >>> get_exp_for_next_level(1)
        100
        >>> get_exp_for_next_level(49)
        15900
    """
    if current_level >= MAX_BOND_LEVEL:
        raise ValueError(f"Already at max bond level ({MAX_BOND_LEVEL})")

    if not MIN_BOND_LEVEL <= current_level < MAX_BOND_LEVEL:
        raise ValueError(
            f"Bond level must be between {MIN_BOND_LEVEL} and {MAX_BOND_LEVEL - 1}, got {current_level}"
        )

    return (
        BOND_LEVEL_CUMULATIVE_EXP[current_level]
        - BOND_LEVEL_CUMULATIVE_EXP[current_level - 1]
    )


def calculate_level_from_total_exp(total_exp: int) -> tuple[int, int]:
    """
    Calculate bond level and remaining EXP from total cumulative EXP.

    This function determines:
    1. What level the total EXP corresponds to
    2. How much EXP remains toward the next level

    Args:
        total_exp: Total cumulative EXP earned

    Returns:
        Tuple of (level, remaining_exp_in_current_level)
        - level: Current bond level (1-50)
        - remaining_exp: EXP progress within current level (0 to exp_for_next_level)

    Example:
        >>> calculate_level_from_total_exp(0)
        (1, 0)
        >>> calculate_level_from_total_exp(150)  # Past level 2 (100 EXP)
        (2, 50)
        >>> calculate_level_from_total_exp(2700)  # Exactly level 10
        (10, 0)
        >>> calculate_level_from_total_exp(3000)  # Level 10 + 300 EXP
        (10, 300)
    """
    # Clamp total_exp to valid range
    if total_exp < 0:
        total_exp = 0
    elif total_exp >= MAX_TOTAL_EXP:
        return (MAX_BOND_LEVEL, 0)

    # Binary search for the level
    for level in range(1, MAX_BOND_LEVEL + 1):
        level_exp = BOND_LEVEL_CUMULATIVE_EXP[level - 1]

        if total_exp < level_exp:
            # We haven't reached this level yet
            # So we're at the previous level with some remaining EXP
            prev_level = level - 1
            prev_level_exp = (
                BOND_LEVEL_CUMULATIVE_EXP[prev_level - 1] if prev_level > 0 else 0
            )
            remaining_exp = total_exp - prev_level_exp

            return (prev_level, remaining_exp)

    # If we get here, we're at max level
    return (MAX_BOND_LEVEL, 0)
