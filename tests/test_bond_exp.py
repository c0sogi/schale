"""Tests for bond EXP system."""

import unittest

from schale.bond_exp_table import (
    MIN_BOND_LEVEL,
    MAX_BOND_LEVEL,
    get_cumulative_exp_for_level,
    get_exp_for_next_level,
    calculate_level_from_total_exp,
)
from schale.bond_progress import BondProgress


class TestBondEXPTable(unittest.TestCase):
    """Test bond EXP table functions."""

    def test_get_cumulative_exp_for_level(self):
        """Test cumulative EXP retrieval."""
        # Level 1 should be 0
        self.assertEqual(get_cumulative_exp_for_level(1), 0)

        # Level 2 should be 100
        self.assertEqual(get_cumulative_exp_for_level(2), 100)

        # Level 50 should be the max
        max_exp = get_cumulative_exp_for_level(50)
        self.assertGreater(max_exp, 0)

    def test_get_cumulative_exp_invalid_level(self):
        """Test error handling for invalid levels."""
        with self.assertRaises(ValueError):
            get_cumulative_exp_for_level(0)

        with self.assertRaises(ValueError):
            get_cumulative_exp_for_level(51)

    def test_get_exp_for_next_level(self):
        """Test EXP required for next level."""
        # Level 1->2 should be 100
        self.assertEqual(get_exp_for_next_level(1), 100)

        # Level 49->50
        exp_needed = get_exp_for_next_level(49)
        self.assertGreater(exp_needed, 0)

    def test_get_exp_for_next_level_at_max(self):
        """Test error when already at max level."""
        with self.assertRaises(ValueError):
            get_exp_for_next_level(50)

    def test_calculate_level_from_total_exp_zero(self):
        """Test level calculation with 0 EXP."""
        level, remaining = calculate_level_from_total_exp(0)
        self.assertEqual(level, 1)
        self.assertEqual(remaining, 0)

    def test_calculate_level_from_total_exp_exact_level(self):
        """Test level calculation at exact level thresholds."""
        # Exactly level 2 (100 EXP)
        level, remaining = calculate_level_from_total_exp(100)
        self.assertEqual(level, 2)
        self.assertEqual(remaining, 0)

        # Exactly level 10 (2700 EXP)
        level, remaining = calculate_level_from_total_exp(2700)
        self.assertEqual(level, 10)
        self.assertEqual(remaining, 0)

    def test_calculate_level_from_total_exp_partial(self):
        """Test level calculation with partial EXP."""
        # 150 EXP = Level 2 + 50 remaining
        level, remaining = calculate_level_from_total_exp(150)
        self.assertEqual(level, 2)
        self.assertEqual(remaining, 50)

    def test_calculate_level_from_total_exp_max(self):
        """Test level calculation at max level."""
        very_high_exp = 999999999
        level, remaining = calculate_level_from_total_exp(very_high_exp)
        self.assertEqual(level, MAX_BOND_LEVEL)
        self.assertEqual(remaining, 0)


class TestBondProgress(unittest.TestCase):
    """Test BondProgress class."""

    def test_creation_default(self):
        """Test default creation."""
        progress = BondProgress()
        self.assertEqual(progress.level, 1)
        self.assertEqual(progress.current_exp, 0)

    def test_creation_with_params(self):
        """Test creation with parameters."""
        progress = BondProgress(level=10, current_exp=100)
        self.assertEqual(progress.level, 10)
        self.assertEqual(progress.current_exp, 100)

    def test_creation_invalid_level(self):
        """Test error handling for invalid level."""
        with self.assertRaises(ValueError):
            BondProgress(level=0)

        with self.assertRaises(ValueError):
            BondProgress(level=51)

    def test_creation_negative_exp(self):
        """Test error handling for negative EXP."""
        with self.assertRaises(ValueError):
            BondProgress(level=1, current_exp=-10)

    def test_from_total_exp(self):
        """Test creation from total EXP."""
        # 0 EXP
        progress = BondProgress.from_total_exp(0)
        self.assertEqual(progress.level, 1)
        self.assertEqual(progress.current_exp, 0)

        # 150 EXP
        progress = BondProgress.from_total_exp(150)
        self.assertEqual(progress.level, 2)
        self.assertEqual(progress.current_exp, 50)

        # 2700 EXP (exactly level 10)
        progress = BondProgress.from_total_exp(2700)
        self.assertEqual(progress.level, 10)
        self.assertEqual(progress.current_exp, 0)

    def test_total_exp_property(self):
        """Test total_exp property."""
        progress = BondProgress(level=10, current_exp=300)
        total = progress.total_exp
        self.assertEqual(total, 3000)  # 2700 (level 10 base) + 300

    def test_add_exp(self):
        """Test adding EXP."""
        progress = BondProgress(level=1, current_exp=0)

        # Add 150 EXP -> should reach level 2 with 50 remaining
        new_progress = progress.add_exp(150)
        self.assertEqual(new_progress.level, 2)
        self.assertEqual(new_progress.current_exp, 50)

        # Original should be unchanged (immutable)
        self.assertEqual(progress.level, 1)
        self.assertEqual(progress.current_exp, 0)

    def test_add_operator(self):
        """Test + operator."""
        progress = BondProgress(level=1, current_exp=0)
        new_progress = progress + 150

        self.assertEqual(new_progress.level, 2)
        self.assertEqual(new_progress.current_exp, 50)

    def test_sub_operator(self):
        """Test - operator."""
        progress = BondProgress(level=10, current_exp=100)
        new_progress = progress - 200

        # Should go back a level
        self.assertLess(new_progress.total_exp, progress.total_exp)

    def test_add_large_exp(self):
        """Test adding large amount of EXP."""
        progress = BondProgress(level=1, current_exp=0)
        new_progress = progress + 999999  # Way more than needed for max level

        self.assertEqual(new_progress.level, MAX_BOND_LEVEL)

    def test_is_max_level(self):
        """Test max level detection."""
        progress = BondProgress(level=MAX_BOND_LEVEL, current_exp=0)
        self.assertTrue(progress.is_max_level)

        progress = BondProgress(level=10, current_exp=0)
        self.assertFalse(progress.is_max_level)

    def test_exp_to_next_level(self):
        """Test EXP needed for next level."""
        progress = BondProgress(level=1, current_exp=50)

        # Level 1->2 needs 100 total, 50 already gained
        self.assertEqual(progress.exp_to_next_level, 50)

    def test_exp_to_next_level_at_max(self):
        """Test EXP to next level when at max."""
        progress = BondProgress(level=MAX_BOND_LEVEL, current_exp=0)
        self.assertEqual(progress.exp_to_next_level, 0)

    def test_str_representation(self):
        """Test string representation."""
        progress = BondProgress(level=10, current_exp=100)
        str_repr = str(progress)
        self.assertIn("10", str_repr)
        self.assertIn("100", str_repr)

    def test_immutability(self):
        """Test that BondProgress is immutable."""
        progress = BondProgress(level=10, current_exp=100)

        # Should not be able to modify
        with self.assertRaises(Exception):  # dataclass frozen raises FrozenInstanceError
            progress.level = 20  # type: ignore


if __name__ == "__main__":
    unittest.main()
