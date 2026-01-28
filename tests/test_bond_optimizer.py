"""Tests for bond optimization with independent student levels."""

import unittest

from schale.bond_progress import BondProgress
from schale.bond_optimizer import (
    BondOptimizationResult,
    BondPool,
    calculate_stat_gain_for_level,
    optimize_bond_distribution_greedy,
    create_bond_pool_with_alts,
)
from schale.schema.student import Student


class TestBondOptimizer(unittest.TestCase):
    """Test bond optimization algorithms with independent levels."""

    def setUp(self):
        """Set up test students (Hina variants)."""
        # Hina (Original) - Tank with good HP
        self.hina = Student.model_validate({
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
            "FavorAlts": [10022, 10086],
            "Equipment": ["Hat", "Gloves", "Shoes"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10004",
        })

        # Hina (Swimsuit) - Damage dealer
        self.hina_swimsuit = Student.model_validate({
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
        })

    def test_calculate_stat_gain(self):
        """Test stat gain calculation."""
        gain = calculate_stat_gain_for_level(self.hina, 10, 20, "AttackPower")
        self.assertGreater(gain, 0)

        # Same level = 0 gain
        gain_zero = calculate_stat_gain_for_level(self.hina, 10, 10, "AttackPower")
        self.assertEqual(gain_zero, 0)

        # Backward = 0 gain
        gain_back = calculate_stat_gain_for_level(self.hina, 20, 10, "AttackPower")
        self.assertEqual(gain_back, 0)

    def test_optimize_greedy_basic(self):
        """Test basic greedy optimization with independent levels."""
        # Start with different levels
        current_levels = {
            10004: BondProgress(level=20, current_exp=0),  # Hina
            10022: BondProgress(level=15, current_exp=0),  # Hina Swimsuit
        }

        result = optimize_bond_distribution_greedy(
            students=[self.hina, self.hina_swimsuit],
            current_levels=current_levels,
            available_exp=5000,
            target_stat="AttackPower"
        )

        # Should allocate to student with max level (Hina, level 20)
        self.assertGreater(result.exp_allocation[10004], 0)
        self.assertGreaterEqual(result.final_levels[10004], 20)  # Should level up or stay same
        # Max level should increase
        self.assertGreaterEqual(result.max_level, 20)

    def test_optimize_greedy_equal_levels(self):
        """Test optimization when all students at same level."""
        current_levels = {
            10004: BondProgress(level=10, current_exp=0),
            10022: BondProgress(level=10, current_exp=0),
        }

        result = optimize_bond_distribution_greedy(
            students=[self.hina, self.hina_swimsuit],
            current_levels=current_levels,
            available_exp=3000,
            target_stat="AttackPower"
        )

        # Should pick one and level it up
        total_allocated = sum(result.exp_allocation.values())
        self.assertGreater(total_allocated, 0)
        self.assertGreaterEqual(result.max_level, 10)

    def test_optimize_greedy_insufficient_exp(self):
        """Test optimization with insufficient EXP for level up."""
        current_levels = {
            10004: BondProgress(level=20, current_exp=0),
        }

        result = optimize_bond_distribution_greedy(
            students=[self.hina],
            current_levels=current_levels,
            available_exp=100,  # Not enough for level 21
            target_stat="AttackPower"
        )

        # Should not level up
        self.assertEqual(result.final_levels[10004], 20)
        self.assertEqual(result.stat_gain, 0)
        self.assertEqual(result.exp_used, 0)

    def test_optimize_greedy_maxhp(self):
        """Test optimization for MaxHP."""
        current_levels = {
            10004: BondProgress(level=10, current_exp=0),
        }

        result = optimize_bond_distribution_greedy(
            students=[self.hina],
            current_levels=current_levels,
            available_exp=10000,
            target_stat="MaxHP"
        )

        self.assertGreater(result.stat_gain, 0)
        self.assertGreater(result.final_levels[10004], 10)

    def test_optimize_invalid_stat(self):
        """Test optimization with invalid stat type."""
        current_levels = {
            10004: BondProgress(level=10, current_exp=0),
        }

        with self.assertRaises(ValueError):
            optimize_bond_distribution_greedy(
                students=[self.hina],
                current_levels=current_levels,
                available_exp=1000,
                target_stat="InvalidStat"
            )

    def test_optimize_empty_student_list(self):
        """Test optimization with empty student list."""
        with self.assertRaises(ValueError):
            optimize_bond_distribution_greedy(
                students=[],
                current_levels={},
                available_exp=1000,
                target_stat="AttackPower"
            )

    def test_bond_pool_creation(self):
        """Test BondPool creation."""
        student_progress = {
            10004: BondProgress(level=20, current_exp=0),
            10022: BondProgress(level=15, current_exp=0),
        }

        pool = BondPool(
            students=[self.hina, self.hina_swimsuit],
            student_progress=student_progress
        )

        self.assertEqual(len(pool.students), 2)
        self.assertEqual(pool.max_level, 20)  # Max among 20 and 15

    def test_bond_pool_max_level(self):
        """Test BondPool max_level property."""
        student_progress = {
            10004: BondProgress(level=25, current_exp=0),
            10022: BondProgress(level=30, current_exp=0),
        }

        pool = BondPool(
            students=[self.hina, self.hina_swimsuit],
            student_progress=student_progress
        )

        self.assertEqual(pool.max_level, 30)

    def test_bond_pool_get_stat_bonus(self):
        """Test getting stat bonus based on max level."""
        student_progress = {
            10004: BondProgress(level=10, current_exp=0),
            10022: BondProgress(level=15, current_exp=0),
        }

        pool = BondPool(
            students=[self.hina, self.hina_swimsuit],
            student_progress=student_progress
        )

        # Stat bonus should be based on level 15 (max)
        bonus = pool.get_stat_bonus("AttackPower")
        expected_bonus = self.hina.get_bond_stats(15)["AttackPower"]
        self.assertEqual(bonus, expected_bonus)

    def test_bond_pool_optimize(self):
        """Test BondPool.optimize method."""
        student_progress = {
            10004: BondProgress(level=10, current_exp=0),
            10022: BondProgress(level=8, current_exp=0),
        }

        pool = BondPool(
            students=[self.hina, self.hina_swimsuit],
            student_progress=student_progress
        )

        result = pool.optimize("AttackPower", 5000)

        self.assertIsInstance(result, BondOptimizationResult)
        self.assertGreaterEqual(result.max_level, 10)
        self.assertGreaterEqual(result.stat_gain, 0)

    def test_create_bond_pool_with_alts(self):
        """Test creating bond pool with FavorAlts - new API."""
        # Create test students dict
        test_students = {
            10004: self.hina,
            10022: self.hina_swimsuit,
        }

        pool = create_bond_pool_with_alts(
            self.hina,
            initial_levels={
                "Hina": 20,
                "Hina (Swimsuit)": 15,
            },
            _test_students=test_students
        )

        self.assertEqual(len(pool.students), 2)  # Hina + Swimsuit
        self.assertEqual(pool.max_level, 20)

    def test_create_bond_pool_no_initial_progress(self):
        """Test creating bond pool without initial progress - new API."""
        test_students = {
            10004: self.hina,
            10022: self.hina_swimsuit,
        }

        pool = create_bond_pool_with_alts(
            self.hina,
            _test_students=test_students
        )

        # Should default to level 1
        self.assertEqual(pool.max_level, 1)
        for progress in pool.student_progress.values():
            self.assertEqual(progress.level, 1)

    def test_optimize_allocates_to_max_level_student(self):
        """Test that optimization prioritizes student with max level."""
        current_levels = {
            10004: BondProgress(level=25, current_exp=0),  # Max
            10022: BondProgress(level=20, current_exp=0),
        }

        result = optimize_bond_distribution_greedy(
            students=[self.hina, self.hina_swimsuit],
            current_levels=current_levels,
            available_exp=3000,
            target_stat="AttackPower"
        )

        # Should allocate primarily to student 10004 (max level)
        self.assertGreater(result.exp_allocation[10004], result.exp_allocation.get(10022, 0))


if __name__ == "__main__":
    unittest.main()
