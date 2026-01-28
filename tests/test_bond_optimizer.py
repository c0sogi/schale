"""Tests for bond optimization."""

import unittest

from schale.bond_progress import BondProgress
from schale.bond_optimizer import (
    calculate_stat_gain_for_level,
    find_best_student_for_next_level,
    optimize_bond_distribution_greedy,
    BondPool,
    create_bond_pool_with_alts,
)
from schale.schema.student import Student


class TestBondOptimizer(unittest.TestCase):
    """Test bond optimization functions."""

    def setUp(self):
        """Create test students."""
        # Create Hina variants with different stat gains
        self.hina_data = {
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
            "FavorStatValue": [
                [3, 38],
                [5, 63],
                [7, 88],
                [9, 106],
                [2, 18],
                [3, 29],
                [5, 47],
            ],
            "FavorAlts": [10022, 10086],
            "Equipment": ["Hat", "Gloves", "Shoes"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10004",
        }

        self.hina_swimsuit_data = self.hina_data.copy()
        self.hina_swimsuit_data.update(
            {
                "Id": 10022,
                "DevName": "Hina_Swimsuit",
                "Name": "Hina (Swimsuit)",
                "PathName": "Hina_Swimsuit",
                "TacticRole": "DamageDealer",
                "ArmorType": "LightArmor",
                "WeaponType": "SR",
                "FavorStatValue": [
                    [3, 0],
                    [5, 0],
                    [7, 43],
                    [9, 51],
                    [2, 8],
                    [3, 13],
                    [5, 21],
                ],
                "FavorAlts": [10004, 10086],
            }
        )

        self.hina = Student.model_validate(self.hina_data)
        self.hina_swimsuit = Student.model_validate(self.hina_swimsuit_data)

    def test_calculate_stat_gain(self):
        """Test stat gain calculation."""
        # Hina gains AttackPower from level 1 to 10
        gain = calculate_stat_gain_for_level(self.hina, 1, 10, "AttackPower")
        self.assertGreater(gain, 0)

        # Hina Swimsuit has different gains
        gain_swimsuit = calculate_stat_gain_for_level(
            self.hina_swimsuit, 1, 10, "AttackPower"
        )
        self.assertGreater(gain_swimsuit, 0)

    def test_calculate_stat_gain_same_level(self):
        """Test stat gain when levels are the same."""
        gain = calculate_stat_gain_for_level(self.hina, 10, 10, "AttackPower")
        self.assertEqual(gain, 0)

    def test_calculate_stat_gain_backwards(self):
        """Test stat gain when going backwards."""
        gain = calculate_stat_gain_for_level(self.hina, 20, 10, "AttackPower")
        self.assertEqual(gain, 0)

    def test_find_best_student(self):
        """Test finding best student for next level."""
        students = [self.hina, self.hina_swimsuit]

        # At level 10, find which student benefits more from level 11
        best = find_best_student_for_next_level(students, 10, "AttackPower")
        self.assertIsNotNone(best)
        self.assertIn(best, students)

    def test_find_best_student_at_max(self):
        """Test finding best student when at max level."""
        students = [self.hina, self.hina_swimsuit]
        best = find_best_student_for_next_level(students, 50, "AttackPower")
        self.assertIsNone(best)

    def test_optimize_greedy_basic(self):
        """Test basic greedy optimization."""
        students = [self.hina, self.hina_swimsuit]
        current = BondProgress(level=1, current_exp=0)

        # Optimize with 1000 EXP
        result = optimize_bond_distribution_greedy(
            students, current, 1000, "AttackPower"
        )

        self.assertGreater(result.final_level, current.level)
        self.assertEqual(result.exp_used + result.exp_remaining, 1000)
        self.assertGreater(result.stat_gain, 0)
        self.assertIn(result.target_student_id, [self.hina.Id, self.hina_swimsuit.Id])

    def test_optimize_greedy_maxhp(self):
        """Test optimization for MaxHP."""
        students = [self.hina, self.hina_swimsuit]
        current = BondProgress(level=10, current_exp=0)

        result = optimize_bond_distribution_greedy(students, current, 5000, "MaxHP")

        self.assertGreater(result.final_level, current.level)
        self.assertGreater(result.stat_gain, 0)

    def test_optimize_greedy_insufficient_exp(self):
        """Test optimization with insufficient EXP for level up."""
        students = [self.hina]
        current = BondProgress(level=10, current_exp=0)

        # Only 50 EXP (not enough for level 11)
        result = optimize_bond_distribution_greedy(students, current, 50, "AttackPower")

        # Should remain at level 10
        self.assertEqual(result.final_level, current.level)
        self.assertEqual(result.level_ups, 0)
        self.assertEqual(result.stat_gain, 0)

    def test_optimize_greedy_to_max_level(self):
        """Test optimization to max level."""
        students = [self.hina]
        current = BondProgress(level=45, current_exp=0)

        # Huge amount of EXP
        result = optimize_bond_distribution_greedy(
            students, current, 999999, "AttackPower"
        )

        self.assertEqual(result.final_level, 50)
        self.assertGreater(result.stat_gain, 0)

    def test_bond_pool_from_student(self):
        """Test creating bond pool from student."""
        pool = BondPool.from_student(self.hina, BondProgress(level=10))

        self.assertEqual(len(pool.students), 1)
        self.assertEqual(pool.shared_progress.level, 10)

    def test_bond_pool_optimize(self):
        """Test bond pool optimization."""
        pool = BondPool(
            students=[self.hina, self.hina_swimsuit],
            shared_progress=BondProgress(level=10),
        )

        result = pool.optimize("AttackPower", 5000)

        self.assertGreater(result.final_level, 10)
        self.assertGreater(result.stat_gain, 0)

    def test_create_bond_pool_with_alts(self):
        """Test creating bond pool with FavorAlts."""
        all_students = {
            self.hina.Id: self.hina,
            self.hina_swimsuit.Id: self.hina_swimsuit,
        }

        pool = create_bond_pool_with_alts(
            self.hina, all_students, BondProgress(level=20)
        )

        # Should include both Hina and Hina Swimsuit
        self.assertGreaterEqual(len(pool.students), 2)
        self.assertEqual(pool.shared_progress.level, 20)

    def test_optimize_invalid_stat(self):
        """Test optimization with invalid stat type."""
        students = [self.hina]
        current = BondProgress(level=10)

        with self.assertRaises(ValueError):
            optimize_bond_distribution_greedy(
                students, current, 1000, "InvalidStat"
            )

    def test_optimize_empty_student_list(self):
        """Test optimization with empty student list."""
        with self.assertRaises(ValueError):
            optimize_bond_distribution_greedy([], BondProgress(), 1000, "AttackPower")


if __name__ == "__main__":
    unittest.main()
