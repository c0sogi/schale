"""Unit tests for bond stat calculation without network dependencies."""

import unittest

from schale.schema.student import Student


class TestBondCalculation(unittest.TestCase):
    """Test bond stat calculation logic directly."""

    def setUp(self):
        """Create a test student with known bond stat values."""
        # Hina (Dress) data based on real SchaleDB data
        self.hina_data = {
            "Id": 10086,
            "DevName": "Hina_Dress",
            "Name": "Hina (Dress)",
            "PathName": "Hina_Dress",
            "IsReleased": [True, True, True],
            "DefaultOrder": 0,
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
            "MaxHP1": 2000,
            "MaxHP100": 20000,
            "AttackPower1": 300,
            "AttackPower100": 3000,
            "DefensePower1": 20,
            "DefensePower100": 200,
            "HealPower1": 1000,
            "HealPower100": 3000,
            "AccuracyPoint": 900,
            "DodgePoint": 200,
            "CriticalPoint": 200,
            "CriticalDamageRate": 20000,
            "StabilityPoint": 2000,
            "Range": 750,
            "AmmoCount": 5,
            "AmmoCost": 1,
            "RegenCost": 700,
            "FavorStatType": ["AttackPower", "MaxHP"],
            "FavorStatValue": [
                [3, 0],  # Index 0: Levels 1-4
                [5, 0],  # Index 1: Levels 5-9
                [7, 43],  # Index 2: Levels 10-14
                [9, 51],  # Index 3: Levels 15-19
                [2, 8],  # Index 4: Levels 20-29
                [3, 13],  # Index 5: Levels 30-39
                [5, 21],  # Index 6: Levels 40-49
            ],
            "Equipment": ["Hat", "Hairpin", "Watch"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10086",
        }

        self.hina = Student.model_validate(self.hina_data)

    def test_bond_level_1_is_zero(self):
        """Bond level 1 should give 0 bonus."""
        bond_stats = self.hina.get_bond_stats(1)
        self.assertEqual(bond_stats["AttackPower"], 0)
        self.assertEqual(bond_stats["MaxHP"], 0)

    def test_bond_level_5_calculation(self):
        """Test bond stats at level 5."""
        # Levels 1-4: 3 each = 3*4 = 12 attack
        bond_stats = self.hina.get_bond_stats(5)
        self.assertEqual(bond_stats["AttackPower"], 12, "Should be 3*4 = 12")
        self.assertEqual(bond_stats["MaxHP"], 0, "Should be 0")

    def test_bond_level_10_calculation(self):
        """Test bond stats at level 10."""
        # Levels 1-4: 3 each = 12
        # Levels 5-9: 5 each = 25
        # Total: 37
        bond_stats = self.hina.get_bond_stats(10)
        self.assertEqual(bond_stats["AttackPower"], 37, "Should be 12 + 25 = 37")
        self.assertEqual(bond_stats["MaxHP"], 0, "Should be 0")

    def test_bond_level_15_calculation(self):
        """Test bond stats at level 15."""
        # Levels 1-4: 3 each = 12
        # Levels 5-9: 5 each = 25
        # Levels 10-14: 7 each = 35
        # Total attack: 72
        # Total HP: 0 + 0 + 43*5 = 215
        bond_stats = self.hina.get_bond_stats(15)
        self.assertEqual(bond_stats["AttackPower"], 72, "Should be 12 + 25 + 35 = 72")
        self.assertEqual(bond_stats["MaxHP"], 215, "Should be 0 + 0 + 43*5 = 215")

    def test_bond_level_20_calculation(self):
        """Test bond stats at level 20."""
        # Levels 1-4: 3 each = 12
        # Levels 5-9: 5 each = 25
        # Levels 10-14: 7 each = 35
        # Levels 15-19: 9 each = 45
        # Total attack: 117
        # Total HP: 0 + 0 + 215 + 51*5 = 470
        bond_stats = self.hina.get_bond_stats(20)
        self.assertEqual(bond_stats["AttackPower"], 117, "Should be 12 + 25 + 35 + 45 = 117")
        self.assertEqual(bond_stats["MaxHP"], 470, "Should be 0 + 0 + 215 + 255 = 470")

    def test_bond_level_30_calculation(self):
        """Test bond stats at level 30."""
        # Up to level 20: attack 117, HP 470
        # Levels 20-29: 2 attack, 8 HP each = 20, 80
        # Total: attack 137, HP 550
        bond_stats = self.hina.get_bond_stats(30)
        self.assertEqual(bond_stats["AttackPower"], 137, "Should be 117 + 20 = 137")
        self.assertEqual(bond_stats["MaxHP"], 550, "Should be 470 + 80 = 550")

    def test_bond_level_40_calculation(self):
        """Test bond stats at level 40."""
        # Up to level 30: attack 137, HP 550
        # Levels 30-39: 3 attack, 13 HP each = 30, 130
        # Total: attack 167, HP 680
        bond_stats = self.hina.get_bond_stats(40)
        self.assertEqual(bond_stats["AttackPower"], 167, "Should be 137 + 30 = 167")
        self.assertEqual(bond_stats["MaxHP"], 680, "Should be 550 + 130 = 680")

    def test_bond_level_50_calculation(self):
        """Test bond stats at max level 50."""
        # Up to level 40: attack 167, HP 680
        # Levels 40-49: 5 attack, 21 HP each = 50, 210
        # Total: attack 217, HP 890
        bond_stats = self.hina.get_bond_stats(50)
        self.assertEqual(bond_stats["AttackPower"], 217, "Should be 167 + 50 = 217")
        self.assertEqual(bond_stats["MaxHP"], 890, "Should be 680 + 210 = 890")

    def test_bond_level_boundary_negative(self):
        """Test bond level 0 and negative values."""
        bond_0 = self.hina.get_bond_stats(0)
        bond_neg = self.hina.get_bond_stats(-5)

        # Both should be treated as level 1
        self.assertEqual(bond_0["AttackPower"], 0)
        self.assertEqual(bond_neg["AttackPower"], 0)

    def test_bond_level_boundary_above_50(self):
        """Test bond levels above 50."""
        bond_50 = self.hina.get_bond_stats(50)
        bond_100 = self.hina.get_bond_stats(100)

        # Should be capped at 50
        self.assertEqual(bond_50, bond_100)

    def test_total_stats_calculation(self):
        """Test get_total_stats combines base and bond stats."""
        # Level 1, bond 1
        stats_1_1 = self.hina.get_total_stats(level=1, bond_level=1)
        self.assertEqual(stats_1_1["AttackPower"], 300)  # Base only
        self.assertEqual(stats_1_1["MaxHP"], 2000)  # Base only

        # Level 1, bond 50
        stats_1_50 = self.hina.get_total_stats(level=1, bond_level=50)
        self.assertEqual(stats_1_50["AttackPower"], 517)  # 300 + 217
        self.assertEqual(stats_1_50["MaxHP"], 2890)  # 2000 + 890

    def test_total_stats_with_level_scaling(self):
        """Test that character level affects base stats."""
        # Level 1
        stats_lv1 = self.hina.get_total_stats(level=1, bond_level=1)

        # Level 100
        stats_lv100 = self.hina.get_total_stats(level=100, bond_level=1)

        # Base stats should scale
        self.assertEqual(stats_lv1["AttackPower"], 300)
        self.assertEqual(stats_lv100["AttackPower"], 3000)

    def test_favor_stat_value_structure(self):
        """Verify FavorStatValue has correct structure."""
        self.assertEqual(len(self.hina.FavorStatValue), 7, "Should have 7 entries")
        for i, entry in enumerate(self.hina.FavorStatValue):
            self.assertEqual(len(entry), 2, f"Entry {i} should have 2 values")

    def test_favor_stat_type_structure(self):
        """Verify FavorStatType has correct structure."""
        self.assertEqual(len(self.hina.FavorStatType), 2, "Should have 2 stat types")
        self.assertIn("AttackPower", self.hina.FavorStatType)
        self.assertIn("MaxHP", self.hina.FavorStatType)


if __name__ == "__main__":
    unittest.main()
