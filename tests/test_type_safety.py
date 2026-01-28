"""
Tests for type safety features.

These tests verify that the type-safe utility functions work correctly.
"""

import unittest

from schale.schema.student import Student


class TestTypeSafety(unittest.TestCase):
    """Test type-safe student query utilities."""

    def setUp(self):
        """Create test students."""
        # Hina variants for testing
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
            "FavorStatValue": [[3, 38], [5, 63], [7, 88], [9, 106], [2, 18], [3, 29], [5, 47]],
            "FavorAlts": [10022, 10086],
            "Equipment": ["Hat", "Gloves", "Shoes"],
            "Skills": [],
            "WeaponImg": "weapon_icon_10004",
        }

    def test_student_pathname_type(self):
        """Test that PathName field accepts StudentPathName literal."""
        student = Student.model_validate(self.hina_data)

        # PathName should be a StudentPathName literal
        self.assertEqual(student.PathName, "Hina")
        self.assertEqual(student.DevName, "Hina")

        # Type is preserved
        self.assertIsInstance(student.PathName, str)
        self.assertIsInstance(student.DevName, str)

    def test_student_creation_with_valid_pathnames(self):
        """Test creating students with valid PathNames."""
        valid_pathnames = [
            "Hina",
            "Hina_Swimsuit",
            "Hina_Dress",
            "Aru",
            "Aru_NewYear",
            "Shiroko_Terror",
        ]

        for pathname in valid_pathnames:
            data = self.hina_data.copy()
            data["PathName"] = pathname
            data["DevName"] = pathname

            # Should not raise validation error
            student = Student.model_validate(data)
            self.assertEqual(student.PathName, pathname)

    def test_student_typed_fields(self):
        """Test that typed fields work correctly."""
        student = Student.model_validate(self.hina_data)

        # School should be a School literal
        self.assertEqual(student.School, "Gehenna")

        # TacticRole should be a TacticRole literal
        self.assertEqual(student.TacticRole, "Tanker")

        # WeaponType should be a WeaponType literal
        self.assertEqual(student.WeaponType, "SG")

        # Position should be a Position literal
        self.assertEqual(student.Position, "Middle")

    def test_favor_stat_type_literals(self):
        """Test that FavorStatType uses StatType literals."""
        student = Student.model_validate(self.hina_data)

        # FavorStatType should contain valid StatType values
        self.assertEqual(len(student.FavorStatType), 2)
        self.assertIn(student.FavorStatType[0], ["AttackPower", "MaxHP", "DefensePower", "HealPower"])
        self.assertIn(student.FavorStatType[1], ["AttackPower", "MaxHP", "DefensePower", "HealPower"])


if __name__ == "__main__":
    unittest.main()
