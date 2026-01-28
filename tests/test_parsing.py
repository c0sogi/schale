import logging
import unittest

from schale import cache_control as cc

logging.basicConfig(level=logging.DEBUG)


class TestParsing(unittest.TestCase):
    def test_equipment(self):
        cc.cache_collection.equipments

    def test_items(self):
        cc.cache_collection.items

    def test_stages(self):
        cc.cache_collection.stages

    def test_groups(self):
        cc.cache_collection.groups

    def test_furniture(self):
        cc.cache_collection.furnitures

    def test_students(self):
        students = cc.cache_collection.students
        self.assertGreater(len(students), 0, "Should have at least one student")

    def test_refresh_all(self):
        cc.force_refresh = True
        cc.cache_collection.refresh_all()
