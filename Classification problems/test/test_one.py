import __init__
from src.src_test import src_class
import unittest


class TestSrcClass(unittest.TestCase):

    def setUp(self):
        self.src_obj = src_class()

    def test_src_method_1(self):
        # Test with positive integers
        result = self.src_obj.src_method_1(3, 5)
        self.assertEqual(result, 8)

        # Test with negative integers
        result = self.src_obj.src_method_1(-3, -5)
        self.assertEqual(result, -8)

        # Test with zero
        result = self.src_obj.src_method_1(0, 0)
        self.assertEqual(result, 0)
