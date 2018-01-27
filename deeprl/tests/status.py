import unittest
from deeprl.agents.AbstractAgent import Status


class TestStatus(unittest.TestCase):
    def test_dummy(self):
        s = Status('a', 1, print)

        for v in s.values():
            self.assertIsNone(v)

        s.a = 2
        self.assertEqual(s.a, 2)
        self.assertEqual(s['a'], 2)

        # Setting values for non-initial keys should not be allowed
        with self.assertRaises(KeyError):
            s.b = 1

        with self.assertRaises(KeyError):
            s['b'] = 1

        pass
