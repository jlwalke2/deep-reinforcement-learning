import unittest
from deeprl.utils.metrics import callback_return

class TestMetrics(unittest.TestCase):

    def test_returns_none(self):
        """All metrics should equal None if no return value given."""
        metrics = ('input', 'output')

        @callback_return(*metrics)
        def returns_none():
            pass

        r = returns_none()
        for metric in metrics:
            self.assertIn(metric, r)
            self.assertIsNone(r[metric])


    def test_returns_dict(self):
        """All metrics should match the corresponding returned value."""
        metrics = ('input', 'output')

        @callback_return(*metrics)
        def returns_dict():
            return {'output': 1, 'input': 2, 'extra': 3}

        r = returns_dict()
        self.assertEqual(len(metrics), len(r.keys()), 'Extra return values should be dropped.')
        self.assertEqual(2, r['input'])
        self.assertEqual(1, r['output'])
        self.assertNotIn('extra', r)


    def test_returns_list(self):
        """All metrics should match the corresponding return value, as indicated by the metric order."""
        metrics = ('input', 'output')

        @callback_return(*metrics)
        def returns_list():
            return [2, 1, 3]

        r = returns_list()
        self.assertEqual(len(metrics), len(r.keys()), 'Extra return values should be dropped.')
        self.assertEqual(2, r['input'])
        self.assertEqual(1, r['output'])
        self.assertNotIn('extra', r)


    def test_returns_tuple(self):
        """All metrics should match the corresponding return value, as indicated by the metric order."""
        metrics = ('input', 'output')

        @callback_return(*metrics)
        def returns_tuple():
            return 2, 1, 3

        r = returns_tuple()
        self.assertEqual(len(metrics), len(r.keys()), 'Extra return values should be dropped.')
        self.assertEqual(2, r['input'])
        self.assertEqual(1, r['output'])
        self.assertNotIn('extra', r)

    def test_returns_string(self):
        """Only return types of dict, list, and tuple are mapped to metric names."""
        metrics = ('input', 'output')

        @callback_return(*metrics)
        def returns_string():
            return 'something stupid'

        r = returns_string()
        for metric in metrics:
            self.assertIn(metric, r)
            self.assertIsNone(r[metric])