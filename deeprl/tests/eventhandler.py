import unittest
from EventHandler import EventHandler

class Test(unittest.TestCase):
    def testAddEvent(self):
        handler = EventHandler()
        event = lambda x: print(x)

        self.assertNotIn(event, handler)
        handler += event
        self.assertIn(event, handler)

    def testRemoveEvent(self):
        handler = EventHandler()
        event = lambda x: print(x)

        handler += event
        self.assertIn(event, handler)
        handler -= event
        self.assertNotIn(event, handler)

    def testCall(self):
        handler = EventHandler()
        handler += lambda x: x/0 # Will raise ZeroDivisionError

        with self.assertRaises(ZeroDivisionError):
            handler(10)

    def testMultipleEvents(self):
        handler = EventHandler()
        event1 = lambda x: x
        event2 = lambda x: x

        handler += event1
        self.assertIn(event1, handler)
        self.assertNotIn(event2, handler)

        handler += event2
        self.assertIn(event1, handler)
        self.assertIn(event2, handler)

        handler -= event1
        self.assertNotIn(event1, handler)
        self.assertIn(event2, handler)

    def testDuplicateEvents(self):
        handler = EventHandler()
        def event():
            pass

        handler += event
        handler += event
        self.assertIn(event, handler)
        self.assertEqual(len(handler), 1)

        handler -= event
        self.assertNotIn(event, handler)
