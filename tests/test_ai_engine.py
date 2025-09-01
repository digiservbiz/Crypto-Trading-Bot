import unittest
from scripts.inference.ai_engine import AIEngine

class TestAIEngine(unittest.TestCase):
    def test_singleton(self):
        """Test that the AIEngine is a singleton."""
        engine1 = AIEngine()
        engine2 = AIEngine()
        self.assertIs(engine1, engine2)

if __name__ == '__main__':
    unittest.main()