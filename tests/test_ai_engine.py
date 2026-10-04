import unittest

from scripts.inference.ai_engine import AIEngine


class TestAIEngine(unittest.TestCase):
    def test_initializes_with_config(self):
        config = {
            "inference": {"models_dir": "models"},
            "models": {
                "model_type": "lstm",
                "model_selection": {
                    "enabled": True,
                    "volatility_threshold": 0.02,
                },
            },
        }
        engine = AIEngine(config)
        self.assertEqual(engine.models_dir, "models")
        self.assertEqual(engine.sequential_models, {})


if __name__ == "__main__":
    unittest.main()
