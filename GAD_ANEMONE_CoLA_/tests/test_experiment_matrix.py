import json
import unittest
from pathlib import Path


class TestThreeBaselineMatrix(unittest.TestCase):
    def test_exact_matrix_and_greet_gate(self):
        path = Path(__file__).absolute().parents[1] / 'configs' / 'experiment.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(set(config['models']), {'anemone', 'cola', 'greet'})
        self.assertEqual(set(config['datasets']), {'amazon', 'yelp', 'elliptic', 'tfinance'})
        self.assertEqual(config['seeds'], list(range(10)))
        self.assertEqual(len(config['models']) * len(config['datasets']) * len(config['seeds']), 120)
        self.assertEqual(config['model_config']['greet']['status'], 'BLOCKED_SCORE_PROTOCOL')
        self.assertEqual(config['model_config']['greet']['score_method'], 'UNDECIDED')
        self.assertEqual(config['model_config']['greet']['label_use'], 'UNDECIDED')


if __name__ == '__main__':
    unittest.main()
