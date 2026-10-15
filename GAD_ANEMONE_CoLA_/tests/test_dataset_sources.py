import tempfile
import unittest
from pathlib import Path
import numpy as np

from gad_handoff.data import dataset_path, file_sha256, _elliptic_masks


class TestPublishedSources(unittest.TestCase):
    def test_exact_paths_and_hetero_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'amazon').write_bytes(b'binary')
            elliptic = root / 'elliptic_bitcoin'
            elliptic.mkdir()
            for filename in ('elliptic_txs_classes.csv', 'elliptic_txs_features.csv',
                             'elliptic_txs_edgelist.csv'):
                (elliptic / filename).write_text('example', encoding='utf-8')
            self.assertEqual(dataset_path(root, 'amazon'), root / 'amazon')
            self.assertEqual(dataset_path(root, 'elliptic'), elliptic)
            self.assertEqual(len(file_sha256(elliptic)), 64)
            with self.assertRaises(FileNotFoundError):
                dataset_path(root, 'yelp')
            hetero = root / 'hetero'
            hetero.mkdir()
            with self.assertRaises(ValueError):
                dataset_path(hetero, 'amazon')

    def test_elliptic_trials_partition_known_labels_only(self):
        labels = np.array([0] * 100 + [1] * 30 + [-1] * 20)
        train, val, test = _elliptic_masks(labels)
        self.assertEqual(train.shape, (150, 10))
        self.assertFalse(((train & val) | (train & test) | (val & test)).any())
        self.assertTrue(np.array_equal(train | val | test,
                                       np.broadcast_to((labels >= 0)[:, None], (150, 10))))
        for trial in range(10):
            self.assertEqual(set(labels[val[:, trial]]), {0, 1})
            self.assertEqual(set(labels[test[:, trial]]), {0, 1})


if __name__ == '__main__':
    unittest.main()
