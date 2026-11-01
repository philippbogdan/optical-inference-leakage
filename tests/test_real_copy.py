"""Check noise statistics, estimator bias and external-only weight exports."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optical_inference_leakage import real_copy_data as real


class RealCopyChecks(unittest.TestCase):
    def test_nested_deliveries_and_fixed_crop_shrinkage(self):
        crop = np.full((256, 256), 2.0)
        pack, rows = real.simulate(crop, s=1, v=1, counts=(1, 2, 4), seed=23)
        noise = pack['average'] - crop
        increment = 2 * noise[1] - noise[0]
        self.assertLess(abs(np.corrcoef(noise[0].ravel(), increment.ravel())[0, 1]), 0.02)
        self.assertAlmostEqual(float(np.mean(noise[0] * noise[1])), 0.5, delta=0.02)
        for i, k in enumerate(pack['K']):
            self.assertAlmostEqual(float(np.mean(noise[i] ** 2)), 1 / k, delta=0.02 / k)
        self.assertAlmostEqual(float(np.mean(pack['estimate'][0])), 1.0, delta=0.01)
        self.assertAlmostEqual(rows[0]['estimate_rmse'] ** 2, 1.25, delta=0.02)
        self.assertAlmostEqual(rows[0]['expected_estimate_rms_pct_matrix'], 100 * np.sqrt(1.25))
        self.assertNotAlmostEqual(rows[0]['expected_estimate_rms_pct_matrix'],
                                  rows[0]['gaussian_prior_rms_pct_matrix'])

    def test_digest_accepts_known_bytes_and_rejects_modified_bytes(self):
        expected = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input'
            path.write_bytes(b'abc')
            self.assertEqual(real.verify_sha256(path, expected), expected)
            path.write_bytes(b'abd')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                real.verify_sha256(path, expected)

    def test_repository_and_redirected_output_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside the repository'):
            real.output_directory(real.ROOT / 'derived-weights')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(real.output_directory(root), root.resolve())
            (root / 'redirect').symlink_to(real.ROOT, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'outside the repository'):
                real.output_directory(root / 'redirect' / 'derived-weights')
            (root / real.FILES[0]).symlink_to(real.ROOT / 'README.md')
            with self.assertRaisesRegex(ValueError, 'Unsafe output target'):
                real.output_directory(root)

    def test_wrong_run_and_invalid_copy_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'strict-quality'):
            real.parameters(dict(model=real.MODEL, level=0.02))
        for counts in ((0,), (2, 1), (1, 1), (1.5,)):
            with self.assertRaisesRegex(ValueError, 'Text counts'):
                real.simulate(np.ones((2, 2)), 1, 1, counts=counts)


if __name__ == '__main__':
    unittest.main()
