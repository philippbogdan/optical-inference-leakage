"""Check Gaussian channel units, sampled noise and external-only weight exports."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optical_inference_leakage import real_copy_data as real


class RealCopyChecks(unittest.TestCase):
    def test_bits_invert_channel_and_final_variance_is_stored(self):
        levels, variances = real.noise_schedule(s=12, v=0.01)
        self.assertAlmostEqual(variances[1], 4.0)
        self.assertAlmostEqual(variances[2], 12 / 63)
        self.assertEqual(variances[-1], 0.01)
        np.testing.assert_allclose(0.5 * np.log2(1 + 12 / variances), levels, rtol=1e-14)
        self.assertAlmostEqual(levels[-1], 0.5 * np.log2(1201))

    def test_common_noise_field_and_fixed_crop_normalisations(self):
        crop = np.full((128, 128), 2.0)
        pack, rows = real.simulate(crop, s=1, v=0.0001, seed=23)
        repeated, _ = real.simulate(crop, s=1, v=0.0001, seed=23)
        np.testing.assert_array_equal(pack['copy'], repeated['copy'])
        self.assertEqual(pack['copy'].shape, (4, 128, 128))
        for i, row in enumerate(rows):
            noise = pack['copy'][i] - crop
            np.testing.assert_allclose(noise / row['noise_std'], pack['standard_normal'], atol=1e-13)
            self.assertAlmostEqual(float(np.mean(noise ** 2)) / row['noise_variance'], 1, delta=0.04)
            self.assertEqual(row['noise_rms_pct_matrix'], 2 * row['noise_rms_pct_crop'])
            self.assertAlmostEqual(row['measured_rmse'], float(np.sqrt(np.mean(noise ** 2))))
        self.assertEqual(rows[-1]['noise_variance'], 0.0001)
        self.assertEqual([r['p0_one_text'] for r in rows], [False, False, False, True])
        # A constant crop retains its mean; no low-information shrinkage is applied.
        self.assertAlmostEqual(float(np.mean(pack['copy'][0])), 2.0, delta=0.06)

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

    def test_wrong_run_and_invalid_channel_levels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'strict-quality'):
            real.parameters(dict(model=real.MODEL, level=0.02))
        for bits in ((), (0,), (-1,), (np.inf,), (np.nan,), (2, 1), (1, 1), (8,)):
            with self.assertRaisesRegex(ValueError, 'Bit levels'):
                real.noise_schedule(1, 0.0001, bits=bits)
        for s, v in ((0, 1), (1, 0), (np.inf, 1), (1, -1)):
            with self.assertRaisesRegex(ValueError, 'positive'):
                real.noise_schedule(s, v)

    def test_multiply_linked_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'original'
            source.write_bytes(b'preserved')
            (root / real.FILES[0]).hardlink_to(source)
            with self.assertRaisesRegex(ValueError, 'hard links'):
                real.output_directory(root)
            self.assertEqual(source.read_bytes(), b'preserved')


if __name__ == '__main__':
    unittest.main()
