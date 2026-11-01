"""Independent checks of counting, optical conventions and mismatch detection."""
import copy
import csv
import unittest
from unittest.mock import patch
import numpy as np
import reproduce
from optical_inference_leakage import figure_data, hero_data, leakage, recompute as rc


class EvidenceChecks(unittest.TestCase):
    def test_copy_error_combines_tied_observations_by_precision(self):
        common = dict(model='example', level=0.005, rows=2, columns=3, N_view=6,
                      layer='', mean_square=2, shared_weight_key='embed/head')
        rows = hero_data.copy_matrices([
            dict(common, key='embed', matrix_type='embed', view_snr=2),
            dict(common, key='head', matrix_type='head', view_snr=1)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['N'], 6)
        self.assertAlmostEqual(rows[0]['rms_pct_K1'], 100 / np.sqrt(3))
        self.assertAlmostEqual(rows[0]['rms_pct_K489'], 100 / np.sqrt(1467))

    def test_motion_counts_endpoints_and_independent_interior_capacity(self):
        runs = reproduce.load_runs()
        pack = hero_data.build(runs)
        np.testing.assert_array_equal(pack['K'], np.arange(1, 490))
        self.assertEqual(pack['rms_pct'].shape, (489, 2703))
        self.assertEqual(len(pack['timeline']), 537)
        self.assertEqual({r['K'] for r in pack['timeline']}, set(range(1, 490)))
        for i, run in enumerate(runs):
            self.assertAlmostEqual(pack['leakage_bits_per_weight'][0, i], run['test']['bound']['1'])
            self.assertAlmostEqual(pack['leakage_bits_per_weight'][-1, i], run['test']['bound']['1e6'])
            self.assertAlmostEqual(pack['leakage_bits_per_weight'][36, i], rc.capacity(run, 37))
            selected = [r for r in pack['matrices'] if r['series_index'] == i]
            self.assertEqual(sum(r['N'] for r in selected), run['Ntot'])
        np.testing.assert_allclose(pack['rms_pct'][0] / pack['rms_pct'][-1], np.sqrt(489))

    def test_motion_rejects_missing_runs_and_changed_stored_bound(self):
        runs = reproduce.load_runs()
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            hero_data.build(runs[:-1])
        runs[0]['test']['bound']['1e6'] += 0.1
        with self.assertRaisesRegex(ValueError, 'hero/1e6'):
            hero_data.build(runs)

    def test_tied_views_add_precision_without_double_counting(self):
        run = {'keys': ['embed', 'head'], 'tied': True, 'N': [6, 6],
               'Ntot': 6, 's': [2, 2], 'v': [1, 2]}
        self.assertAlmostEqual(rc.capacity(run, 1), 1.0)
        self.assertAlmostEqual(rc.capacity(run, 3), 0.5 * np.log2(10))

    def test_finite_optics_respects_modulation_bound(self):
        s, v, alpha_squared = 2.0, 0.1, 50.0
        honest_snr = s / (v + 1 / (4 * alpha_squared))
        for copies in (1, 489):
            finite = leakage.broadcast_leak(honest_snr, copies, nbar=alpha_squared * s)
            bright = leakage.broadcast_leak(s / v, copies)
            self.assertLessEqual(finite, bright)

    def test_changed_reported_value_is_rejected(self):
        actual = {'model': 'Qwen3-0.6B', 'L1': 6.486687942379634}
        altered = {'model': 'Qwen3-0.6B', 'L1': '6.5'}
        with self.assertRaisesRegex(ValueError, 'L1'):
            reproduce.compare_rows([actual], [altered], 'summary')

    def test_changed_quality_draws_are_rejected(self):
        run = copy.deepcopy(reproduce.load_runs()[0])
        run['test']['draws'][0] *= 1.1
        with self.assertRaisesRegex(ValueError, 'unfavourable'):
            rc.quality(run)

    def test_matrix_atlas_keeps_tied_views_and_unique_identity(self):
        run = reproduce.load_runs()[0]
        rows = figure_data.matrix_rows(run)
        views = {r['key']: r for r in rows}
        self.assertEqual(len(rows), len(run['keys']))
        self.assertEqual(views['embed']['shared_weight_key'], views['head']['shared_weight_key'])
        self.assertNotEqual(views['embed']['noise_rms_pct'], views['head']['noise_rms_pct'])
        unique = {r['shared_weight_key']: r['N_view'] for r in rows}
        self.assertEqual(sum(unique.values()), run['Ntot'])
        self.assertGreater(sum(r['N_view'] for r in rows), run['Ntot'])
        self.assertEqual(views['embed']['layer'], '')

    def test_matrix_atlas_rejects_a_missing_layer_view(self):
        run = copy.deepcopy(reproduce.load_runs()[0])
        for key in ('keys', 's', 'v', 'N'):
            del run[key][1]
        with self.assertRaisesRegex(ValueError, 'matrix views'):
            figure_data.matrix_rows(run)

    def test_changed_long_text_claim_is_rejected(self):
        runs = reproduce.load_runs()
        tables = reproduce.table_rows(runs, detailed=False)
        with (reproduce.ROOT / 'results/attacks.csv').open() as stream:
            tables['attacks'] = [dict(r, fraction=float(r['fraction'])) for r in csv.DictReader(stream)]
        original = reproduce.Path.read_text
        def altered(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            return text.replace('1.043 to 1.100', '1.000 to 1.001') if path == reproduce.ROOT / 'README.md' else text
        with patch.object(reproduce.Path, 'read_text', altered):
            with self.assertRaisesRegex(ValueError, 'README.md'):
                reproduce.check_prose(tables, rc.scaling(runs))

    def test_changed_hero_noise_value_is_rejected(self):
        runs = reproduce.load_runs()
        tables = reproduce.table_rows(runs, detailed=False)
        with (reproduce.ROOT / 'results/attacks.csv').open() as stream:
            tables['attacks'] = [dict(r, fraction=float(r['fraction'])) for r in csv.DictReader(stream)]
        original = reproduce.Path.read_text
        def altered(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path == reproduce.ROOT / 'REPRODUCE.md':
                text = text.replace('0.100, 1.000, 3.000, 5.895', '0.100, 1.000, 3.000, 6.500')
            return text
        with patch.object(reproduce.Path, 'read_text', altered):
            with self.assertRaisesRegex(ValueError, 'hero bit levels or noise'):
                reproduce.check_prose(tables, rc.scaling(runs))


if __name__ == '__main__':
    unittest.main()
