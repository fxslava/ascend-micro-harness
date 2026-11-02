"""Independent numerical and retrieval oracles for the research runner."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import spearmanr, ttest_rel
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tq_cdf_vs_lloydmax_ablation as tq


class CodecTests(unittest.TestCase):
    def test_rotation_is_orthogonal_and_self_inverse(self):
        rng = np.random.default_rng(1)
        for dimension in (64, 128):
            x = rng.standard_normal((32, dimension)).astype(np.float32)
            rotation = tq.Rotation(dimension, 91)
            y = rotation.apply(x)
            np.testing.assert_allclose(rotation.apply(y), x, atol=8e-7)
            np.testing.assert_allclose(x.astype(np.float64) @ x.T,
                                       y.astype(np.float64) @ y.T, atol=1e-5, rtol=5e-7)

    def test_all_codes_are_packed_at_exact_bit_budget(self):
        for bits in tq.BIT_WIDTHS:
            indices = np.tile(np.arange(128, dtype=np.uint8) % (1 << bits), (5, 1))
            packed = tq.pack_indices(indices, bits)
            self.assertEqual(packed.shape, (5, 128 * bits // 8))
            np.testing.assert_array_equal(tq.unpack_indices(packed, 128, bits), indices)
            # Independent scalar little-endian bitstream oracle.
            scalar = sum(int(code) << (i * bits) for i, code in enumerate(indices[0]))
            self.assertEqual(packed[0].tobytes(), scalar.to_bytes(128 * bits // 8, "little"))

    def test_gaussian_grid_and_logistic_inverse(self):
        for bits in tq.BIT_WIDTHS:
            midpoints = (np.arange(1 << bits, dtype=np.float32) + 0.5) / (1 << bits)
            z = tq.inverse_sigmoid_polynomial(midpoints)
            w = np.log(midpoints) - np.log1p(-midpoints)
            np.testing.assert_allclose(z * (tq.C1 + tq.C3 * z * z), w, atol=2e-6)
            np.testing.assert_allclose(ndtr(ndtri(midpoints)), midpoints, atol=1e-7)
            self.assertTrue(np.all(np.diff(z) > 0))
            self.assertLess(np.max(np.abs(z - ndtri(midpoints))), 0.002)

    def test_lloyd_codebook_stationarity_and_gaussian_mse(self):
        # Fine deterministic normal quantiles avoid an empirical sample fit.
        x = ndtri((np.arange(200000) + 0.5) / 200000)
        for bits in tq.BIT_WIDTHS:
            centres, thresholds = tq.lloyd_max(bits)
            np.testing.assert_allclose(centres, -centres[::-1], atol=1e-7)
            edges = np.r_[-np.inf, thresholds.astype(np.float64), np.inf]
            density = np.exp(-edges ** 2 / 2) / np.sqrt(2 * np.pi)
            conditional_means = (density[:-1] - density[1:]) / np.diff(ndtr(edges))
            # Match the suite's finite 256-iteration baseline, including its tolerance.
            np.testing.assert_allclose(centres, conditional_means, atol=3e-7 if bits == 5 else 3e-5)
            lloyd = centres[np.searchsorted(thresholds, x)]
            gaussian = ndtri((np.clip(np.floor(ndtr(x) * (1 << bits)), 0, (1 << bits) - 1) + 0.5) / (1 << bits))
            self.assertLess(np.mean((x - lloyd) ** 2), np.mean((x - gaussian) ** 2))

    def test_fp16_scales_zero_vectors_and_payload(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((9, 128)).astype(np.float32)
        x[0] = 0
        for regime in tq.REGIMES:
            for bits in tq.BIT_WIDTHS:
                for group_size in (0, 32):
                    codec = tq.Quantizer(regime, bits, group_size)
                    packed = codec.encode(x)
                    decoded = codec.decode(packed)
                    self.assertEqual(packed.scales.dtype, np.float16)
                    self.assertEqual(packed.codes.nbytes, len(x) * 128 * bits // 8)
                    self.assertEqual(packed.scales.nbytes, len(x) * (128 // (group_size or 128)) * 2)
                    np.testing.assert_array_equal(decoded[0], np.zeros(128))
                    self.assertTrue(np.all(np.isfinite(decoded)))
                    self.assertGreater(np.mean(np.sum(tq.unit_rows(x[1:]) * tq.unit_rows(decoded[1:]), axis=1)), 0.9)
                    with self.assertRaises(ValueError):
                        codec.encode(np.full((1, 128), np.inf))

    def test_uniform_signed_codes_zero_and_grid_boundaries(self):
        for bits in tq.BIT_WIDTHS:
            half = 1 << (bits - 1)
            for regime in tq.REGIMES[3:]:
                codec = tq.Quantizer(regime, bits)
                codes = np.tile(np.arange(1 << bits, dtype=np.uint8), 128 // (1 << bits))[None]
                packed = tq.Packed(tq.pack_indices(codes, bits), np.ones((1, 1), dtype=np.float16), 128)
                offset = 0.5 if regime == tq.REGIMES[3] else 0.0
                np.testing.assert_array_equal(codec.decode(packed), codes.astype(np.int16) - half + offset)
                # Exact scale=1 gives an independent scalar rounding oracle at ties,
                # zero and both extrema. Signed extremes must not wrap in uint8.
                limit = half - (0.5 if regime == tq.REGIMES[3] else 1)
                values = [-limit, -1.5, -0.5, 0, 0.5, 1.5, limit]
                x = np.resize(np.array(values, dtype=np.float32), 128)[None]
                encoded = codec.encode(x)
                self.assertEqual(float(encoded.scales[0, 0]), 1.0)
                if regime == tq.REGIMES[3]:
                    expected = [min((1 << bits) - 1, max(0, int(np.floor(v + half)))) for v in x[0]]
                else:
                    expected = [min(half - 1, max(-half, round(float(v)))) + half for v in x[0]]
                np.testing.assert_array_equal(tq.unpack_indices(encoded.codes, 128, bits)[0], expected)
                if regime == tq.REGIMES[4]:
                    self.assertEqual(float(codec.decode(encoded)[0, 3]), 0.0)

    def test_uniform_fp16_group_step_uses_absmax_not_rms(self):
        x = np.arange(1, 129, dtype=np.float32)[None]
        for bits in tq.BIT_WIDTHS:
            for regime in tq.REGIMES[3:]:
                codec = tq.Quantizer(regime, bits, 32)
                denominator = (1 << (bits - 1)) - (0.5 if regime == tq.REGIMES[3] else 1)
                expected = (np.array([[32, 64, 96, 128]], dtype=np.float64) / denominator).astype(np.float16)
                np.testing.assert_array_equal(codec.encode(x).scales, expected)

    def test_chunk_stable_gaussian_generator(self):
        full = tq.gaussian_rows(np.arange(1024), 71)
        pieces = np.concatenate([tq.gaussian_rows(np.arange(i, min(i + 77, 1024)), 71)
                                 for i in range(0, 1024, 77)])
        np.testing.assert_array_equal(full, pieces)
        self.assertLess(abs(full.mean()), 0.01)
        self.assertLess(abs(full.std() - 1), 0.01)

    def test_reference_suite_compatibility_when_available(self):
        reference = Path("C:/ascend-moe-engine/tests/research")
        if not reference.is_dir():
            self.skipTest("optional original reference checkout unavailable")
        sys.path.insert(0, str(reference))
        from turbo_lattice_vq import LloydMaxQuantizer
        from extreme_niah_benchmark import _pack_indices_batch, _unpack_indices_batch
        from niah_attention_regression import spearman_rank_correlation
        from dsv4_mla_niah_benchmark import _rope_tables, _apply_rope_half
        for bits in (3, 4):
            np.testing.assert_allclose(tq.lloyd_max(bits)[0],
                                       LloydMaxQuantizer.standard_normal(1 << bits).centres, atol=2e-7)
            indices = np.tile(np.arange(128, dtype=np.uint8) % (1 << bits), (3, 1))
            np.testing.assert_array_equal(tq.pack_indices(indices, bits), _pack_indices_batch(indices, bits))
            np.testing.assert_array_equal(tq.unpack_indices(tq.pack_indices(indices, bits), 128, bits),
                                          _unpack_indices_batch(_pack_indices_batch(indices, bits), 128, bits))
        a, b = np.array([1, 2, 2, 4])[:, None], np.array([3, 2, 2, 1])[:, None]
        self.assertAlmostEqual(tq.spearman_columns(a, b)[0], spearman_rank_correlation(a[:, 0].tolist(), b[:, 0].tolist()))
        positions = np.array([0, 1, 8192, 131071])
        carriers = np.random.default_rng(1).standard_normal((4, 64)).astype(np.float32)
        cos, sin = _rope_tables(positions, rope_dim=64, theta=160000)
        rotated = _apply_rope_half(carriers, cos, sin)
        angles = positions[:, None] * 160000.0 ** (-2 * np.arange(32)[None] / 64)
        expected = np.c_[np.cos(angles) * carriers[:, :32] - np.sin(angles) * carriers[:, 32:],
                         np.sin(angles) * carriers[:, :32] + np.cos(angles) * carriers[:, 32:]]
        np.testing.assert_allclose(rotated, expected, atol=3e-7)


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiter = threadpool_limits(limits=1)
        cls.rows, cls.needles = tq.run_case("niah", 1024, 8, 11, chunk_tokens=256, sample_slots=128)

    @classmethod
    def tearDownClass(cls):
        cls.limiter.restore_original_limits()

    def test_chunk_size_preserves_metrics(self):
        with threadpool_limits(limits=1):
            rows, needles = tq.run_case("niah", 1024, 8, 11, chunk_tokens=512, sample_slots=128)
        for first, second in zip(self.rows, rows):
            for key in ("recall_at_128", "needle_absolute_rank", "distractor_false_positive_rate", "spearman_rho",
                        "dilution", "levels_used", "nmse", "cosine_similarity", "packed_stream_bytes"):
                self.assertAlmostEqual(first[key], second[key], places=6, msg=key)
        self.assertEqual(self.needles, needles)

    def test_streamed_ranks_against_dense_sort_oracle(self):
        seed = 11
        transform = tq.anisotropy(seed)
        needles, queries, distractors = tq.adversaries(seed, transform)
        special = np.r_[needles, distractors]
        positions = tq.special_positions(1024, seed)
        token_keys = tq.transform_rows(tq.gaussian_rows(np.arange(1024), seed ^ 0x484159), transform, 8)
        token_keys[positions] = special
        keys = token_keys.reshape(-1, 4, 128).mean(axis=1, dtype=np.float32)
        rotation = tq.Rotation(128, seed ^ 0x484144)
        keys, queries = rotation.apply(keys), rotation.apply(queries)
        exact_scores = tq.scores(keys, queries)
        # All pre-compression paired adversaries lie in the required cosine band.
        original_queries = tq.adversaries(seed, transform)[1]
        cosines = np.sum(original_queries * distractors, axis=1)
        self.assertTrue(np.all((cosines >= 0.70) & (cosines <= 0.82)))
        for row in self.rows:
            codec = tq.Quantizer(row["regime"], row["bits"])
            quant_scores = tq.scores(codec.decode(codec.encode(keys)), queries)
            per_needle = [r for r in self.needles if r["bits"] == row["bits"] and r["regime"] == row["regime"]]
            for q, result in enumerate(per_needle):
                order = np.lexsort((np.arange(256), -quant_scores[:, q]))
                self.assertEqual(result["needle_rank"], int(np.flatnonzero(order == positions[q] // 4)[0]) + 1)
                self.assertEqual(result["distractor_rank"], int(np.flatnonzero(order == positions[16 + q] // 4)[0]) + 1)
                exact_order = np.lexsort((np.arange(256), -exact_scores[:, q]))
                self.assertEqual(result["exact_needle_rank"], int(np.flatnonzero(exact_order == positions[q] // 4)[0]) + 1)
        self.assertEqual(len(self.needles), len(tq.REGIMES) * len(tq.BIT_WIDTHS) * 16)
        for row in self.rows:
            self.assertAlmostEqual(row["logit_nmse"], row["logit_relative_rmse"] ** 2, places=12)

    def test_spearman_ties_and_paired_t(self):
        a = np.array([1, 2, 2, 4, 7], dtype=float)
        b = np.array([2, 1, 1, 5, 3], dtype=float)
        self.assertAlmostEqual(tq.spearman_columns(a[:, None], b[:, None])[0], spearmanr(a, b).statistic)
        result = tq.paired_summary(b - a)
        expected = ttest_rel(b, a)
        self.assertAlmostEqual(result["t"], expected.statistic)
        self.assertAlmostEqual(result["p_two_sided"], expected.pvalue)
        self.assertEqual(tq.paired_summary(np.zeros(5))["status"], "zero_variance")
        self.assertIsNone(tq.paired_summary(np.zeros(5))["p_two_sided"])
        self.assertEqual(tq.paired_summary(np.ones(1))["status"], "insufficient_seeds")


class CrossBitParityTests(unittest.TestCase):
    def test_tost_matches_90_percent_ci_and_handles_degenerate_pairs(self):
        from tq_cross_bit_parity import equivalence
        effects = np.array([-0.0002, 0.0001, -0.0001, 0.00015, 0.0])
        result = equivalence(effects, -0.001, 0.001)
        self.assertTrue(result["equivalent_05"])
        self.assertGreater(result["ci90_low"], -0.001)
        self.assertLess(result["ci90_high"], 0.001)
        self.assertFalse(equivalence(effects + 0.003, -0.001, 0.001)["equivalent_05"])
        self.assertFalse(equivalence(np.zeros(5), -0.001, 0.001)["equivalent_05"])

    def test_cross_bit_pairing_uses_corrected_5_vs_4_and_4_vs_3(self):
        from tq_cross_bit_parity import cross_bit_pairs
        rows = []
        for seed in range(5):
            for bits, regime in ((5, tq.REGIMES[3]), (4, tq.REGIMES[0]),
                                 (4, tq.REGIMES[3]), (3, tq.REGIMES[0])):
                row = dict.fromkeys(tq.METRICS, 1.0)
                row.update(source="niah", context_length=8192, kappa=8.0, group_size=0,
                           bits=bits, regime=regime, seed=seed)
                row["nmse"] = row["logit_nmse"] = (0.002 if regime == tq.REGIMES[3] else 0.01) * (1 + 0.01 * seed)
                row["spearman_rho"] = (0.995 + seed * 0.00001 if regime == tq.REGIMES[3] else 0.99)
                rows.append(row)
        comparisons = cross_bit_pairs(rows)
        self.assertEqual({(r["candidate_bits"], r["baseline_bits"]) for r in comparisons}, {(5, 4), (4, 3)})
        self.assertTrue(all(r["n"] == 5 for r in comparisons))
        for row in comparisons:
            if row["metric"] in ("nmse", "logit_nmse"):
                self.assertAlmostEqual(row["mean_ratio"], 0.2)
                self.assertEqual(row["equivalence_scale"], "log(candidate/baseline)")


if __name__ == "__main__":
    unittest.main()
