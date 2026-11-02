#!/usr/bin/env python3
"""CPU ablation of rotated 3/4/5-bit quantizers (NumPy, SciPy, threadpoolctl).

Examples (from the repository root):
  python tests/research/scripts/tq_cdf_vs_lloydmax_ablation.py --source synthetic
  python tests/research/scripts/tq_cdf_vs_lloydmax_ablation.py --source niah

Adapted from ascend-moe-engine/tests/research: Gaussian Lloyd-Max from
turbo_lattice_vq.py; little-endian packing and counter RNG from
extreme_niah_benchmark.py; anisotropy/adversaries from
adversarial_multi_needle_benchmark.py; half-mode RoPE and absolute logit share
from dsv4_mla_niah_benchmark.py. No external checkout is required at runtime.
This isolates the scalar codec: no FP8 intermediate, lattice residual or QJL.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import scipy
from scipy.special import expit, ndtr, ndtri
from scipy.stats import rankdata, t as student_t
from threadpoolctl import threadpool_limits

REGIMES = ("TQ-LloydMax", "TQ-GaussianCDF", "TQ-Sigmoid-Vasicek",
           "TQ-Uniform-Midrise", "TQ-Uniform-Midtread")
BIT_WIDTHS = (3, 4, 5)
DIM, NEEDLES, TOP_K, RATIO = 128, 16, 128, 4
C1, C3 = 1.5976, 0.070565  # requested cubic logistic fit; c5 = 0
ROOT = Path(__file__).resolve().parents[3]
METRICS = (
    "recall_at_128", "needle_absolute_rank", "distractor_false_positive_rate",
    "spearman_rho", "dilution", "nmse", "logit_nmse", "cosine_similarity", "logit_relative_rmse",
    "mean_rope_logit_share", "counterfactual_rope_phase_error_deg", "levels_used", "code_levels_used",
    "codec_latency_s", "stream_latency_s", "codec_us_per_slot", "us_per_slot",
)


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


class Rotation:
    """Orthogonal Pi = D H D, with normalized Walsh-Hadamard H (self-inverse)."""

    def __init__(self, dimension: int, seed: int):
        if dimension < 1 or dimension & (dimension - 1):
            raise ValueError("Hadamard dimension must be a power of two")
        self.signs = np.random.default_rng(seed).choice(
            np.array([-1, 1], dtype=np.float32), dimension
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32) * self.signs
        width = 1
        while width < y.shape[1]:
            blocks = y.reshape(len(y), -1, 2 * width)
            low, high = blocks[..., :width].copy(), blocks[..., width:].copy()
            blocks[..., :width] = low + high
            blocks[..., width:] = low - high
            width *= 2
        return y * (self.signs / np.float32(math.sqrt(y.shape[1])))


@lru_cache(maxsize=3)
def lloyd_max(bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Preserve 3/4-bit reference; converge the new 32-level Gaussian codebook."""
    if bits not in BIT_WIDTHS:
        raise ValueError("Lloyd-Max supports 3, 4 or 5 bits")
    levels = 1 << bits
    centres = ndtri((np.arange(levels) + 0.5) / levels)
    for _ in range(20000 if bits == 5 else 256):
        edges = np.r_[-np.inf, (centres[:-1] + centres[1:]) / 2, np.inf]
        density = np.exp(-edges * edges / 2) / math.sqrt(2 * math.pi)
        updated = (density[:-1] - density[1:]) / np.diff(ndtr(edges))
        converged = np.max(np.abs(updated - centres)) < 1e-14
        centres = updated
        if converged:
            break
    if bits == 5 and not converged:
        raise RuntimeError("32-level Gaussian Lloyd-Max failed to converge")
    centres = centres.astype(np.float32)
    return centres, (centres[:-1] + centres[1:]) / 2


def inverse_sigmoid_polynomial(u: np.ndarray) -> np.ndarray:
    """Three Newton iterations; maximum residual on 3/4/5-bit midpoints < 2e-6."""
    w = np.log(u) - np.log1p(-u)
    z = w / C1
    for _ in range(3):
        z -= (z * (C1 + C3 * z * z) - w) / (C1 + 3 * C3 * z * z)
    return z


def pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    planes = ((indices[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1)
    return np.packbits(planes.reshape(len(indices), -1), axis=1, bitorder="little")


def unpack_indices(payload: np.ndarray, dimension: int, bits: int) -> np.ndarray:
    planes = np.unpackbits(payload, axis=1, count=dimension * bits, bitorder="little")
    planes = planes.reshape(len(payload), dimension, bits)
    return np.sum(planes * (1 << np.arange(bits, dtype=np.uint8)), axis=2, dtype=np.uint8)


@dataclass
class Packed:
    codes: np.ndarray
    scales: np.ndarray
    dimension: int


class Quantizer:
    def __init__(self, regime: str, bits: int, group_size: int = 0):
        if regime not in REGIMES or bits not in BIT_WIDTHS:
            raise ValueError("expected a supported regime and 3, 4 or 5 bits")
        if group_size < 0 or (group_size and 64 % group_size):
            raise ValueError("group size must be zero (per vector) or divide both 64 and 128")
        self.regime, self.bits, self.group_size = regime, bits, group_size
        self.scale_kind = "absmax_step" if regime in REGIMES[3:] else "rms"
        if regime == REGIMES[0]:
            self.centres, self.thresholds = lloyd_max(bits)

    def encode(self, rotated: np.ndarray) -> Packed:
        x = np.asarray(rotated, dtype=np.float32)
        if x.ndim != 2 or not np.all(np.isfinite(x)):
            raise ValueError("codec expects finite [vectors, dimension] inputs")
        group = self.group_size or x.shape[1]
        if x.shape[1] % group:
            raise ValueError("group size must divide input dimension")
        grouped = x.reshape(len(x), -1, group)
        half = 1 << (self.bits - 1)
        if self.scale_kind == "absmax_step":
            denominator = half - (0.5 if self.regime == REGIMES[3] else 1)
            scale = np.max(np.abs(grouped), axis=2).astype(np.float64) / denominator
        else:
            scale = np.sqrt(np.mean(grouped.astype(np.float64) ** 2, axis=2))
        # Encode using the stored FP16 scale, avoiding encoder/decoder mismatch.
        scales = scale.astype(np.float16)
        if np.any(~np.isfinite(scales)) or np.any((scale > 0) & (scales == 0)):
            raise ValueError("scale is outside representable nonzero FP16 range")
        divisor = np.where(scales == 0, 1, scales).astype(np.float32)
        z = (grouped / divisor[:, :, None]).reshape(x.shape)
        if self.regime == REGIMES[0]:
            indices = np.searchsorted(self.thresholds, z, side="right").astype(np.uint8)
        elif self.regime == REGIMES[3]:
            indices = np.clip(np.floor(z + half), 0, (1 << self.bits) - 1).astype(np.uint8)
        elif self.regime == REGIMES[4]:
            # Signed nearest integer, ties to even; shift into an unsigned payload.
            signed = np.clip(np.rint(z), -half, half - 1).astype(np.int16)
            indices = (signed + half).astype(np.uint8)
        else:
            u = ndtr(z) if self.regime == REGIMES[1] else expit(z * (C1 + C3 * z * z))
            indices = np.clip(np.floor(u * (1 << self.bits)), 0, (1 << self.bits) - 1).astype(np.uint8)
        return Packed(pack_indices(indices, self.bits), scales, x.shape[1])

    def decode(self, packed: Packed) -> np.ndarray:
        indices = unpack_indices(packed.codes, packed.dimension, self.bits)
        if self.regime == REGIMES[0]:
            z = self.centres[indices]
        elif self.regime in REGIMES[3:]:
            # Cast before subtracting the signed zero point (uint8 would wrap).
            z = indices.astype(np.float32) - (1 << (self.bits - 1))
            if self.regime == REGIMES[3]:
                z += 0.5
        else:
            u = (indices.astype(np.float32) + 0.5) / (1 << self.bits)
            # Evaluate every coordinate: do not cache an analytical codebook.
            z = ndtri(u) if self.regime == REGIMES[1] else inverse_sigmoid_polynomial(u)
        group = self.group_size or packed.dimension
        return (z.reshape(len(z), -1, group) * packed.scales.astype(np.float32)[:, :, None]).reshape(z.shape)


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def gaussian_rows(positions: np.ndarray, seed: int, dimension: int = DIM) -> np.ndarray:
    """Random-access Box-Muller Gaussian rows: independent of streaming chunks."""
    positions = positions.astype(np.uint64)[:, None]
    lanes = np.arange(dimension // 2, dtype=np.uint64)[None, :]
    counter = (np.uint64(seed) ^ ((positions + np.uint64(1)) * np.uint64(0xD6E8FEB86659FD93))
               ^ ((lanes + np.uint64(1)) * np.uint64(0xA5A3564E27F8862B)))
    u = ((splitmix64(counter) >> np.uint64(11)).astype(np.float64) + 0.5) * (2.0 ** -53)
    v = ((splitmix64(counter ^ np.uint64(0x123456789ABCDEF)) >> np.uint64(11)).astype(np.float64) + 0.5) * (2.0 ** -53)
    radius, angle = np.sqrt(-2 * np.log(u)), 2 * np.pi * v
    return np.concatenate((radius * np.cos(angle), radius * np.sin(angle)), axis=1).astype(np.float32)


def anisotropy(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed ^ 0x434F5641)
    basis, triangular = np.linalg.qr(rng.standard_normal((DIM, DIM)))
    basis *= np.where(np.diag(triangular) < 0, -1, 1)[None, :]
    return (np.sqrt(np.exp(-2.5 * np.arange(DIM) / DIM))[:, None] * basis.T).astype(np.float32)


def transform_rows(x: np.ndarray, transform: np.ndarray | None, kappa: float) -> np.ndarray:
    if transform is not None:
        # Fixed reduction order avoids chunk-size-dependent matrix multiplication.
        x = np.einsum("ij,jk->ik", x, transform, optimize=False)
    x[:, (0, 16, 64, 96)] *= np.float32(kappa)
    return unit_rows(x) if transform is not None else x


def orthogonal(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(DIM).astype(np.float32)
    x -= np.dot(x, reference) * reference
    return unit_rows(x[None])[0]


def adversaries(seed: int, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed ^ 0x414456)
    needles = transform_rows(rng.standard_normal((NEEDLES, DIM)).astype(np.float32), transform, 8)
    queries, distractors = np.empty_like(needles), np.empty_like(needles)
    for i, needle in enumerate(needles):
        queries[i] = unit_rows((needle + np.float32(10 ** (-9 / 20)) * orthogonal(needle, rng))[None])[0]
        cosine = rng.uniform(0.70, 0.82)
        distractors[i] = cosine * queries[i] + math.sqrt(1 - cosine ** 2) * orthogonal(queries[i], rng)
    return needles, queries, distractors


def special_positions(context: int, seed: int) -> np.ndarray:
    needles = np.rint(np.linspace(0.05, 0.95, NEEDLES) * (context - 1)).astype(np.int64)
    forbidden = {int(p // RATIO) + offset for p in needles for offset in (-1, 0, 1)}
    rng = random.Random(seed ^ context ^ 0x504C4143)
    distractors = []
    while len(distractors) < NEEDLES:
        slot = rng.randrange(context // RATIO)
        if slot not in forbidden:
            forbidden.add(slot)
            distractors.append(slot * RATIO + rng.randrange(RATIO))
    return np.r_[needles, distractors]


def scores(keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return np.einsum("nd,qd->nq", keys, queries, optimize=False) / np.float32(math.sqrt(DIM))


def spearman_columns(exact: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    a, b = rankdata(exact, axis=0), rankdata(decoded, axis=0)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return np.divide(np.sum(a * b, axis=0), denominator, out=np.full(a.shape[1], np.nan), where=denominator > 0)


def rope_probe(codec: Quantizer, seed: int) -> dict:
    """Isolated decoupled 64-lane probe, not part of NIAH retrieval scores."""
    rng = np.random.default_rng(seed ^ 0x524F5045)
    rope = rng.standard_normal((256, 64)).astype(np.float32)
    positions = rng.integers(0, 131072, size=256)
    angles = positions[:, None] * 160000.0 ** (-2 * np.arange(32)[None] / 64)
    lo, hi = rope[:, :32].copy(), rope[:, 32:].copy()
    rope[:, :32] = np.cos(angles) * lo - np.sin(angles) * hi
    rope[:, 32:] = np.sin(angles) * lo + np.cos(angles) * hi
    rotation = Rotation(64, seed ^ 0x524F54)
    decoded = rotation.apply(codec.decode(codec.encode(rotation.apply(rope))))
    cosine = np.sum(rope * decoded, axis=1, dtype=np.float64) / (
        np.linalg.norm(rope.astype(np.float64), axis=1) * np.linalg.norm(decoded.astype(np.float64), axis=1))
    phase = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    # Existing DSV4 helper defines energy share via absolute contributions.
    rotary = np.sum(rope * rng.standard_normal(rope.shape), axis=1)
    content = np.sum(rng.standard_normal((256, DIM)) * rng.standard_normal((256, DIM)), axis=1)
    share = np.mean(np.abs(rotary) / np.maximum(np.abs(rotary) + np.abs(content), 1e-30))
    return {"mean_rope_logit_share": float(share),
            "counterfactual_rope_phase_error_deg": float(np.mean(phase))}


def run_case(source: str, size: int, kappa: float, seed: int, *, bits=BIT_WIDTHS,
             group_size=0, chunk_tokens=8192, sample_slots=4096) -> tuple[list[dict], list[dict]]:
    if source not in ("synthetic", "niah") or size < 1024 or (source == "niah" and size % 4):
        raise ValueError("size must be >=1024; NIAH context must be divisible by four")
    if chunk_tokens <= 0 or chunk_tokens % 4 or sample_slots <= 0:
        raise ValueError("chunk tokens must be a positive multiple of four; sample slots positive")
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    rotation = Rotation(DIM, seed ^ 0x484144)
    codecs = [Quantizer(r, b, group_size) for b in bits for r in REGIMES]
    ratio = RATIO if source == "niah" else 1
    count = size // ratio
    transform = anisotropy(seed) if source == "niah" else None
    positions = special_positions(size, seed) if source == "niah" else np.array([], dtype=np.int64)
    if source == "niah":
        needles, queries, distractors = adversaries(seed, transform)
        special = np.r_[needles, distractors]
    else:
        queries = unit_rows(transform_rows(gaussian_rows(np.arange(NEEDLES), seed ^ 0x515545), None, kappa))
    qrot = rotation.apply(queries)

    def build(token_positions: np.ndarray) -> np.ndarray:
        x = transform_rows(gaussian_rows(token_positions, seed ^ 0x484159), transform, kappa)
        if source == "niah":
            order = np.argsort(token_positions)
            sorted_positions = token_positions[order]
            for position, vector in zip(positions, special):
                local = np.searchsorted(sorted_positions, position)
                if local < len(sorted_positions) and sorted_positions[local] == position:
                    x[order[local]] = vector
        return x.reshape(-1, ratio, DIM).mean(axis=1, dtype=np.float32)

    special_slots = positions // ratio
    excluded = set(map(int, special_slots))
    # random.sample(range) is O(samples), with no context-length-sized bitmap.
    sampler = random.Random(seed ^ size ^ 0x53414D50)
    pool = sorted(set(sampler.sample(range(count), min(count, sample_slots + len(excluded)))) - excluded)
    sampled = np.array(sorted(sampler.sample(pool, min(sample_slots, len(pool)))), dtype=np.int64)
    sample_exact = np.empty((len(sampled), NEEDLES), dtype=np.float32)
    if source == "niah":
        token_positions = (special_slots[:, None] * ratio + np.arange(ratio)).reshape(-1)
        special_keys = rotation.apply(build(token_positions))
        special_exact = scores(special_keys, qrot)
    else:
        special_keys = np.empty((0, DIM), dtype=np.float32)
        special_exact = np.empty((0, NEEDLES), dtype=np.float32)

    states = []
    for codec in codecs:
        # Warm up every codec, outside timing, before streaming in shuffled order.
        warm = rotation.apply(build(np.arange(128 * ratio)))
        codec.decode(codec.encode(warm))
        decoded = codec.decode(codec.encode(special_keys)) if len(special_keys) else special_keys
        special_scores = scores(decoded, qrot)
        states.append(dict(codec=codec, special_decoded=decoded, special_scores=special_scores,
                           sample=np.empty_like(sample_exact), ranks=np.ones(NEEDLES, dtype=np.int64),
                           distractor_ranks=np.ones(NEEDLES, dtype=np.int64), exact_ranks=np.ones(NEEDLES, dtype=np.int64),
                           latency=0.0, codec_latency=0.0, levels=0.0, code_levels=0.0, squared_error=0.0,
                           energy=0.0, cosine=0.0, logit_error=0.0, logit_energy=0.0, packed_bytes=0))
    common_latency = 0.0
    order_rng = random.Random(seed ^ 0x54494D45)
    for start in range(0, size, chunk_tokens):
        common_start = time.perf_counter()
        stop = min(size, start + chunk_tokens)
        rotated = rotation.apply(build(np.arange(start, stop)))
        exact = scores(rotated, qrot)
        first, last = start // ratio, stop // ratio
        slot_ids = np.arange(first, last)
        begin, end = np.searchsorted(sampled, (first, last))
        sample_exact[begin:end] = exact[sampled[begin:end] - first]
        common_latency += time.perf_counter() - common_start
        order = list(range(len(states)))
        order_rng.shuffle(order)
        for index in order:
            state = states[index]
            codec = state["codec"]
            started = time.perf_counter()
            packed = codec.encode(rotated)
            decoded = codec.decode(packed)
            state["codec_latency"] += time.perf_counter() - started
            quantized = scores(decoded, qrot)
            state["sample"][begin:end] = quantized[sampled[begin:end] - first]
            if source == "niah":
                for q in range(NEEDLES):
                    for key, target, target_slot in (
                        ("ranks", state["special_scores"][q, q], special_slots[q]),
                        ("distractor_ranks", state["special_scores"][NEEDLES + q, q], special_slots[NEEDLES + q]),
                        ("exact_ranks", special_exact[q, q], special_slots[q]),
                    ):
                        candidate = exact[:, q] if key == "exact_ranks" else quantized[:, q]
                        # Exclude self explicitly; ties go to the smaller slot ID.
                        better = ((candidate > target) | ((candidate == target) & (slot_ids < target_slot)))
                        state[key][q] += np.count_nonzero(better & (slot_ids != target_slot))
            state["latency"] += time.perf_counter() - started
            # Diagnostic reductions are deliberately outside stream/codec timing.
            indices = unpack_indices(packed.codes, DIM, codec.bits)
            present = np.zeros((len(indices), 1 << codec.bits), dtype=bool)
            present[np.arange(len(indices))[:, None], indices] = True
            state["code_levels"] += np.sum(present)
            ordered_coordinates = np.sort(decoded, axis=1)
            state["levels"] += len(decoded) + np.count_nonzero(np.diff(ordered_coordinates, axis=1), axis=None)
            state["packed_bytes"] += packed.codes.nbytes + packed.scales.nbytes
            state["squared_error"] += np.sum((rotated.astype(np.float64) - decoded) ** 2)
            state["energy"] += np.sum(rotated.astype(np.float64) ** 2)
            state["cosine"] += np.sum(np.sum(rotated * decoded, axis=1) / np.maximum(
                np.linalg.norm(rotated, axis=1) * np.linalg.norm(decoded, axis=1), 1e-30))
            state["logit_error"] += np.sum((exact.astype(np.float64) - quantized) ** 2)
            state["logit_energy"] += np.sum(exact.astype(np.float64) ** 2)

    rows, needle_rows = [], []
    for state in states:
        codec = state["codec"]
        rho = spearman_columns(np.r_[sample_exact, special_exact], np.r_[state["sample"], state["special_scores"]])
        identity = dict(source=source, context_length=size if source == "niah" else 0,
                        vectors=count, kappa=kappa, seed=seed, bits=codec.bits,
                        regime=codec.regime, group_size=group_size, scale_kind=codec.scale_kind)
        per_needle = []
        if source == "niah":
            for q in range(NEEDLES):
                exact_cos = np.dot(qrot[q], special_keys[q]) / (np.linalg.norm(qrot[q]) * np.linalg.norm(special_keys[q]))
                quant_cos = np.dot(qrot[q], state["special_decoded"][q]) / (np.linalg.norm(qrot[q]) * np.linalg.norm(state["special_decoded"][q]))
                rank, distractor_rank = int(state["ranks"][q]), int(state["distractor_ranks"][q])
                per_needle.append(dict(**identity, needle=q, needle_slot=int(special_slots[q]),
                                       needle_rank=rank, needle_absolute_rank=rank, recall_at_128=int(rank <= TOP_K),
                                       exact_needle_rank=int(state["exact_ranks"][q]),
                                       exact_recall_at_128=int(state["exact_ranks"][q] <= TOP_K),
                                       distractor_rank=distractor_rank, distractor_in_top_128=int(distractor_rank <= TOP_K),
                                       dilution=float(quant_cos / exact_cos), spearman_rho=float(rho[q])))
            needle_rows.extend(per_needle)
        latency = common_latency + state["latency"]
        row = dict(**identity, recall_at_128=float(np.mean([r["recall_at_128"] for r in per_needle])) if per_needle else None,
                   needle_absolute_rank=float(np.mean(state["ranks"])) if per_needle else None,
                   exact_needle_rank=float(np.mean(state["exact_ranks"])) if per_needle else None,
                   exact_recall_at_128=float(np.mean(state["exact_ranks"] <= TOP_K)) if per_needle else None,
                   distractor_false_positive_rate=float(np.mean(state["distractor_ranks"] <= TOP_K)) if per_needle else None,
                   spearman_rho=float(np.mean(rho)),
                   dilution=float(np.mean([r["dilution"] for r in per_needle])) if per_needle else None,
                   nmse=float(state["squared_error"] / state["energy"]),
                   logit_nmse=float(state["logit_error"] / state["logit_energy"]),
                   cosine_similarity=float(state["cosine"] / count),
                   logit_relative_rmse=float(math.sqrt(state["logit_error"] / state["logit_energy"])),
                   levels_used=float(state["levels"] / count), code_levels_used=float(state["code_levels"] / count),
                   spearman_samples=len(sampled) + len(special_slots),
                   stream_latency_s=latency, us_per_slot=latency * 1e6 / count,
                   codec_latency_s=state["codec_latency"], codec_us_per_slot=state["codec_latency"] * 1e6 / count,
                   packed_stream_bytes=state["packed_bytes"], effective_bits_per_coordinate=state["packed_bytes"] * 8 / (count * DIM),
                   **rope_probe(codec, seed))
        rows.append(row)
    return rows, needle_rows


def paired_summary(delta: np.ndarray) -> dict:
    n = len(delta)
    mean = float(np.mean(delta))
    if n < 2:
        return dict(n=n, mean_delta=mean, t=None, p_two_sided=None, ci95_low=None, ci95_high=None,
                    lower_one_sided_95=None, status="insufficient_seeds")
    sd = float(np.std(delta, ddof=1))
    if sd == 0:
        # A zero-variance t statistic is undefined, including all-identical recall.
        return dict(n=n, mean_delta=mean, t=None, p_two_sided=None, ci95_low=mean, ci95_high=mean,
                    lower_one_sided_95=mean, status="zero_variance")
    se = sd / math.sqrt(n)
    statistic = mean / se
    half = float(student_t.ppf(0.975, n - 1)) * se
    return dict(n=n, mean_delta=mean, t=statistic,
                p_two_sided=float(2 * student_t.sf(abs(statistic), n - 1)),
                ci95_low=mean - half, ci95_high=mean + half,
                lower_one_sided_95=mean - float(student_t.ppf(0.95, n - 1)) * se, status="ok")


def pairwise(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("source", "context_length", "kappa", "bits", "group_size"))
        groups.setdefault(key, {}).setdefault(row["regime"], {})[row["seed"]] = row
    comparisons = []
    for key, regimes in groups.items():
        baseline = regimes[REGIMES[0]]
        for regime in REGIMES[1:]:
            seeds = sorted(baseline.keys() & regimes[regime].keys())
            for metric in METRICS:
                if any(baseline[s][metric] is None for s in seeds):
                    continue
                delta = np.array([regimes[regime][s][metric] - baseline[s][metric] for s in seeds])
                comparisons.append(dict(zip(("source", "context_length", "kappa", "bits", "group_size"), key),
                                        regime=regime, baseline=REGIMES[0], metric=metric,
                                        **paired_summary(delta)))
    return comparisons


def display(x) -> str:
    return "n/a" if x is None else f"{x:.6g}" if isinstance(x, (float, np.floating)) else str(x)


def markdown(rows: list[dict], comparisons: list[dict], args: argparse.Namespace) -> str:
    lines = ["# Extended TurboQuant ablation: uniform grids and 5-bit scaling", "",
             "CPU synthetic evidence only; no CAModel or device latency claims.", "",
             "128 coordinates, Pi = D H D, one FP16 scale per vector or group; no FP8 intermediate. "
             "NIAH uses uniform 4:1 CSA reduction, 16 needles and 16 paired adversarial distractors "
             "(pre-compression cosine 0.70–0.82). Rank is 1-based against every compressed slot; "
             "ties prefer the smaller slot ID. FPR is the fraction of the 16 paired distractors in top-128.", "",
             "Synthetic inputs are Gaussian with pre-rotation gains on channels 0,16,64,96. "
             "Synthetic retrieval fields are n/a. NIAH uses anisotropic unit vectors with gain 8. "
             "Spearman uses sampled backgrounds plus all 32 special slots. Each seed is one paired observation; "
             "needles are averaged within a seed. Contexts and kappa strata are never pooled for t tests.", "",
             "GaussianCDF evaluates ndtr/ndtri. Sigmoid-Vasicek evaluates expit(1.5976*z + "
             "0.070565*z^3) and logit plus three Newton iterations at every reconstructed coordinate. "
             "The requested label identifies this cubic logistic fit; no claim of historical attribution is made. "
             "Analytical paths do not use reconstruction lookup tables.", "",
             "Unflattened Uniform-Midrise stores s=absmax/(2^(b-1)-0.5), codes floor(v/s+2^(b-1)), "
             "and reconstructs (code-2^(b-1)+0.5)*s. Uniform-Midtread stores s=absmax/(2^(b-1)-1), "
             "rounds to nearest signed integer (ties to even), clips to [-2^(b-1),2^(b-1)-1], "
             "and reconstructs q*s. Its symmetric absmax convention normally leaves the most negative code unused. "
             "Both normalize with the stored FP16 step and use no CDF. The other three methods store FP16 RMS. "
             "Scale storage overhead is identical; scale estimation rules differ intentionally.", "",
             "The 3/4-bit Lloyd-Max reference retains its original 256 updates. The 5-bit baseline solves "
             "the analytic 32-cell Gaussian conditional-mean equations to convergence (FP64 update <1e-14), "
             "then stores FP32 centres and thresholds; it is not a slice of a smaller codebook.", "",
             "NMSE(key) is sum(||K-Khat||^2)/sum(||K||^2), preserving the previous nmse CSV column. "
             "NMSE(QK) is sum((QK^T-QKhat^T)^2)/sum((QK^T)^2), recorded as logit_nmse. "
             "Both are evaluated across the full streamed cache; only Spearman is sampled.", "",
             "RoPE is a separate clean 64-channel half-mode probe, theta=160000. "
             "Phase error is the requested full-carrier acos angle, not an estimate of each pair's phase. "
             "Logit share follows DSV4's |rotary|/(|rotary|+|semantic|) convention. "
             "The counterfactual carrier is quantized; NIAH retrieval scores use content only.", "",
             "stream_latency_s includes shared data generation/rotation/exact scores and the method's "
             "encode/pack/decode/score/rank/sample work. codec_us_per_slot isolates encode/pack/decode. "
             "Setup, warmup, final correlation/RoPE and diagnostic reductions are excluded. "
             "Method order is shuffled per chunk. Latency is host NumPy/SciPy wall time.", "",
             "levels_used counts distinct decoded coordinates per vector (different group scales may "
             "produce more than 2^b values). code_levels_used in runs.csv counts distinct integer codes.", "",
             f"Seeds: {args.seeds}; group size: {args.group_size or 'per vector'}; "
             f"chunk tokens: {args.chunk_tokens}; BLAS threads: {args.threads}.", "",
             "## Mean metrics across seeds", "",
             "| Source | Context / kappa | Bits | Regime | Recall@128 | Exact recall | Rank | Distractor FPR | Rho | Dilution | NMSE(key) | NMSE(QK) | Cosine | Levels |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in ("source", "context_length", "kappa", "bits", "regime"))
        grouped.setdefault(key, []).append(row)
    def avg(rs, name):
        return None if rs[0][name] is None else float(np.mean([r[name] for r in rs]))
    for (source, context, kappa, bits, regime), rs in grouped.items():
        fields = [avg(rs, n) for n in ("recall_at_128", "exact_recall_at_128", "needle_absolute_rank", "distractor_false_positive_rate",
                                       "spearman_rho", "dilution", "nmse", "logit_nmse", "cosine_similarity", "levels_used")]
        lines.append("| " + " | ".join(map(display, [source, context if context else kappa, bits, regime, *fields])) + " |")
    lines += ["", "## RoPE and timing means", "",
              "| Source | Context / kappa | Bits | Regime | RoPE share | Phase error deg | Stream s | us/slot | Codec us/slot | Effective bits/coord |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for (source, context, kappa, bits, regime), rs in grouped.items():
        fields = [avg(rs, n) for n in ("mean_rope_logit_share", "counterfactual_rope_phase_error_deg", "stream_latency_s",
                                       "us_per_slot", "codec_us_per_slot", "effective_bits_per_coordinate")]
        lines.append("| " + " | ".join(map(display, [source, context if context else kappa, bits, regime, *fields])) + " |")
    lines += ["", "## Paired differences from Lloyd-Max", "",
              "Two-sided paired t tests, unadjusted exploratory p-values; 95% paired CIs. "
              "Positive deltas mean the candidate metric is larger. Full statistics for every metric are in paired.csv. "
              "Undefined t tests are reported as n/a with their status.", "",
              "| Source | Context / kappa | Bits | Regime | Metric | n | Delta | t | p | 95% CI | Status |",
              "|---|---|---:|---|---|---:|---:|---:|---:|---|---|"]
    for r in comparisons:
        if r["metric"] not in ("recall_at_128", "spearman_rho", "nmse", "logit_nmse", "codec_us_per_slot"):
            continue
        fields = [r["source"], r["context_length"] or r["kappa"], r["bits"], r["regime"], r["metric"],
                  r["n"], r["mean_delta"], r["t"], r["p_two_sided"],
                  f"[{display(r['ci95_low'])}, {display(r['ci95_high'])}]", r["status"]]
        lines.append("| " + " | ".join(map(display, fields)) + " |")
    lines += ["", "## Sigmoid-Vasicek parity assessment", "",
              f"Prespecified noninferiority margins: Recall@128={args.recall_margin:g}, "
              f"rho={args.rho_margin:g}. A one-sided 95% lower CI must exceed minus the margin; "
              "zero-variance and single-seed outcomes cannot establish statistical noninferiority. "
              "No multiplicity correction is applied. A nonsignificant paired t test does not establish parity.", ""]
    retrieval = [r for r in comparisons if r["source"] == "niah" and r["regime"] == REGIMES[2]
                 and r["metric"] in ("recall_at_128", "spearman_rho")]
    if not retrieval:
        lines += ["Retrieval parity is unassessed: run --source niah."]
    else:
        degraded = any(r["mean_delta"] < -(args.recall_margin if r["metric"] == "recall_at_128" else args.rho_margin)
                       for r in retrieval)
        established = all(r["status"] == "ok" and r["lower_one_sided_95"] > -(
            args.recall_margin if r["metric"] == "recall_at_128" else args.rho_margin) for r in retrieval)
        lines += ["Observed retrieval parity fails: at least one mean Recall@128 or rho loss exceeds its margin."
                  if degraded else "Statistical retrieval noninferiority meets the configured margins on this grid."
                  if established else "Statistical retrieval parity is not established. Mean losses are within the configured "
                  "margins, but at least one interval or zero-variance outcome is inconclusive."]
        exact_loss = any(r["mean_delta"] < 0 for r in retrieval)
        lines += ["At least one stratum has a lower mean Recall@128 or rho than Lloyd-Max."
                  if exact_loss else "No mean Recall@128 or rho degradation was observed on this grid."]
    lines += extension_synthesis(rows, comparisons)
    return "\n".join(lines) + "\n"


def extension_synthesis(rows: list[dict], comparisons: list[dict]) -> list[str]:
    """Compare distances without pooling independent statistical strata."""
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("source", "context_length", "kappa", "bits"))
        groups.setdefault(key, {}).setdefault(row["regime"], []).append(row)
    lines = ["", "## Unflattened grids: closer to Lloyd-Max or CDF?", "",
             "For each stratum, distance compares the mean metric with Lloyd-Max and the mean "
             "of the two CDF methods. 'Closer' describes absolute distance, not an equivalence test. "
             "No statistical observations are pooled across contexts or kappa values.", "",
             "| Source | Context / kappa | Bits | Uniform regime | NMSE(key) / Lloyd | Closer by NMSE(key) | Closer by NMSE(QK) | Closer by rho |",
             "|---|---|---:|---|---:|---|---|---|"]
    for (source, context, kappa, bits), regimes in groups.items():
        if not all(r in regimes for r in REGIMES):
            continue
        def mean(regime, metric):
            return float(np.mean([r[metric] for r in regimes[regime]]))
        for regime in REGIMES[3:]:
            closest = []
            for metric in ("nmse", "logit_nmse", "spearman_rho"):
                uniform, baseline = mean(regime, metric), mean(REGIMES[0], metric)
                cdf = (mean(REGIMES[1], metric) + mean(REGIMES[2], metric)) / 2
                closest.append("Lloyd-Max" if abs(uniform - baseline) < abs(uniform - cdf) else "CDF")
            fields = [source, context or kappa, bits, regime, mean(regime, "nmse") / mean(REGIMES[0], "nmse"), *closest]
            lines.append("| " + " | ".join(map(display, fields)) + " |")
    lines += ["", "## Does the 5-bit gap disappear?", "",
              "The following counts use separate context/kappa paired tests at 5 bits. "
              "Significant means two-sided p<0.05, without multiplicity adjustment. A nonsignificant "
              "difference is not evidence that the gap disappeared. Ratios are ranges of "
              "stratum means relative to same-bit Lloyd-Max, not pooled effect estimates.", "",
              "| Source | Regime | NMSE(key) / Lloyd range | Significant NMSE(key) strata | Significant NMSE(QK) strata | Significant rho strata | Significant recall strata |",
              "|---|---|---|---|---|---|---|"]
    for source in sorted({r["source"] for r in rows if r["bits"] == 5}):
        for regime in REGIMES[1:]:
            ratios = []
            for key, regimes in groups.items():
                if key[0] == source and key[3] == 5:
                    ratios.append(np.mean([r["nmse"] for r in regimes[regime]]) /
                                  np.mean([r["nmse"] for r in regimes[REGIMES[0]]]))
            counts = []
            for metric in ("nmse", "logit_nmse", "spearman_rho", "recall_at_128"):
                tests = [r for r in comparisons if r["source"] == source and r["bits"] == 5
                         and r["regime"] == regime and r["metric"] == metric]
                significant = sum(r["p_two_sided"] is not None and r["p_two_sided"] < 0.05 for r in tests)
                counts.append(f"{significant}/{len(tests)}" if tests else "n/a")
            lines.append("| " + " | ".join([source, regime, f"{min(ratios):.3f}–{max(ratios):.3f}", *counts]) + " |")
    return lines


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "niah"), default="synthetic")
    parser.add_argument("--bits", default="3,4,5")
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--kappas", default="1,8,64")
    parser.add_argument("--context-lengths", default="8192,32768,131072")
    parser.add_argument("--vectors", type=int, default=8192)
    parser.add_argument("--group-size", type=int, default=0, help="0=per-vector; otherwise must divide 64 and 128")
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument("--sample-slots", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--recall-margin", type=float, default=0.0)
    parser.add_argument("--rho-margin", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        bits = tuple(map(int, args.bits.split(",")))
        seeds = tuple(map(int, args.seeds.split(",")))
        kappas = tuple(map(float, args.kappas.split(",")))
        contexts = tuple(map(int, args.context_lengths.split(",")))
        if len(set(seeds)) != len(seeds) or min(seeds) < 0 or max(seeds) >= 2 ** 64:
            raise ValueError("seeds must be unique unsigned 64-bit integers")
        if len(set(bits)) != len(bits) or not set(bits) <= set(BIT_WIDTHS):
            raise ValueError("bits must be a nonempty unique subset of 3,4,5")
        if args.group_size < 0 or (args.group_size and 64 % args.group_size):
            raise ValueError("group size must be zero or a divisor of 64")
        if args.threads < 1 or any(not math.isfinite(v) or v < 0 for v in (args.recall_margin, args.rho_margin)):
            raise ValueError("positive threads and nonnegative margins required")
        if args.vectors < 1024 or any(n < 1024 or n % 4 for n in contexts):
            raise ValueError("vectors/contexts must be >=1024, contexts divisible by four")
        if any(not math.isfinite(k) or k <= 0 for k in kappas):
            raise ValueError("kappas must be finite and positive")
        if len(set(kappas)) != len(kappas) or len(set(contexts)) != len(contexts):
            raise ValueError("kappas and contexts must be unique")
        if args.chunk_tokens <= 0 or args.chunk_tokens % 4 or args.sample_slots <= 0:
            raise ValueError("positive chunk tokens divisible by four and positive sample slots required")
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir or ROOT / "reports" / "tq_cdf_vs_lloydmax" / "extended" / args.source
    output.mkdir(parents=True, exist_ok=True)
    rows, needles = [], []
    with threadpool_limits(limits=args.threads):
        for seed in seeds:
            cases = [(args.vectors, k) for k in kappas] if args.source == "synthetic" else [(n, 8.0) for n in contexts]
            for size, kappa in cases:
                print(f"Running {args.source} size={size} kappa={kappa:g} seed={seed}", file=sys.stderr, flush=True)
                batch, per_needle = run_case(args.source, size, kappa, seed, bits=bits,
                                             group_size=args.group_size, chunk_tokens=args.chunk_tokens,
                                             sample_slots=args.sample_slots)
                rows.extend(batch)
                needles.extend(per_needle)
                write_csv(output / "runs.csv", rows)
                write_csv(output / "needles.csv", needles)
    comparisons = pairwise(rows)
    write_csv(output / "paired.csv", comparisons)
    report = markdown(rows, comparisons, args)
    (output / "summary.md").write_text(report, encoding="utf-8")
    (output / "summary_ext.md").write_text(report, encoding="utf-8")
    metadata = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    command=[sys.executable, *sys.argv], python=sys.version, numpy=np.__version__,
                    scipy=scipy.__version__, platform=platform.platform(), processor=platform.processor(),
                    coefficient_reference="https://www.econstor.eu/bitstream/10419/188388/1/v02-i01-p114_60-313-1-PB.pdf",
                    c1=C1, c3=C3, c5=0.0, newton_iterations=3,
                    uniform_scaling="FP16 absmax-derived step; midrise denominator=2^(b-1)-0.5; midtread=2^(b-1)-1",
                    midtread_rounding="nearest, ties to even; clip [-2^(b-1),2^(b-1)-1]; unsigned zero point=2^(b-1)",
                    codebooks={str(b): {"centres": lloyd_max(b)[0].tolist(), "thresholds": lloyd_max(b)[1].tolist(),
                                       "fit": "analytic Gaussian iteration to <1e-14" if b == 5 else "legacy 256 updates"}
                               for b in bits},
                    parity_scope="CPU synthetic research; no NPU/CAModel gate requested or executed")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
