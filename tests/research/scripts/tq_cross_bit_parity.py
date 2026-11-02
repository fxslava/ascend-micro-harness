"""Paired cross-bit comparisons: (uniform5, reference4) and (uniform4, reference3)."""

import math

import numpy as np
from scipy.stats import t as student_t

import tq_cdf_vs_lloydmax_ablation as tq


def equivalence(delta, low, high):
    """Paired TOST at alpha=.05, equivalent to a 90% CI inside (low, high)."""
    summary = tq.paired_summary(delta)
    result = dict(equivalence_low=low, equivalence_high=high, tost_p=None,
                  ci90_low=None, ci90_high=None, equivalent_05=False,
                  equivalence_status=summary["status"])
    if summary["status"] != "ok":
        # Identical finite-sample observations do not supply an estimated SE.
        return result
    se = float(np.std(delta, ddof=1)) / math.sqrt(len(delta))
    mean, df = summary["mean_delta"], len(delta) - 1
    p_low = student_t.sf((mean - low) / se, df)
    p_high = student_t.cdf((mean - high) / se, df)
    half = student_t.ppf(0.95, df) * se
    result.update(tost_p=float(max(p_low, p_high)), ci90_low=mean - half, ci90_high=mean + half,
                  equivalent_05=bool(max(p_low, p_high) < 0.05))
    return result


def cross_bit_pairs(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("source", "context_length", "kappa", "group_size"))
        groups.setdefault(key, {}).setdefault((row["regime"], row["bits"]), {})[row["seed"]] = row
    output = []
    for key, configs in groups.items():
        for candidate_bits, baseline_bits in ((5, 4), (4, 3)):
            for candidate in tq.REGIMES[3:]:
                for baseline in tq.REGIMES[:3]:
                    a = configs.get((candidate, candidate_bits), {})
                    b = configs.get((baseline, baseline_bits), {})
                    seeds = sorted(a.keys() & b.keys())
                    if not seeds:
                        continue
                    for metric in tq.METRICS:
                        if any(a[s][metric] is None or b[s][metric] is None for s in seeds):
                            continue
                        av = np.array([a[s][metric] for s in seeds])
                        bv = np.array([b[s][metric] for s in seeds])
                        row = dict(zip(("source", "context_length", "kappa", "group_size"), key),
                                   candidate_regime=candidate, candidate_bits=candidate_bits,
                                   baseline_regime=baseline, baseline_bits=baseline_bits, metric=metric,
                                   mean_candidate=float(np.mean(av)), mean_baseline=float(np.mean(bv)),
                                   mean_ratio=float(np.mean(av) / np.mean(bv)) if np.mean(bv) else None,
                                   **tq.paired_summary(av - bv))
                        # Explicit practical equivalence bands, kept separate from
                        # zero-loss and two-sided difference tests. Not population guarantees.
                        band = {"recall_at_128": 1 / 16, "spearman_rho": 0.001,
                                "distractor_false_positive_rate": 1 / 16}.get(metric)
                        if metric in ("nmse", "logit_nmse"):
                            effect = np.log(av / bv)
                            band, scale, direction = math.log(1.10), "log(candidate/baseline)", -1
                        else:
                            effect, scale = av - bv, "candidate-baseline"
                            direction = -1 if metric == "distractor_false_positive_rate" else 1
                        row.update(equivalence_scale=scale,
                                   equivalence_low=None, equivalence_high=None, tost_p=None,
                                   ci90_low=None, ci90_high=None, equivalent_05=False,
                                   equivalence_status="not_assessed", strict_noninferiority_supported=False,
                                   strict_inferiority_supported=False, noninferiority_with_margin_supported=False,
                                   oriented_lower_95=None, oriented_upper_95=None)
                        if band is not None:
                            row.update(equivalence(effect, -band, band))
                            oriented = tq.paired_summary(direction * effect)
                            lower = oriented["lower_one_sided_95"]
                            upper = None if lower is None else 2 * oriented["mean_delta"] - lower
                            supported = oriented["status"] == "ok"
                            # Recall parity additionally reports the strict zero-loss test.
                            row.update(oriented_lower_95=lower, oriented_upper_95=upper,
                                       strict_noninferiority_supported=bool(supported and lower > 0),
                                       strict_inferiority_supported=bool(supported and upper < 0),
                                       noninferiority_with_margin_supported=bool(supported and lower > -band))
                        output.append(row)
    return output


def cross_bit_markdown(comparisons):
    lines = ["# Cross-bit parity: plain uniform versus Lloyd-Max / inverse CDF", "",
             "Comparisons: **5-bit uniform vs 4-bit reference**, and **4-bit uniform vs 3-bit reference**. "
             "'Plain' retains the common D H D rotation and one FP16 absmax-derived step per vector; "
             "it omits CDF flattening and Lloyd-Max codebooks. Both midrise and midtread are tested. "
             "The three references are Lloyd-Max, exact Gaussian-CDF, and sigmoid-CDF. "
             "The same five seeds are paired within each context or synthetic kappa stratum.", "",
             "Full paired differences, two-sided t tests and 95% CIs for all metrics are in cross_bit.csv. "
             "Practical equivalence uses paired TOST at alpha=.05 (a 90% CI inside the band): "
             "Recall/FPR +/-1/16, rho +/-0.001, and NMSE ratios within [1/1.10,1.10] "
             "(tested in log space). These are explicit working tolerances, not a claim that any "
             "loss is acceptable to every application. Strict zero-loss noninferiority is recorded separately. "
             "No multiplicity correction; zero-variance or single-seed tests remain inconclusive. "
             "A better method can fail equivalence because it lies outside the band in the favorable direction.", ""]
    lines += ["Table labels: equiv=TOST equivalence within the working band; better/worse=one-sided "
              "95% zero-loss test in the indicated direction; unresolved=neither conclusion supported. "
              "These labels do not replace the paired 95% difference CIs.", ""]
    niah = [r for r in comparisons if r["source"] == "niah"]
    if niah:
        lines += ["## NIAH synthesis", "",
                  "Ranges below span the three contexts and both uniform grids. They summarize separate "
                  "strata; confidence intervals and tests remain paired within each stratum. "
                  "CDF includes exact Gaussian and sigmoid variants.", "",
                  "| Comparison | Reference family | NMSE(key) ratio | NMSE(QK) ratio | Rho delta | Recall delta (percentage points) |",
                  "|---|---|---|---|---|---|"]
        for candidate_bits, baseline_bits in ((5, 4), (4, 3)):
            for family in ("Lloyd-Max", "CDF"):
                subset = [r for r in niah if r["candidate_bits"] == candidate_bits and
                          (r["baseline_regime"] == tq.REGIMES[0]) == (family == "Lloyd-Max")]
                cells = []
                for metric in ("nmse", "logit_nmse", "spearman_rho", "recall_at_128"):
                    selected = [r for r in subset if r["metric"] == metric]
                    values = [r["mean_ratio"] if metric in ("nmse", "logit_nmse") else
                              r["mean_delta"] * (100 if metric == "recall_at_128" else 1) for r in selected]
                    cells.append(f"{min(values):.6g} to {max(values):.6g}")
                lines.append("| " + " | ".join([f"uniform {candidate_bits} vs reference {baseline_bits} bits", family, *cells]) + " |")
        fidelity = [r for r in niah if r["metric"] in ("nmse", "logit_nmse", "spearman_rho")]
        supported = sum(r["status"] == "ok" and (r["ci95_high"] < 0 if r["metric"] != "spearman_rho" else r["ci95_low"] > 0)
                        for r in fidelity)
        recall = [r for r in niah if r["metric"] == "recall_at_128"]
        losses = sum(r["mean_delta"] < 0 for r in recall)
        significant_recall = sum(r["p_two_sided"] is not None and r["p_two_sided"] < 0.05 for r in recall)
        lines += ["", f"The extra-bit uniform method has a favorable two-sided 95% difference CI in "
                  f"{supported}/{len(fidelity)} fidelity comparisons (key NMSE, logit NMSE, rho). "
                  f"Mean recall is lower in {losses}/{len(recall)} comparisons; "
                  f"{significant_recall}/{len(recall)} recall difference tests have p<0.05. "
                  "This supports the fidelity advantage on this grid, while strict loss-free retrieval parity "
                  "is not established. Absence of a significant recall difference is not proof of parity.", ""]
    for source in sorted({r["source"] for r in comparisons}):
        lines += [f"## {source}: paired cross-bit results", "",
                  "Ratio <1 means less reconstruction/projection error for uniform. Positive rho/recall "
                  "delta favors uniform. NMSE(key) and NMSE(QK) are distinct measures.", "",
                  "| Context / kappa | Comparison | Uniform grid | Reference | Recall (uniform / ref) | Rho delta [95% CI] | NMSE(key) ratio | NMSE(QK) ratio | TOST: recall / rho / key / QK |",
                  "|---|---|---|---|---|---|---:|---:|---|"]
        groups = {}
        for r in comparisons:
            if r["source"] != source:
                continue
            key = tuple(r[k] for k in ("context_length", "kappa", "candidate_bits", "baseline_bits", "candidate_regime", "baseline_regime"))
            groups.setdefault(key, {})[r["metric"]] = r
        fmt = tq.display
        for (context, kappa, cb, bb, candidate, baseline), metrics in groups.items():
            rho = metrics["spearman_rho"]
            recall = metrics.get("recall_at_128")
            statuses = []
            for metric in ("recall_at_128", "spearman_rho", "nmse", "logit_nmse"):
                r = metrics.get(metric)
                statuses.append("n/a" if r is None else "equiv" if r["equivalent_05"] else
                                "better" if r["strict_noninferiority_supported"] else
                                "worse" if r["strict_inferiority_supported"] else "unresolved")
            recall_text = f"{fmt(recall['mean_candidate'])} / {fmt(recall['mean_baseline'])}" if recall else "n/a"
            fields = [context or kappa, f"{cb} vs {bb} bits", candidate, baseline, recall_text,
                      f"{fmt(rho['mean_delta'])} [{fmt(rho['ci95_low'])}, {fmt(rho['ci95_high'])}]",
                      fmt(metrics["nmse"]["mean_ratio"]), fmt(metrics["logit_nmse"]["mean_ratio"]), " / ".join(statuses)]
            lines.append("| " + " | ".join(map(str, fields)) + " |")
    return "\n".join(lines) + "\n"
