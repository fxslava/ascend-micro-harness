#!/usr/bin/env python3
"""Combine the extended source reports and audit preservation of old metrics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import tq_cdf_vs_lloydmax_ablation as tq
from tq_cross_bit_parity import cross_bit_markdown, cross_bit_pairs


def read_runs(path):
    strings = {"source", "regime", "scale_kind"}
    integers = {"context_length", "vectors", "seed", "bits", "group_size", "spearman_samples", "packed_stream_bytes"}
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: value if key in strings else None if value == "" else int(value) if key in integers else float(value)
                 for key, value in row.items()} for row in csv.DictReader(handle)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=tq.ROOT / "reports/tq_cdf_vs_lloydmax/extended")
    args = parser.parse_args(argv)
    root = args.report_root
    rows, needles = [], []
    metadata = {}
    audit = []
    checked = ("recall_at_128", "needle_absolute_rank", "distractor_false_positive_rate",
               "spearman_rho", "dilution", "nmse", "cosine_similarity", "logit_relative_rmse",
               "levels_used", "counterfactual_rope_phase_error_deg", "mean_rope_logit_share")
    identity = lambda r: tuple(r[k] for k in ("source", "context_length", "kappa", "seed", "bits", "regime", "group_size"))
    for source in ("synthetic", "niah"):
        current = read_runs(root / source / "runs.csv")
        rows.extend(current)
        metadata[source] = json.loads((root / source / "metadata.json").read_text(encoding="utf-8"))
        config = metadata[source]["arguments"]
        conditions = config["kappas"] if source == "synthetic" else config["context_lengths"]
        expected = len(config["seeds"].split(",")) * len(config["bits"].split(",")) * len(conditions.split(",")) * len(tq.REGIMES)
        if len(current) != expected or len({identity(r) for r in current}) != expected:
            raise RuntimeError(f"{source}: incomplete or duplicated grid, expected {expected} unique rows, got {len(current)}")
        if source == "niah":
            with (root / source / "needles.csv").open(newline="", encoding="utf-8") as handle:
                needles.extend(csv.DictReader(handle))
        legacy = root.parent / source / "runs.csv"
        if legacy.exists():
            old = {identity(r): r for r in read_runs(legacy)}
            matches, maximum = 0, 0.0
            for row in current:
                previous = old.get(identity(row))
                if previous is None:
                    continue
                matches += 1
                for metric in checked:
                    if row[metric] is not None:
                        maximum = max(maximum, abs(row[metric] - previous[metric]))
            if maximum > 1e-7:
                raise RuntimeError(f"{source} legacy non-timing metric changed: max delta {maximum}")
            audit.append(f"{source}: {matches} prior 3/4-bit rows compared; maximum non-timing metric delta {maximum:g}.")
        else:
            audit.append(f"{source}: prior results unavailable; legacy comparison not executed.")
    paired = tq.pairwise(rows)
    cross_bit = cross_bit_pairs(rows)
    namespace = argparse.Namespace(**metadata["niah"]["arguments"])
    report = tq.markdown(rows, paired, namespace)
    cross_report = cross_bit_markdown(cross_bit)
    (root / "cross_bit_parity.md").write_text(cross_report, encoding="utf-8")
    tq.write_csv(root / "cross_bit.csv", cross_bit)
    report += "\n" + cross_report.replace("# Cross-bit parity:", "## Cross-bit parity:", 1)
    report += "\n## Reproducibility and baseline audit\n\n" + "\n\n".join(audit) + "\n"
    (root / "summary_ext.md").write_text(report, encoding="utf-8")
    tq.write_csv(root / "runs.csv", rows)
    tq.write_csv(root / "needles.csv", needles)
    tq.write_csv(root / "paired.csv", paired)
    (root / "audit.json").write_text(json.dumps(dict(rows=len(rows), needle_rows=len(needles),
                                                    paired_tests=len(paired), cross_bit_tests=len(cross_bit), baseline_audit=audit,
                                                    analysis_script_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                                                            for name in ("summarize_tq_cdf_extension.py", "tq_cross_bit_parity.py")},
                                                    cross_bit_policy=dict(pairs=[[5, 4], [4, 3]], alpha=0.05,
                                                                          paired_difference_ci=0.95, tost_ci=0.90,
                                                                          recall_fpr_margin=1/16, rho_margin=0.001,
                                                                          nmse_equivalence_factor=1.10,
                                                                          multiplicity_correction="none; exploratory"),
                                                    source_metadata=metadata), indent=2), encoding="utf-8")
    print(f"Wrote {root / 'summary_ext.md'}: {len(rows)} runs, {len(needles)} needle records, {len(paired)} paired comparisons")


if __name__ == "__main__":
    main()
