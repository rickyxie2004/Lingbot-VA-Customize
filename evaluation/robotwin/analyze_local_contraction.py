#!/usr/bin/env python3
"""Summarize local-contraction samples collected during RoboTwin rollout."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _finite_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _rate(values):
    values = [bool(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _mean(values):
    values = [value for value in (_finite_float(v) for v in values)
              if value is not None]
    return float(np.mean(values)) if values else None


def _correlation(rows, x_key, y_key):
    pairs = []
    for row in rows:
        x_value = _finite_float(row.get(x_key))
        y_value = _finite_float(row.get(y_key))
        if x_value is not None and y_value is not None:
            pairs.append((x_value, y_value))
    if len(pairs) < 2:
        return None
    x_values, y_values = np.asarray(pairs, dtype=np.float64).T
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _sample_group_summary(rows):
    q_values = np.asarray(
        [float(row["q"]) for row in rows if _finite_float(row.get("q")) is not None],
        dtype=np.float64,
    )
    if q_values.size == 0:
        return {"count": len(rows)}
    return {
        "count": len(rows),
        "contractive_ratio": _rate(row.get("contractive") for row in rows),
        "mean_q": float(np.mean(q_values)),
        "median_q": float(np.median(q_values)),
        "p90_q": float(np.percentile(q_values, 90)),
        "max_q": float(np.max(q_values)),
        "mean_Lx": _mean(row.get("Lx") for row in rows),
        "mean_verifier_mse": _mean(row.get("verifier_mse") for row in rows),
        "verifier_accept_rate": _rate(
            row.get("verifier_accept") for row in rows),
        "episode_success_rate": _rate(
            row.get("episode_success") for row in rows),
        "mean_oracle_action_mse": _mean(
            row.get("oracle_action_mse") for row in rows),
    }


def _load_records(input_path):
    samples = []
    episode_results = {}
    invalid_lines = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_lines.append({"line": line_number, "error": str(exc)})
                continue
            record_type = record.get("record_type", "sample")
            if record_type == "sample":
                samples.append(record)
            elif record_type == "episode_end":
                key = (str(record.get("task", "unknown")),
                       int(record.get("episode_id", -1)))
                episode_results[key] = bool(record.get("episode_success", False))

    for sample in samples:
        key = (str(sample.get("task", "unknown")),
               int(sample.get("episode_id", -1)))
        if key in episode_results:
            sample["episode_success"] = episode_results[key]
    return samples, episode_results, invalid_lines


def _write_csv(rows, output_path):
    preferred_fields = [
        "sample_id", "episode_id", "step_id", "task", "phase",
        "segment_index", "verify_frame_st_id", "requested_t", "t",
        "model_timestep", "verifier_mse", "Lx", "q", "contractive",
        "oracle_action_mse", "verifier_accept", "oracle_accept",
        "replan_triggered", "episode_success", "estimation_method",
        "power_iterations", "method_error", "tensor_path", "action_shape",
        "condition_shape",
    ]
    extra_fields = sorted({
        key for row in rows for key in row
        if key not in preferred_fields and key != "record_type"
    })
    fieldnames = preferred_fields + extra_fields
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            for key, value in serializable.items():
                if isinstance(value, (dict, list)):
                    serializable[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(serializable)


def build_summary(samples, episode_results, invalid_lines):
    if not samples:
        return {
            "sample_count": 0,
            "episode_count": len(episode_results),
            "invalid_jsonl_lines": invalid_lines,
        }

    by_t = defaultdict(list)
    for sample in samples:
        actual_t = _finite_float(sample.get("t"))
        key = f"{actual_t:.6f}" if actual_t is not None else "unknown"
        by_t[key].append(sample)

    verifier_mse_values = [
        value for value in (
            _finite_float(sample.get("verifier_mse")) for sample in samples)
        if value is not None
    ]
    low_mse_threshold = (
        float(np.percentile(verifier_mse_values, 25))
        if verifier_mse_values else None
    )
    low_mse_rows = []
    if low_mse_threshold is not None:
        low_mse_rows = [
            sample for sample in samples
            if _finite_float(sample.get("verifier_mse")) is not None
            and float(sample["verifier_mse"]) <= low_mse_threshold
        ]
    low_mse_noncontractive = [
        sample for sample in low_mse_rows
        if not bool(sample.get("contractive", False))
    ]

    contractive_rows = [
        sample for sample in samples if bool(sample.get("contractive", False))
    ]
    noncontractive_rows = [
        sample for sample in samples if not bool(sample.get("contractive", False))
    ]
    oracle_rows = [
        sample for sample in samples
        if _finite_float(sample.get("oracle_action_mse")) is not None
    ]

    summary = {
        "sample_count": len(samples),
        "episode_count": len(episode_results),
        "samples_with_episode_success": sum(
            sample.get("episode_success") is not None for sample in samples),
        "overall": _sample_group_summary(samples),
        "by_t": {
            key: _sample_group_summary(rows)
            for key, rows in sorted(by_t.items())
        },
        "relations": {
            "pearson_q_vs_verifier_mse": _correlation(
                samples, "q", "verifier_mse"),
            "pearson_q_vs_oracle_action_mse": _correlation(
                oracle_rows, "q", "oracle_action_mse"),
            "contractive_sample_episode_success_rate": _rate(
                sample.get("episode_success") for sample in contractive_rows),
            "noncontractive_sample_episode_success_rate": _rate(
                sample.get("episode_success") for sample in noncontractive_rows),
            "contractive_mean_verifier_mse": _mean(
                sample.get("verifier_mse") for sample in contractive_rows),
            "noncontractive_mean_verifier_mse": _mean(
                sample.get("verifier_mse") for sample in noncontractive_rows),
            "low_mse_definition": "verifier_mse <= sample P25",
            "low_mse_threshold": low_mse_threshold,
            "low_mse_count": len(low_mse_rows),
            "low_mse_noncontractive_count": len(low_mse_noncontractive),
            "low_mse_noncontractive_ratio": (
                len(low_mse_noncontractive) / len(low_mse_rows)
                if low_mse_rows else None
            ),
        },
        "estimation_methods": dict(Counter(
            str(sample.get("estimation_method", "unknown"))
            for sample in samples
        )),
        "estimation_error_count": sum(
            bool(sample.get("method_error")) for sample in samples),
        "invalid_jsonl_lines": invalid_lines,
        "notes": [
            "Episode-success comparisons are sample-level and include correlated samples from the same episode.",
            "The CSV columns q/verifier_mse/oracle_action_mse are the scatter-plot source data.",
        ],
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("local_contraction_analysis.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_contraction_summary"),
    )
    args = parser.parse_args()

    samples, episode_results, invalid_lines = _load_records(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "local_contraction_samples.csv"
    summary_path = args.output_dir / "local_contraction_summary.json"
    _write_csv(samples, csv_path)
    summary = build_summary(samples, episode_results, invalid_lines)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    overall = summary.get("overall", {})
    print(f"samples: {summary.get('sample_count', 0)}")
    print(f"episodes: {summary.get('episode_count', 0)}")
    if overall:
        print(f"contractive ratio: {overall.get('contractive_ratio')}")
        print(
            "q: "
            f"mean={overall.get('mean_q')} "
            f"median={overall.get('median_q')} "
            f"P90={overall.get('p90_q')} "
            f"max={overall.get('max_q')}"
        )
    for t_value, t_summary in summary.get("by_t", {}).items():
        print(
            f"t={t_value}: N={t_summary.get('count')} "
            f"R_contract={t_summary.get('contractive_ratio')} "
            f"mean_q={t_summary.get('mean_q')}"
        )
    print(f"CSV: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
