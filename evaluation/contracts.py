"""Offline intent/tool-contract scoring. These metrics are not task accuracy."""
import argparse
import hashlib
import json
import math
from pathlib import Path


STRATEGIES = ("single", "rule", "multi")


def contains(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and contains(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(contains(a, b) for a, b in zip(actual, expected))
    if type(expected) in (int, float):
        return type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected) and actual == expected
    return type(actual) is type(expected) and actual == expected


def matched_tool_requirements(requirements, traces):
    """Maximum one-to-one matching: one call cannot satisfy two requirements."""
    available = [row for row in traces if row.get("success") is True and row.get("result_success") is not False and not row.get("error")]
    edges = [[index for index, row in enumerate(available)
              if row.get("tool_name") == required["tool_name"] and contains(row.get("input"), required.get("input", {}))]
             for required in requirements]
    assigned = {}
    def augment(index, seen):
        for target in edges[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or augment(assigned[target], seen):
                assigned[target] = index
                return True
        return False
    return sum(augment(index, set()) for index in range(len(requirements)))


def score_contracts(cases, results):
    by_case = {}
    for case in cases:
        identifier = case.get("case_id")
        if not isinstance(identifier, str) or not identifier or identifier in by_case:
            raise ValueError("case_id must be a unique nonempty string")
        if not isinstance(case.get("expected_intents"), list) or any(not isinstance(label, str) for label in case["expected_intents"]):
            raise ValueError("expected_intents must be an explicit list of labels")
        by_case[identifier] = case
    indexed = {}
    for row in results:
        key = (row.get("strategy"), row.get("case_id"))
        if key[0] not in STRATEGIES or key[1] not in by_case or key in indexed:
            raise ValueError("unknown strategy/case or duplicate result; do not overwrite attempts")
        labels = row.get("predicted_intents", [])
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ValueError("predicted_intents must be a list; do not infer intent labels from agent names")
        indexed[key] = row
    summaries = {}
    for strategy in STRATEGIES:
        tp = fp = fn = exact = passed = expected_calls = matched = 0
        tool_cases = 0
        missing = []
        for identifier, case in by_case.items():
            row = indexed.get((strategy, identifier))
            if row is None:
                missing.append(identifier)
            expected = set(case["expected_intents"])
            predicted = set(row.get("predicted_intents", [])) if row else set()
            tp += len(expected & predicted)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            exact += row is not None and predicted == expected
            requirements = case.get("required_tools", [])
            expected_calls += len(requirements)
            if requirements:
                tool_cases += 1
                count = matched_tool_requirements(requirements, row.get("tool_traces", [])) if row else 0
                matched += count
                passed += row is not None and row.get("status") == "success" and count == len(requirements)
        summaries[strategy] = {
            "expected_cases": len(cases), "recorded_cases": len(cases) - len(missing), "missing_case_ids": missing,
            "intent_micro_precision": tp / (tp + fp) if tp + fp else 0,
            "intent_micro_recall": tp / (tp + fn) if tp + fn else 0,
            "intent_micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "intent_exact_match_rate": exact / len(cases) if cases else 0,
            "tool_contract_cases": tool_cases, "tool_contract_passed": passed,
            "tool_contract_pass_rate": passed / tool_cases if tool_cases else None,
            "required_tool_calls": expected_calls, "matched_required_calls": matched,
            "task_completion_rate": None,
        }
    return {"scope": "offline intent/tool-contract metrics; not end-to-end task accuracy",
            "strategies": summaries,
            "limitations": ["Missing cases stay in the denominator; duplicate attempts are rejected.",
                            "Tool input annotations are subset requirements; successful transport with result_success=false does not pass.",
                            "Intent label sets must come from the evaluated recognizer, not be invented from routed agent names.",
                            "Final answer facts, clarification quality and task completion require separate verification."]}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = score_contracts(read_jsonl(args.cases), read_jsonl(args.results))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "contracts.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"schema_version": "echomind-contract-report-v1", "model_calls": 0,
                "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
                "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "metric_scope": report["scope"]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("offline contract report saved; no model calls made")


if __name__ == "__main__":
    main()
