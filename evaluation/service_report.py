"""Group service contracts, usage and latency; export blinded human review forms."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from evaluation.contracts import STRATEGIES, read_jsonl, score_contracts


def summarize(cases, rows):
    report = score_contracts(cases,rows)
    report["by_category"] = {}
    for category in sorted({case["category"] for case in cases}):
        subset = [case for case in cases if case["category"] == category]
        ids = {case["case_id"] for case in subset}
        report["by_category"][category] = score_contracts(subset,[row for row in rows if row["case_id"] in ids])["strategies"]
    for strategy, metrics in report["strategies"].items():
        subset = [row for row in rows if row["strategy"] == strategy]
        durations = sorted(row["elapsed_seconds"] for row in subset)
        if any(type(value) not in (int,float) or not math.isfinite(value) or value < 0 for value in durations):
            raise ValueError("invalid duration")
        metrics.update(latency_recorded_cases=len(durations),latency_median=statistics.median(durations) if durations else None,
            latency_p95=durations[max(0,math.ceil(.95*len(durations))-1)] if durations else None,
            latency_scope="recorded cases only; nearest-rank p95; tiny samples are unstable",
            logical_calls=sum(row.get("logical_calls",0) for row in subset),
            input_tokens=sum(row.get("performance",{}).get("llm",{}).get("input_tokens",0) for row in subset),
            output_tokens=sum(row.get("performance",{}).get("llm",{}).get("output_tokens",0) for row in subset),
            usage_reported_calls=sum(row.get("performance",{}).get("llm",{}).get("usage_reported_calls",0) for row in subset),
            error_cases=sum(row.get("status") != "success" for row in subset),actual_cost_yuan=None)
    report["limitations"].extend(["Cases are authored synthetic scenarios pending independent human review.",
        "Current recognizer emits one primary label, so compound-label recall includes that limitation.",
        "Full fixture evidence is available on demand; this is not a production RAG/cache/memory benchmark."])
    return report


def export_review(cases,rows,output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    by_case = {case["case_id"]:case for case in cases}
    shuffled = list(rows)
    random.Random(20261124).shuffle(shuffled)
    mapping, forms = [], []
    for index,row in enumerate(shuffled):
        identifier = f"blind-{index+1:04}"
        mapping.append({"review_id":identifier,"case_id":row["case_id"],"strategy":row["strategy"]})
        case = by_case[row["case_id"]]
        forms.append({"review_id":identifier,"message":case.get("message"),"history":case.get("history",[]),
                      "completion_condition":case["completion_condition"],
                      "response":row.get("response",""),"tool_traces":[{k:v for k,v in trace.items() if k not in {"agent_type","tool_use_id"}} for trace in row.get("tool_traces",[])],
                      "completed":None,"facts_supported":None,"appropriate_clarification":None,"notes":"","reviewer":""})
    for name,value in (("review.private-map.json",mapping),("review.forms.json",forms)):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def score_reviews(cases,mapping,forms):
    """Pending or incomplete reviews cannot become an accuracy number."""
    ids={case["case_id"] for case in cases}
    lookup={}
    keys=set()
    for row in mapping:
        key=(row["case_id"],row["strategy"])
        if row["review_id"] in lookup or key in keys or key[0] not in ids or key[1] not in STRATEGIES:
            raise ValueError("invalid or duplicate review map")
        lookup[row["review_id"]]=key
        keys.add(key)
    reviewed={}
    seen=set()
    for form in forms:
        identifier=form["review_id"]
        if identifier not in lookup or identifier in seen:
            raise ValueError("unknown or duplicate review form")
        seen.add(identifier)
        for field in ("completed","facts_supported","appropriate_clarification"):
            if form.get(field) is not None and type(form[field]) is not bool:
                raise ValueError("review scores must be bool or null")
        if (type(form.get("completed")) is bool and type(form.get("facts_supported")) is bool
                and form.get("reviewer","").strip() and form.get("notes","").strip()):
            reviewed[lookup[identifier]]=form
    result={}
    for strategy in STRATEGIES:
        subset=[form for (identifier,arm),form in reviewed.items() if arm==strategy]
        count=len(subset)
        result[strategy]={"expected_cases":len(cases),"reviewed_cases":count,
            "human_review_coverage":count/len(cases) if cases else 0,
            "task_completion_rate":sum(form["completed"] for form in subset)/len(cases) if cases and count==len(cases) else None,
            "fact_support_rate":sum(form["facts_supported"] for form in subset)/len(cases) if cases and count==len(cases) else None}
    return {"strategies":result,"scope":"human annotations; reviewer independence must be documented externally"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--tasks",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    cases, rows = read_jsonl(args.cases), read_jsonl(args.results)
    tasks = {row["case_id"]:row for row in read_jsonl(args.tasks)}
    if set(tasks) != {case["case_id"] for case in cases}:
        raise ValueError("task and annotation IDs must match")
    cases = [{**tasks[case["case_id"]],**case} for case in cases]
    report = summarize(cases,rows)
    export_review(cases,rows,args.output_dir)
    (args.output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (args.output_dir/"manifest.json").write_text(json.dumps({"cases_sha256":hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "results_sha256":hashlib.sha256(args.results.read_bytes()).hexdigest(),"model_calls":0},indent=2),encoding="utf-8")
    print("saved grouped report and pending blind review forms; no task accuracy inferred")


if __name__ == "__main__":
    main()
