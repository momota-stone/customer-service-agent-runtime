import asyncio
import json

import pytest

from evaluation.compatible_client import chat_payload, message_response
from evaluation.service_batch import actor_fields, run_batch
from evaluation.service_dataset import build_dataset


def test_adapter_round_trip_tool_use_and_length():
    response = message_response({"choices":[{"message":{"tool_calls":[{"id":"x","function":{"name":"compare_amounts","arguments":"{\"amount_a\":119,\"amount_b\":99}"}}]},"finish_reason":"length"}]})
    payload = chat_payload({"model":"fixture","max_tokens":64,"messages":[
        {"role":"assistant","content":response.content},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":"{\"difference\":20}"}]}]})
    assert payload["messages"][0]["tool_calls"][0]["id"] == "x"
    assert payload["messages"][1]["role"] == "tool"
    assert payload["enable_thinking"] is False
    assert response.stop_reason == "max_tokens"
    with pytest.raises(ValueError):
        message_response({"choices":[{"message":{"tool_calls":[{"id":"x","function":{"name":"bad","arguments":"[]"}}]}}]})


def test_dataset_separates_labels_and_does_not_claim_human_review(tmp_path):
    manifest = build_dataset(tmp_path/"suite")
    assert manifest["cases"] == 16
    tasks = [json.loads(line) for line in (tmp_path/"suite/dev.tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 8
    assert "expected_intents" not in tasks[0]
    assert set(actor_fields({**tasks[0],"expected_intents":["secret"],"completion_condition":"secret"})) == {"message","history","context"}


def test_batch_runs_all_arms_without_injecting_gold_and_persists_failures(tmp_path,monkeypatch):
    monkeypatch.setenv("INTENT_EMBEDDING_ENABLED","false")
    calls = []
    async def sender(payload):
        calls.append(payload)
        raise TimeoutError("private credential should not be saved")
    cases = [{"case_id":"a","category":"clarification","message":"那个", "expected_intents":["gold-secret"]}]
    asyncio.run(run_batch(cases,"fixture",sender,[],tmp_path/"run",max_calls=1))
    rows = (tmp_path/"run/results.jsonl").read_text(encoding="utf-8")
    assert len(rows.splitlines()) == 3
    assert "private credential" not in rows
    assert all("gold-secret" not in json.dumps(payload) for payload in calls)


def test_real_orchestrator_tool_roundtrip_through_compatible_adapter(monkeypatch):
    from evaluation.service_batch import execute_case
    monkeypatch.setenv("INTENT_EMBEDDING_ENABLED","false")
    async def sender(payload):
        if not payload.get("tools"):
            message = {"content":'{"intent":"technical_login","confidence":1,"reasoning":"401"}'}
        elif not any(row["role"] == "tool" for row in payload["messages"]):
            message = {"tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup_error_code","arguments":'{"error_code":"401"}'}}]}
        else:
            message = {"content":"401表示认证失败，请核对凭据。"}
        return {"choices":[{"message":message,"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":20}}
    case={"case_id":"roundtrip","category":"single","message":"登录401请解释"}
    result=asyncio.run(execute_case(case,"single","fixture",sender,[]))
    assert result["status"] == "success"
    assert result["tool_traces"][0]["tool_name"] == "lookup_error_code"
    assert result["tool_traces"][0]["success"] is True
    assert result["logical_calls"] == 3


def test_cache_workload_preserves_payload_and_user_isolation():
    from evaluation.cache_benchmark import measure
    cold=asyncio.run(measure(enabled=False,repeat_rate=.9,warm=False,concurrency=4,count=20,delay=0))
    cached=asyncio.run(measure(enabled=True,repeat_rate=.9,warm=False,concurrency=4,count=20,delay=0))
    assert cold["exact_outputs"] == cached["exact_outputs"] == 20
    assert cold["handler_calls"] == 20
    assert cached["handler_calls"] == 2


def test_service_report_rejects_duplicates_and_does_not_invent_completion():
    from evaluation.service_report import summarize
    cases=[{"case_id":"a","category":"single","expected_intents":["billing"],"required_tools":[]}]
    rows=[{"case_id":"a","strategy":"single","predicted_intents":["billing"],"elapsed_seconds":2}]
    report=summarize(cases,rows)
    assert report["strategies"]["single"]["task_completion_rate"] is None
    assert report["strategies"]["single"]["latency_p95"] == 2
    with pytest.raises(ValueError):
        summarize(cases,rows*2)


def test_pending_or_unsigned_reviews_cannot_produce_accuracy():
    from evaluation.service_report import score_reviews
    cases=[{"case_id":"a"},{"case_id":"b"}]
    mapping=[{"case_id":"a","strategy":"single","review_id":"r"}]
    forms=[{"review_id":"r","completed":True,"facts_supported":True,"reviewer":"","notes":""}]
    report=score_reviews(cases,mapping,forms)["strategies"]["single"]
    assert report["reviewed_cases"] == 0
    forms[0].update(reviewer="human",notes="checked evidence")
    report=score_reviews(cases,mapping,forms)["strategies"]["single"]
    assert report["reviewed_cases"] == 1
    assert report["task_completion_rate"] is None
    with pytest.raises(ValueError):
        score_reviews(cases,mapping,forms*2)


def test_dataset_audit_rejects_tampered_inputs(tmp_path):
    from evaluation.service_dataset import audit_dataset
    path=tmp_path/"suite"
    build_dataset(path)
    report=audit_dataset(path)
    assert report["splits"]["dev"]["cases"] == 8
    assert report["splits"]["holdout"]["human_reviews_pending"] == 8
    with (path/"dev.tasks.jsonl").open("a",encoding="utf-8") as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError,match="hash mismatch"):
        audit_dataset(path)
