"""Controlled service batch: fresh per-case state, fixed local evidence, no memory service."""
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from agents.agent_orchestrator import Request
from core.performance import collect_performance, performance_snapshot
from evaluation.compatible_client import CompatibleClient
from evaluation.protocol import SharedLLMBudget, apply_controlled_protocol
from evaluation.strategies import EvaluationOrchestrator

ACTIVE_CASE = ContextVar("service_case", default=None)


class FixtureKnowledge:
    """Same small evidence corpus for all arms, not production retrieval efficacy."""
    def __init__(self, documents, fault=None):
        self.documents, self.fault = documents, fault

    async def search_with_rewrite(self, name, query, top_k=5):
        if self.fault == "timeout":
            raise TimeoutError("injected fixture retrieval timeout")
        # Fixed full corpus avoids a hidden retrieval-model or ranking variable.
        return SimpleNamespace(success=True, data=self.documents, cached=False, reranked=False)


def actor_fields(case):
    return {key: case.get(key, default) for key, default in (
        ("message", ""), ("history", []), ("context", ""))}


async def execute_case(case, strategy, model, sender, documents, *, max_calls=12, token_budget=200000):
    # Constructor clients receive a dummy key and are replaced before use.
    # Embeddings and production memory/cache are explicitly outside this protocol.
    if os.environ.get("INTENT_EMBEDDING_ENABLED", "").lower() != "false":
        raise ValueError("set INTENT_EMBEDDING_ENABLED=false for this controlled protocol")
    orch = EvaluationOrchestrator(api_key="offline-constructor-placeholder", model=model, strategy=strategy,
        rag_tool_manager=FixtureKnowledge(documents, case.get("rag_fault")))
    originals = [orch._intent_recognizer.client, orch._composer._client]
    client = CompatibleClient(sender)
    for pool in orch._pool.values():
        for agent in pool:
            originals.append(agent._client)
            agent._client = client
    orch._composer._client = orch._intent_recognizer.client = client
    budget = SharedLLMBudget(model, max_calls=max_calls, token_budget=token_budget)
    apply_controlled_protocol(orch, budget)
    req = Request(**actor_fields(case), user_id="synthetic-eval", conv_id=case["case_id"])
    started = time.monotonic()
    row = {"case_id":case["case_id"],"strategy":strategy,"category":case["category"]}
    active_token = ACTIVE_CASE.set(case["case_id"]+":"+strategy)
    try:
        with collect_performance():
            result = await orch.run(req)
            row.update(status=result.status,response=result.response,tool_traces=result.tool_traces,
                       predicted_intents=[req.intent.value] if req.intent else [],
                       primary_agent=result.primary_agent.value if result.primary_agent else None,
                       supporting_agents=[agent.value for agent in result.supporting_agents],
                       performance=performance_snapshot())
    except Exception as exc:
        # No exception message: providers may include request details in it.
        row.update(status="error",error_type=type(exc).__name__,response="",tool_traces=[],
                   predicted_intents=[req.intent.value] if req.intent else [])
    finally:
        ACTIVE_CASE.reset(active_token)
        for original in {id(item):item for item in originals}.values():
            if hasattr(original,"close"):
                await original.close()
    row.update(elapsed_seconds=time.monotonic()-started, logical_calls=budget.calls,
               reserved_tokens=budget.reserved_tokens, task_completion=None)
    return row


def append_row(path, row):
    with Path(path).open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


async def run_batch(cases, model, sender, documents, output, *, max_calls=12, token_budget=200000):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    manifest = {"protocol":"service-batch-v1", "model":model,"temperature":0,"max_output":512,
                "max_calls":max_calls,"per_task_reservation":token_budget,"case_order":[c["case_id"] for c in cases],
                "knowledge_sha256":hashlib.sha256(json.dumps(documents,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                "state":"fresh orchestrator per case/strategy; no remote embedding, memory or cache",
                "intent_scope":"single primary label from current recognizer; secondary intent recall is not implemented",
                "knowledge_mode":"all fixed fixture documents when requested; not a production RAG benchmark"}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    for index, case in enumerate(cases):
        strategies = ("single","rule","multi")
        # Rotate execution order to reduce systematic first-arm/time effects.
        order = strategies[index%3:] + strategies[:index%3]
        for strategy in order:
            row = await execute_case(case,strategy,model,sender,documents,max_calls=max_calls,token_budget=token_budget)
            append_row(output/"results.jsonl",row)
    return output
