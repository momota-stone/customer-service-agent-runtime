"""Offline cache microbenchmark using real manager and deterministic simulated I/O."""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tooling.tool_manager import MCPToolManager, Tool


async def measure(*, enabled, repeat_rate, warm, concurrency, count=40, delay=.01):
    manager = MCPToolManager(api_key="offline-placeholder")
    executions = 0
    async def handler(params, context):
        nonlocal executions
        executions += 1
        await asyncio.sleep(delay)
        return [{"document_id":context["user_id"]+":"+params["query"],"content":context["user_id"]+":"+params["query"]}]
    manager.register(Tool("fixture","synthetic read-only fixture",handler,{},cache_ttl=300 if enabled else 0))
    async def rewrite(query,n=3):
        return [query]
    manager.rewrite_query = rewrite
    unique = max(1,round(count*(1-repeat_rate)))
    # Repeat identical user/query keys; alternating users remain isolated.
    requests = [(f"q-{i%unique}",{"user_id":f"u-{(i%unique)%2}"}) for i in range(count)]
    if warm:
        for query,context in requests[:unique]:
            await manager.search_with_rewrite("fixture",query,top_k=1,context=context)
    warmup_executions = executions
    semaphore = asyncio.Semaphore(concurrency)
    async def one(query,context):
        async with semaphore:
            start=time.monotonic()
            result=await manager.search_with_rewrite("fixture",query,top_k=1,context=context)
            expected=[{"document_id":context["user_id"]+":"+query,"content":context["user_id"]+":"+query}]
            return {"query":query,"user_id":context["user_id"],"elapsed_seconds":time.monotonic()-start,
                    "cached":result.cached,"success":result.success,"exact_output":result.data==expected}
    started=time.monotonic()
    rows=await asyncio.gather(*(one(*request) for request in requests))
    elapsed=time.monotonic()-started
    await manager._client.close()
    durations=sorted(row["elapsed_seconds"] for row in rows)
    return {"enabled":enabled,"repeat_rate":repeat_rate,"warm":warm,"concurrency":concurrency,
            "requests":count,"handler_calls":executions-warmup_executions,"warmup_handler_calls":warmup_executions,
            "exact_outputs":sum(row["exact_output"] for row in rows),"errors":sum(not row["success"] for row in rows),
            "cache_hits":sum(row["cached"] for row in rows),"elapsed_seconds":elapsed,
            "median_seconds":statistics.median(durations),"p95_seconds":durations[math.ceil(.95*len(durations))-1],"raw":rows}


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    for rate in (0,.5,.9):
        for concurrency in (1,4):
            for warm in (False,True):
                for enabled in (False,True):
                    rows.append(await measure(enabled=enabled,repeat_rate=rate,warm=warm,concurrency=concurrency))
    report={"scope":"deterministic 10ms simulated read-only I/O; real pipeline cache and locks; no LLM, Redis or Chroma",
            "model_calls":0,"cells":rows,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manager_sha256":hashlib.sha256((Path(__file__).resolve().parents[1]/"tooling/tool_manager.py").read_bytes()).hexdigest(),
            "limitations":["Output equality covers synthetic tool payloads, not language answer quality", "Warmup excluded from timed workload and reported separately", "No production throughput, model-token savings or remote RAG conclusions", "Cache context includes user identity; registered fixture is read-only"]}
    (args.output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"{len(rows)} cells; {sum(row['requests'] for row in rows)} measured requests; no model calls")


if __name__=="__main__":
    asyncio.run(main())
