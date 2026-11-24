import asyncio
from types import SimpleNamespace

import pytest

from evaluation.protocol import BudgetedClient, EvaluationBudgetExceeded, SharedLLMBudget, apply_controlled_protocol
from evaluation.strategies import EvaluationOrchestrator


def test_parallel_agents_cannot_each_take_the_full_task_budget():
    async def run():
        calls = []
        async def create(**kwargs):
            calls.append(kwargs)
            await asyncio.sleep(0)
            return "ok"
        budget = SharedLLMBudget("fixed", max_calls=2)
        client = BudgetedClient(SimpleNamespace(messages=SimpleNamespace(create=create)), budget)
        result = await asyncio.gather(*(client.messages.create(model="other", max_tokens=2000, temperature=1) for _ in range(4)), return_exceptions=True)
        assert len(calls) == 2
        assert sum(isinstance(row, EvaluationBudgetExceeded) for row in result) == 2
        assert all(row["model"] == "fixed" and row["temperature"] == 0 and row["max_tokens"] == 512 for row in calls)
    asyncio.run(run())


def test_oversized_input_is_blocked_before_call():
    budget = SharedLLMBudget("fixed", token_budget=10)
    with pytest.raises(EvaluationBudgetExceeded):
        budget.reserve({"messages": [{"role": "user", "content": "hello"}]})
    assert budget.calls == 0


def test_rebinding_client_does_not_consume_exhausted_previous_budget():
    async def run():
        calls = []
        async def create(**kwargs):
            calls.append(kwargs)
            return "ok"
        original = SimpleNamespace(messages=SimpleNamespace(create=create))
        old = SharedLLMBudget("old", max_calls=1)
        wrapped = BudgetedClient(original, old)
        await wrapped.messages.create(messages=[])
        fresh = SharedLLMBudget("fresh", max_calls=1)
        rebound = BudgetedClient(wrapped, fresh)
        await rebound.messages.create(messages=[])
        assert old.calls == 1
        assert fresh.calls == 1
        assert [row["model"] for row in calls] == ["old", "fresh"]
    asyncio.run(run())


def test_protocol_overrides_role_models_for_every_strategy():
    for strategy in ("single", "rule", "multi"):
        orch = EvaluationOrchestrator(api_key="offline-fixture", strategy=strategy)
        budget = SharedLLMBudget("frozen-model")
        apply_controlled_protocol(orch, budget)
        assert orch._composer._client.budget is budget
        assert orch._intent_recognizer.client.budget is budget
        for pool in orch._pool.values():
            for agent in pool:
                assert agent._client.budget is budget
                assert agent._model == "frozen-model"
                assert agent.profile.max_tokens == 512
