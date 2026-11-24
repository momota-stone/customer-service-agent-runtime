"""Common per-task LLM admission budgets for the three evaluation strategies."""
from dataclasses import replace
import json


class EvaluationBudgetExceeded(RuntimeError):
    pass


class SharedLLMBudget:
    """Reserve before awaiting, so parallel agents share one common limit.

    UTF-8 serialized bytes + 8192 estimate input conservatively. This is not a
    provider-tokenizer count or a monetary budget. Failed calls retain their
    reservation. Construct a fresh object for each task/strategy pair.
    """
    def __init__(self, model, max_calls=12, token_budget=200000, max_output=512):
        if min(max_calls, token_budget, max_output) <= 0:
            raise ValueError("budget limits must be positive")
        self.model = model
        self.max_calls, self.token_budget, self.max_output = max_calls, token_budget, max_output
        self.calls = 0
        self.reserved_tokens = 0

    def reserve(self, kwargs):
        options = {**kwargs, "model": self.model, "temperature": 0,
                   "max_tokens": min(kwargs.get("max_tokens", self.max_output), self.max_output)}
        amount = len(json.dumps(options, ensure_ascii=False).encode("utf-8")) + 8192 + options["max_tokens"]
        if self.calls >= self.max_calls or self.reserved_tokens + amount > self.token_budget:
            raise EvaluationBudgetExceeded("shared per-task LLM admission budget exhausted")
        self.calls += 1
        self.reserved_tokens += amount
        return options


class BudgetedClient:
    def __init__(self, client, budget):
        # Rebinding an existing orchestrator starts a fresh task budget rather
        # than reserving against every previously applied wrapper.
        while isinstance(client, BudgetedClient):
            client = client.client
        # Disable SDK-hidden retries so one reserved attempt means one SDK call.
        self.client = client.with_options(max_retries=0) if hasattr(client, "with_options") else client
        self.budget = budget
        self.messages = self

    @property
    def base_url(self):
        return getattr(self.client, "base_url", "")

    async def create(self, **kwargs):
        options = self.budget.reserve(kwargs)
        return await self.client.messages.create(**options)


def apply_controlled_protocol(orchestrator, budget):
    """Explicitly opt an evaluation instance into common LLM controls.

    Does not control remote embedding, RAG providers, cache or memory state.
    Those must be fixed by the experiment harness before comparing results.
    """
    clients = {}
    def wrap(client):
        key = id(client)
        if key not in clients:
            clients[key] = BudgetedClient(client, budget)
        return clients[key]
    for pool in orchestrator._pool.values():
        for agent in pool:
            agent._client = wrap(agent._client)
            agent._model = budget.model
            agent.profile = replace(agent.profile, model=budget.model, temperature=0, max_tokens=budget.max_output)
    orchestrator._composer._client = wrap(orchestrator._composer._client)
    orchestrator._composer._model = budget.model
    orchestrator._intent_recognizer.client = wrap(orchestrator._intent_recognizer.client)
    orchestrator._intent_recognizer.model = budget.model
    return orchestrator
