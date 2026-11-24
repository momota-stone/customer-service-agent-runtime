import asyncio

import pytest

from agents.agent_orchestrator import AgentResponse, AgentType
from evaluation.strategies import EvaluationOrchestrator
from test_agent_orchestrator import make_request


@pytest.mark.parametrize("strategy,primary,support", [
    ("single", AgentType.GENERAL, []),
    ("rule", AgentType.TECHNICAL, []),
    ("multi", AgentType.TECHNICAL, [AgentType.BILLING]),
])
def test_composite_request_executes_selected_strategy(strategy, primary, support):
    async def run():
        orch = EvaluationOrchestrator(api_key="offline-fixture", model="fixture", strategy=strategy)
        calls = []

        async def execute(req, agent_type):
            calls.append(agent_type)
            return AgentResponse(agent_type, "fixture answer", True)

        async def compose(req, responses):
            from agents.agent_orchestrator import AgentCallResult
            return AgentCallResult("combined fixture answer")

        orch._execute = execute
        orch._composer.compose_result = compose
        result = await orch.run(make_request())
        assert result.primary_agent == primary
        assert result.supporting_agents == support
        assert set(calls) == {primary, *support}
        assert result.status == "success"

    asyncio.run(run())


def test_variants_receive_same_local_business_tools():
    scopes = []
    for strategy in ("single", "rule", "multi"):
        orch = EvaluationOrchestrator(api_key="offline-fixture", model="fixture", strategy=strategy)
        for agents in orch._pool.values():
            scopes.extend(set(agent.get_tools()) for agent in agents)
    assert all(scope == scopes[0] for scope in scopes)
    assert {"lookup_error_code", "compare_amounts", "create_handoff_summary"} <= scopes[0]


def test_unknown_strategy_rejected_before_client_creation():
    with pytest.raises(ValueError, match="strategy"):
        EvaluationOrchestrator(api_key="offline-fixture", strategy="unknown")
