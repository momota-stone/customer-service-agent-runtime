"""Evaluation-only routing variants; production orchestration is unchanged.

All variants share the existing intent recognizer. ``rule`` means static intent
to business-agent routing, not a rule-only intent classifier. The same local
business tools are exposed to every variant to avoid weakening the baseline.
"""

from dataclasses import replace

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentType, GeneralAgent, RoutingDecision,
)
from agents.tools import billing_tools, escalation_tools, general_tools, technical_tools


class UnifiedServiceAgent(GeneralAgent):
    profile = replace(
        GeneralAgent.profile,
        role="综合客服",
        mission="独立完成技术、账单和通用咨询的全部诉求",
        workflow=("拆分全部诉求", "逐项调用工具核验", "回答或澄清", "给出下一步"),
        max_tokens=1200,
    )
    system_prompt = (
        "你是综合客服，独立处理技术、账单、通用咨询和复合诉求。"
        "使用提供的全部工具核验证据，逐项解决用户需求；信息不足先澄清。"
        "不得声称执行未接入的退款、订单查询或工单操作。"
    )


class EvaluationOrchestrator(AgentOrchestrator):
    """Switch routing without changing tools, recognizer or production defaults."""

    def __init__(self, *args, strategy="multi", **kwargs):
        if strategy not in {"single", "rule", "multi"}:
            raise ValueError("strategy must be single, rule or multi")
        self.strategy = strategy
        super().__init__(*args, **kwargs)
        if strategy == "single":
            original = self._pool[AgentType.GENERAL][0]
            self._pool[AgentType.GENERAL] = [UnifiedServiceAgent(
                original._client, original._model, self._skill_manager,
            )]
        # These built-in tools inspect data or prepare advice; none executes a
        # refund/purchase. Remote tools retain their existing allowed_agents.
        tools = dict(self._shared_tools)
        for factory in (general_tools, technical_tools, billing_tools, escalation_tools):
            tools.update(factory())
        self.set_shared_tools(tools)

    def _route_decision(self, req):
        if self.strategy == "multi":
            return super()._route_decision(req)
        target = self._route(req.intent, req.urgency)
        # Preserve deterministic human escalation for all variants.
        if self.strategy == "single" and target != AgentType.ESCALATION:
            target = AgentType.GENERAL
        return RoutingDecision(
            primary_agent=target,
            reason=f"evaluation strategy={self.strategy}",
            confidence=req.intent_confidence,
        )
