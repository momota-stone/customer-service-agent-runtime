import pytest

from evaluation.contracts import matched_tool_requirements, score_contracts


def call(name="compare_amounts", **kwargs):
    return {"tool_name": name, "input": kwargs, "success": True, "result_success": True}


def test_one_tool_call_cannot_satisfy_two_requirements_and_matching_is_not_greedy():
    requirements = [{"tool_name": "compare_amounts", "input": {"amount_a": 119}},
                    {"tool_name": "compare_amounts", "input": {"amount_a": 119, "amount_b": 99}}]
    specific = call(amount_a=119, amount_b=99)
    assert matched_tool_requirements(requirements, [specific]) == 1
    assert matched_tool_requirements(requirements, [specific, call(amount_a=119, amount_b=89)]) == 2


def test_business_failure_and_boolean_arguments_are_not_correct_numeric_calls():
    required = [{"tool_name": "compare_amounts", "input": {"amount_a": 1}}]
    assert matched_tool_requirements(required, [call(amount_a=True)]) == 0
    failed = call(amount_a=1)
    failed["result_success"] = False
    assert matched_tool_requirements(required, [failed]) == 0


def test_missing_cases_and_extra_intents_are_not_hidden_as_perfect_accuracy():
    cases = [{"case_id": "a", "expected_intents": ["technical_login", "billing"],
              "required_tools": [{"tool_name": "lookup_error_code", "input": {"error_code": "401"}}]},
             {"case_id": "b", "expected_intents": ["billing"], "required_tools": [{"tool_name": "compare_amounts", "input": {"amount_a": 99}}]}]
    rows = [{"strategy": "single", "case_id": "a", "status": "success",
             "predicted_intents": ["technical_login", "other"], "tool_traces": [call("lookup_error_code", error_code="401")]}]
    report = score_contracts(cases, rows)["strategies"]["single"]
    assert report["missing_case_ids"] == ["b"]
    assert report["intent_micro_precision"] == .5
    assert report["intent_micro_recall"] == 1 / 3
    assert report["tool_contract_pass_rate"] == .5
    assert report["task_completion_rate"] is None
    with pytest.raises(ValueError, match="duplicate result"):
        score_contracts(cases, rows * 2)
