import pytest

from agents.tools import compare_amounts


@pytest.mark.parametrize("first,second", [(float("nan"), 99), (float("inf"), 99),
                                         (99, float("-inf")), (1e308, -1e308), (True, 99)])
def test_invalid_amounts_do_not_return_successful_financial_results(first, second):
    result = compare_amounts(None, {"amount_a": first, "amount_b": second})
    assert result["success"] is False
    assert "difference" not in result
    assert result["error"]


def test_normal_amount_difference_preserves_existing_contract():
    result = compare_amounts(None, {"amount_a": 119, "amount_b": 99})
    assert result["success"] is True
    assert result["difference"] == 20
    assert "不代表" in result["interpretation"]
