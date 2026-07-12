"""Cost ledger and budget kill-switch tests."""
import pytest

from common.costs import BudgetExceededError, Ledger
from tests.test_e2e_offline import run_demo


def test_ledger_accounts_per_agent(monkeypatch):
    monkeypatch.delenv("A2A_BUDGET_TOKENS", raising=False)
    ledger = Ledger()
    ledger.add("research-agent", 100, 50)
    ledger.add("writer-agent", 200, 300, estimated=True)
    assert ledger.total_tokens == 650
    assert ledger.estimated
    assert "research-agent=150tok" in ledger.summary()
    ledger.check_budget("anything")  # no budget set → never raises


def test_budget_raises_when_spent(monkeypatch):
    monkeypatch.setenv("A2A_BUDGET_TOKENS", "100")
    ledger = Ledger()
    ledger.add("research-agent", 90, 20)
    with pytest.raises(BudgetExceededError, match="refusing to delegate"):
        ledger.check_budget("delegate")


def test_budget_kill_switch_stops_before_delegation():
    proc = run_demo({"A2A_BUDGET_TOKENS": "10", "WRITER_AGENT_PORT": "9121"})
    assert proc.returncode != 0
    assert "BudgetExceededError" in proc.stderr
    # The kill switch fired before the A2A handoff, not after.
    assert "A2A HANDOFF" not in proc.stderr


def test_cost_line_in_output():
    proc = run_demo({"WRITER_AGENT_PORT": "9122"})
    assert proc.returncode == 0
    assert "cost:" in proc.stdout
    assert "writer-agent=" in proc.stdout
