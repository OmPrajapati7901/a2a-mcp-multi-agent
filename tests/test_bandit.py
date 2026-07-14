"""Tests for the epsilon-greedy bandit model router."""
import os
import sqlite3

import pytest

os.environ.setdefault("A2A_DEMO_OFFLINE", "1")

from common.bandit import BanditRouter, ArmStats


@pytest.fixture
def router():
    """Fresh in-memory bandit for each test."""
    return BanditRouter(db_path=":memory:", epsilon=0.0)


class TestBanditRouter:
    def test_choose_exploits_best(self, router):
        # Seed glm-5.2 with high rewards, glm-5.2-mini with low.
        for _ in range(10):
            router.record("glm-5.2", 0.9, 0.01)
            router.record("glm-5.2-mini", 0.3, 0.005)

        # With epsilon=0 (no exploration), always pick the best.
        for _ in range(20):
            assert router.choose() == "glm-5.2"

    def test_explore_mode_picks_any(self):
        router = BanditRouter(db_path=":memory:", epsilon=1.0)
        # With epsilon=1 (always explore), picks are random.
        choices = {router.choose() for _ in range(50)}
        assert len(choices) == 2  # both arms explored

    def test_record_updates_stats(self, router):
        router.record("glm-5.2", 0.8, 0.01)
        arm = router.arms["glm-5.2"]
        assert arm.pulls == 1
        assert arm.total_reward == 0.8
        assert arm.total_cost == 0.01
        assert arm.avg_reward == 0.8

    def test_persistence_across_instances(self, tmp_path):
        db = str(tmp_path / "bandit.db")
        r1 = BanditRouter(db_path=db, epsilon=0.0)
        r1.record("glm-5.2", 0.9, 0.01)
        r1.record("glm-5.2", 0.85, 0.01)
        del r1

        r2 = BanditRouter(db_path=db, epsilon=0.0)
        arm = r2.arms["glm-5.2"]
        assert arm.pulls == 2
        assert abs(arm.avg_reward - 0.875) < 1e-6

    def test_frontier(self, router):
        router.record("glm-5.2", 0.9, 0.05)
        router.record("glm-5.2-mini", 0.7, 0.01)
        f = router.frontier()
        assert len(f) == 2
        # Cheapest first.
        assert f[0]["name"] == "glm-5.2-mini"
        assert f[0]["avg_reward"] == 0.7

    def test_reset(self, router):
        router.record("glm-5.2", 0.9, 0.01)
        router.reset()
        assert router.arms["glm-5.2"].pulls == 0
        assert router.frontier() == []

    def test_unknown_arm_ignored(self, router):
        router.record("nonexistent-model", 0.5, 0.01)
        # Should not crash; no data recorded.
        assert router.frontier() == []

    def test_summary(self, router):
        router.record("glm-5.2", 0.9, 0.01)
        s = router.summary()
        assert "glm-5.2" in s
        assert "pulls=" in s
        assert "avg_reward=" in s

    def test_no_pulls_exploits_randomly(self):
        router = BanditRouter(db_path=":memory:", epsilon=0.0)
        # No data — should still return a valid arm name.
        choice = router.choose()
        assert choice in ("glm-5.2", "glm-5.2-mini")

    def test_reward_per_dollar(self, router):
        # Cheap model: 0.7 reward at $0.005 → 140 reward/$
        router.record("glm-5.2-mini", 0.7, 0.005)
        # Strong model: 0.9 reward at $0.02 → 45 reward/$
        router.record("glm-5.2", 0.9, 0.02)
        cheap = router.arms["glm-5.2-mini"]
        strong = router.arms["glm-5.2"]
        assert cheap.avg_reward_per_dollar > strong.avg_reward_per_dollar

    def test_estimate_cost_from_arm_spec(self, router):
        # glm-5.2: $0.50/MTok in, $1.50/MTok out.
        cost = router.estimate_cost("glm-5.2", 1_000_000, 1_000_000)
        assert abs(cost - 2.0) < 1e-9
        assert router.estimate_cost("nonexistent-model", 1000, 1000) == 0.0


def test_pipeline_records_approve_reward(tmp_path):
    """Regression: an approved report must earn the full 1.0 reward — the
    critic's verdict is 'approve' (never 'pass'), and the reward comparison
    once used the wrong literal, flattening every pull to 0.3."""
    from tests.test_e2e_offline import run_demo

    db = tmp_path / "bandit.db"
    proc = run_demo({
        "A2A_CRITIC": "1",
        "A2A_BANDIT_DB": str(db),
        "WRITER_AGENT_PORT": "9158",
        "CRITIC_AGENT_PORT": "9159",
    })
    assert proc.returncode == 0, f"demo failed:\n{proc.stderr[-3000:]}"
    assert "critic verdict: approve" in proc.stderr
    rows = sqlite3.connect(db).execute(
        "SELECT arm_name, reward FROM bandit_pulls"
    ).fetchall()
    assert rows, "bandit recorded no pulls"
    assert any(reward == 1.0 for _, reward in rows), (
        f"no full-credit reward recorded for the approved report: {rows}"
    )
