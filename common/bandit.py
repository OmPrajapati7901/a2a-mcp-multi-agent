"""Epsilon-greedy bandit for cost-quality model routing.

Routes writing tasks across model tiers (cheap vs strong) based on cumulative
reward (judge scores). The bandit balances exploration (trying all models) with
exploitation (using the best-performing one). Over time it learns which model
tier gives the best quality per dollar for a given task class.

Usage:
    from common.bandit import BanditRouter
    router = BanditRouter()             # loads from sqlite if available
    model = router.choose()             # e.g. "glm-5.2" or "glm-5.2-mini"
    score = judge(report)               # 0.0–1.0 quality score
    router.record(model, score, cost)   # update reward history

Environment:
    A2A_BANDIT_EPSILON  — exploration rate (default 0.15)
    A2A_BANDIT_DB       — sqlite path (default: .cache/bandit.db, or
                          ":memory:" for a throwaway router in tests)
"""
import json
import logging
import os
import pathlib
import random
import sqlite3
import time
from dataclasses import dataclass, field

logger = logging.getLogger("bandit")

# Default model tiers with cost hints ($/MTok in, $/MTok out).
DEFAULT_ARMS: list[dict] = [
    {
        "name": "glm-5.2",
        "cost_in_per_mtok": 0.50,
        "cost_out_per_mtok": 1.50,
        "tier": "strong",
    },
    {
        "name": "glm-5.2-mini",
        "cost_in_per_mtok": 0.10,
        "cost_out_per_mtok": 0.30,
        "tier": "cheap",
    },
]


def _default_db_path() -> pathlib.Path:
    """Same directory convention as the semantic cache."""
    root = pathlib.Path(os.environ.get("A2A_CACHE_DIR", ".cache"))
    root.mkdir(exist_ok=True)
    return root / "bandit.db"


@dataclass
class ArmStats:
    name: str
    tier: str
    cost_in: float
    cost_out: float
    pulls: int = 0
    total_reward: float = 0.0
    total_cost: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.pulls if self.pulls else 0.0

    @property
    def avg_reward_per_dollar(self) -> float:
        return self.total_reward / self.total_cost if self.total_cost else 0.0


class BanditRouter:
    """Epsilon-greedy multi-armed bandit over model tiers.

    Persists pull/reward history to sqlite so the bandit learns across
    multiple pipeline runs. In-memory mode for tests.
    """

    def __init__(
        self,
        arms: list[dict] | None = None,
        epsilon: float | None = None,
        db_path: str | None = None,
    ):
        self.epsilon = epsilon if epsilon is not None else float(
            os.environ.get("A2A_BANDIT_EPSILON", "0.15")
        )
        self.arms_spec = arms or DEFAULT_ARMS
        self.arms: dict[str, ArmStats] = {}

        # Persistence: a file-backed DB by default so the bandit actually
        # learns across pipeline runs (each graph node builds its own
        # router); ":memory:" gives tests a throwaway instance.
        self._db_path = (db_path or os.environ.get("A2A_BANDIT_DB")
                         or str(_default_db_path()))
        self._conn = sqlite3.connect(self._db_path)
        self._init_db()
        self._load()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bandit_pulls (
                arm_name TEXT,
                reward REAL,
                cost REAL,
                ts REAL
            )
        """)
        self._conn.commit()

    def _load(self):
        for spec in self.arms_spec:
            self.arms[spec["name"]] = ArmStats(
                name=spec["name"],
                tier=spec["tier"],
                cost_in=spec["cost_in_per_mtok"],
                cost_out=spec["cost_out_per_mtok"],
            )
        rows = self._conn.execute(
            "SELECT arm_name, reward, cost FROM bandit_pulls ORDER BY ts"
        ).fetchall()
        for arm_name, reward, cost in rows:
            if arm_name in self.arms:
                arm = self.arms[arm_name]
                arm.pulls += 1
                arm.total_reward += reward
                arm.total_cost += cost

    def choose(self) -> str:
        """Pick a model tier. Epsilon-greedy: explore randomly with
        probability epsilon, exploit the best arm otherwise."""
        if random.random() < self.epsilon:
            name = random.choice(list(self.arms.keys()))
            logger.info("bandit: explore → %s", name)
            return name

        best = max(self.arms.values(), key=lambda a: a.avg_reward)
        if best.pulls == 0:
            # No data yet — pick randomly.
            name = random.choice(list(self.arms.keys()))
        else:
            name = best.name
        logger.info("bandit: exploit → %s (avg_reward=%.3f, pulls=%d)",
                     name, best.avg_reward, best.pulls)
        return name

    def record(self, arm_name: str, reward: float, cost: float):
        """Record a pull result: reward (0–1 quality score) and cost ($)."""
        if arm_name not in self.arms:
            logger.warning("bandit: unknown arm %r, ignoring", arm_name)
            return
        arm = self.arms[arm_name]
        arm.pulls += 1
        arm.total_reward += reward
        arm.total_cost += cost
        self._conn.execute(
            "INSERT INTO bandit_pulls (arm_name, reward, cost, ts) VALUES (?,?,?,?)",
            (arm_name, reward, cost, time.time()),
        )
        self._conn.commit()
        logger.info("bandit: recorded %s reward=%.3f cost=$%.4f → "
                     "avg_reward=%.3f (pulls=%d)",
                     arm_name, reward, cost, arm.avg_reward, arm.pulls)

    def estimate_cost(self, arm_name: str, input_tokens: int,
                      output_tokens: int) -> float:
        """Dollar cost of one pull, from the arm's $/MTok spec and the
        actual token usage of the call it routed."""
        arm = self.arms.get(arm_name)
        if arm is None:
            return 0.0
        return round(
            (input_tokens * arm.cost_in + output_tokens * arm.cost_out)
            / 1_000_000, 6,
        )

    def frontier(self) -> list[dict]:
        """Return cost-quality frontier data for plotting."""
        result = []
        for arm in self.arms.values():
            if arm.pulls > 0:
                result.append({
                    "name": arm.name,
                    "tier": arm.tier,
                    "pulls": arm.pulls,
                    "avg_reward": round(arm.avg_reward, 4),
                    "avg_cost": round(arm.avg_cost, 6),
                    "reward_per_dollar": round(arm.avg_reward_per_dollar, 4),
                })
        return sorted(result, key=lambda x: x["avg_cost"])

    def reset(self):
        """Clear all history (for tests or re-evaluation)."""
        self._conn.execute("DELETE FROM bandit_pulls")
        self._conn.commit()
        for arm in self.arms.values():
            arm.pulls = 0
            arm.total_reward = 0.0
            arm.total_cost = 0.0

    def summary(self) -> str:
        lines = ["Bandit Router Status:"]
        for arm in sorted(self.arms.values(), key=lambda a: -a.avg_reward):
            lines.append(
                f"  {arm.name:20s}  pulls={arm.pulls:4d}  "
                f"avg_reward={arm.avg_reward:.3f}  "
                f"avg_cost=${arm.avg_cost:.4f}  "
                f"reward/$={arm.avg_reward_per_dollar:.2f}"
            )
        return "\n".join(lines)
