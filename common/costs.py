"""Per-run token/cost accounting across agents, with a budget kill switch.

Each agent reports its token usage; the Writer's flows back over the A2A
artifact's data part so the caller can attribute cost per agent. When the
model doesn't report usage (offline mode), we estimate at ~4 chars/token and
mark it as such. A2A_BUDGET_TOKENS aborts the pipeline *before* the next
delegation once the budget is hit — spend control, not post-hoc reporting.
"""
import logging
import os

logger = logging.getLogger("costs")

# $ per million tokens; override for your provider's pricing.
PRICE_IN_PER_MTOK = float(os.environ.get("A2A_PRICE_IN_PER_MTOK", "0.60"))
PRICE_OUT_PER_MTOK = float(os.environ.get("A2A_PRICE_OUT_PER_MTOK", "2.20"))


class BudgetExceededError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Ledger:
    def __init__(self) -> None:
        self.by_agent: dict[str, dict] = {}
        self.estimated = False

    def add(self, agent: str, input_tokens: int, output_tokens: int,
            estimated: bool = False) -> None:
        entry = self.by_agent.setdefault(agent, {"input": 0, "output": 0})
        entry["input"] += int(input_tokens)
        entry["output"] += int(output_tokens)
        self.estimated = self.estimated or estimated
        logger.info(
            "ledger: %s +%d in / +%d out tokens%s (run total %d)",
            agent, input_tokens, output_tokens,
            " (estimated)" if estimated else "", self.total_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return sum(e["input"] + e["output"] for e in self.by_agent.values())

    def cost_usd(self) -> float:
        cost = sum(
            e["input"] * PRICE_IN_PER_MTOK + e["output"] * PRICE_OUT_PER_MTOK
            for e in self.by_agent.values()
        ) / 1_000_000
        return round(cost, 6)

    def check_budget(self, about_to: str) -> None:
        """Raise if the configured token budget is already spent — called
        before each delegation so we stop spending, not report overspend."""
        budget = int(os.environ.get("A2A_BUDGET_TOKENS", "0"))
        if budget and self.total_tokens >= budget:
            raise BudgetExceededError(
                f"run used {self.total_tokens} tokens ≥ budget {budget} — "
                f"refusing to {about_to}"
            )

    def summary(self) -> str:
        parts = [
            f"{agent}={e['input'] + e['output']}tok"
            for agent, e in self.by_agent.items()
        ]
        est = "~" if self.estimated else ""
        return (f"{' '.join(parts) or 'no usage recorded'} | "
                f"total={self.total_tokens}tok | cost={est}${self.cost_usd()}")
