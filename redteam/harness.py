"""Red-team harness: measure attack-success-rate (ASR) with guard on vs off.

Loads the attack corpus from corpus.jsonl, injects each attack string into
a mock search result, runs it through screen_results() with the guard toggled,
and measures how many injections survive (reach the LLM unneutralized).

Usage:
    uv run python -m redteam.harness
    uv run python -m redteam.harness --corpus redteam/corpus.jsonl --runs 3
"""
import argparse
import json
import pathlib
import sys

CORPUS = pathlib.Path(__file__).parent / "corpus.jsonl"


def load_corpus(path: pathlib.Path) -> list[dict]:
    items = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def simulate_search_result(attack_text: str) -> dict:
    """Wrap an attack string in a mock search-result dict."""
    return {
        "title": "Malicious page title",
        "url": "https://evil.example.com/payload",
        "content": attack_text,
    }


def measure_asr(corpus: list[dict], guard_on: bool, runs: int = 1) -> dict:
    """Run each attack through the guard pipeline and count survivals.

    An attack "survives" if the guard is off, or if the guard is on but the
    attack text appears unmodified in the output (neutralize missed it).
    """
    import os
    os.environ["A2A_GUARD"] = "on" if guard_on else "off"

    # guard_enabled() reads A2A_GUARD at call time, so the toggle above
    # takes effect immediately.
    from common.guard import screen_results

    attacks = [c for c in corpus if c["type"] == "attack"]
    benigns = [c for c in corpus if c["type"] == "benign"]

    survived = 0
    total = 0
    missed_categories: dict[str, int] = {}
    false_positives = 0

    for _ in range(runs):
        for item in attacks:
            total += 1
            result = simulate_search_result(item["text"])
            if guard_on:
                screened, _ = screen_results([result])
                content = screened[0]["content"]
                # The attack survived if the original text still appears
                # unredacted in the output.
                if item["text"] in content:
                    survived += 1
                    missed_categories[item["category"]] = (
                        missed_categories.get(item["category"], 0) + 1
                    )
            else:
                # Guard off: every attack survives.
                survived += 1

        # False-positive check: benign text should pass through unchanged.
        if guard_on:
            for item in benigns:
                result = simulate_search_result(item["text"])
                screened, _ = screen_results([result])
                content = screened[0]["content"]
                # Check that neutralize didn't add REDACTION to benign text.
                from common.guard import REDACTION
                if REDACTION in content:
                    false_positives += 1

    asr = survived / total if total else 0.0
    return {
        "guard_enabled": guard_on,
        "total_attacks": total,
        "survived": survived,
        "asr": round(asr, 4),
        "missed_categories": missed_categories,
        "false_positives": false_positives,
    }


def main():
    parser = argparse.ArgumentParser(description="Red-team guard harness")
    parser.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat N times for statistical stability")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    attacks = [c for c in corpus if c["type"] == "attack"]
    benigns = [c for c in corpus if c["type"] == "benign"]
    print(f"Loaded corpus: {len(attacks)} attacks, {len(benigns)} benign", file=sys.stderr)

    off = measure_asr(corpus, guard_on=False, runs=args.runs)
    on = measure_asr(corpus, guard_on=True, runs=args.runs)

    if args.json:
        print(json.dumps({"guard_off": off, "guard_on": on}, indent=2))
        return

    print("\n=== Red-Team Harness Results ===\n")
    print(f"  Corpus:          {len(attacks)} attacks, {len(benigns)} benign")
    print(f"  Runs per config: {args.runs}")
    print()
    print(f"  Guard OFF:  ASR = {off['asr']:.1%}  "
          f"({off['survived']}/{off['total_attacks']} survived)")
    print(f"  Guard ON:   ASR = {on['asr']:.1%}  "
          f"({on['survived']}/{on['total_attacks']} survived)")
    if on["missed_categories"]:
        print(f"  Missed by category: {on['missed_categories']}")
    print(f"  False positives (guard on, benign text redacted): {on['false_positives']}")
    reduction = (1 - on["asr"] / off["asr"]) * 100 if off["asr"] else 0
    print(f"\n  ASR reduction: {reduction:.1f}%")
    print()


if __name__ == "__main__":
    main()
