from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

TESTING_SUITE: list[str] = [
    "Who directed the titanic?",
    "Please answer this question with a factual approach: From what country is the movie 'Aro Tolbukhin. En la mente del asesino'?",
    "Please answer this question with a factual approach: Who is the screenwriter of 'Shortcut to Happiness'?",
    "Please answer this question with a factual approach: Who directed ‘Fargo’?",
    "Please answer this question with a factual approach: What genre is the movie 'Bandit Queen'?",
    "Please answer this question with a factual approach: When did the movie 'Miracles Still Happen' come out?",
    "Please answer this question with an embedding approach: Who is the director of ‘Apocalypse Now’?",
    "Please answer this question with an embedding approach: Who is the screenwriter of ‘12 Monkeys’?",
    "Please answer this question with an embedding approach: What is the genre of ‘Shoplifters’?",
    "Please answer this question: Who is the director of ‘Good Will Hunting’?",
    "Please answer this question with embedding approach: Who is the director of ‘Good Will Hunting’?",
    "Who directed Titanoc?",
    "Please answer this question with factual approach: Who is the director of Galaxy Quest?",
    "Please answer this question with embedding approach: who is the director of Galaxy Quest?",
    "Please answer this question: who is the director of Galaxy Quest?",
    "Please answer this question with embedding approach: When did the movie 'Miracles still happen' came out?",
    "When was 'It's a Wonderful Life' first released?",
]


@dataclass
class TestSuiteResult:
    index: int
    question: str
    answer: str
    ok: bool
    error: str | None = None


def run_test_suite(agent, questions: Iterable[str] | None = None) -> list[TestSuiteResult]:
    items = list(questions or TESTING_SUITE)
    results: list[TestSuiteResult] = []
    for index, question in enumerate(items, start=1):
        try:
            answer = agent.handle_message(question)
            results.append(TestSuiteResult(index=index, question=question, answer=answer, ok=True))
        except Exception as exc:  # pragma: no cover - defensive wrapper for live debugging
            results.append(
                TestSuiteResult(
                    index=index,
                    question=question,
                    answer="",
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def build_text_report(results: Iterable[TestSuiteResult]) -> str:
    rows = list(results)
    passed = sum(1 for row in rows if row.ok)
    failed = len(rows) - passed
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts: list[str] = [
        "CineReason test suite report",
        f"Generated: {timestamp}",
        f"Total: {len(rows)} | Passed: {passed} | Failed: {failed}",
        "",
    ]

    for row in rows:
        status = "PASS" if row.ok else "FAIL"
        parts.append(f"=== Test {row.index:02d} [{status}] ===")
        parts.append(f"Question: {row.question}")
        if row.ok:
            parts.append(f"Answer: {row.answer}")
        else:
            parts.append(f"Error: {row.error}")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"
