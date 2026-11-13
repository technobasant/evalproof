from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from evalproof.config import GenerationParams
from evalproof.models import CaseResult, Run, RunInfo, Score, SystemOutput
from evalproof.providers import ChatResponse, Message


def make_run(
    outcomes: dict[str, dict[str, bool | float | None]],
    *,
    system: str = "sys",
    dataset_hash: str = "h",
    latency: float = 10.0,
    case_hashes: bool | dict[str, str] = False,
) -> Run:
    """Build a run from ``{case_id: {scorer: passed_or_value}}``. Booleans become binary
    scores; floats become continuous scores; None becomes undecided.

    ``case_hashes`` stamps per-case hashes the way evalproof 0.5 does: True gives every case
    the hash ``h-<id>``, a dict sets the hash of the ids it names. The default writes a run as
    older evalproof did, with none."""
    results = []
    for cid, scores in outcomes.items():
        built: dict[str, Score] = {}
        for name, v in scores.items():
            if v is None:
                built[name] = Score(value=None)
            elif isinstance(v, bool):
                built[name] = Score(value=1.0 if v else 0.0, passed=v)
            else:
                built[name] = Score(value=float(v))
        stamp = (
            case_hashes.get(cid)
            if isinstance(case_hashes, dict)
            else (f"h-{cid}" if case_hashes else None)
        )
        results.append(
            CaseResult(
                case_id=cid,
                output=SystemOutput(output="x", latency_ms=latency),
                scores=built,
                case_hash=stamp,
            )
        )
    info = RunInfo(
        suite="t",
        system=system,
        dataset_path="d.jsonl",
        dataset_hash=dataset_hash,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        wall_time_s=1.0,
        evalproof_version="0.1.0",
        seed=0,
    )
    return Run(info=info, results=results)


class ScriptedProvider:
    """A provider that replies from a list, recording every call."""

    name = "scripted"
    model = "m"

    def __init__(self, replies: list[str] | Callable[[list[Message]], str]) -> None:
        self.replies = replies
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        self.calls.append(messages)
        if callable(self.replies):
            text = self.replies(messages)
        else:
            text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return ChatResponse(text=text, model=self.model, tokens_in=10, tokens_out=5, latency_ms=3.0)


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def write_jsonl(tmp_path: Path) -> Callable[[str, list[dict[str, object]]], Path]:
    def _write(name: str, rows: list[dict[str, object]]) -> Path:
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    return _write
