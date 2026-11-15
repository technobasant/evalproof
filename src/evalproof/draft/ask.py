"""Ask a model for test inputs, never for answers.

The model is told to write only what a user would send. Its reply is parsed strictly as a JSON
array; a reply that is not one gets a single retry, and then the gap is skipped. Nothing the model
returns is ever written as an expected answer.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalproof.config import GenerationParams
from evalproof.providers import Message, Provider

MIN_REPLY_TOKENS = 1024
TOKENS_PER_INPUT = 160  # a rough room for one drafted input (several turns need more)
SAMPLE_EXISTING = 12  # how many existing inputs the model sees as examples
EXAMPLE_CHARS = 160
CONTEXT_CHARS = 900

SYSTEM = (
    "You write test inputs for an LLM application. An input is what a person would type to the "
    "application. You never write the application's answers, expected answers, grading notes or "
    "explanations: only the inputs. Reply with a single JSON array and nothing else: no prose, "
    "no code fence."
)
RETRY = (
    "That reply could not be used. Reply again with only the JSON array described above: no "
    "prose, no code fence, and nothing before or after the brackets."
)

_FENCE = re.compile(r"\A```[a-zA-Z0-9_-]*\s*\n(.*)\n```\s*\Z", re.DOTALL)


@dataclass
class Request:
    """What one gap asks the model for."""

    gap: str
    label: str  # "tool_unchecked refund", for messages
    brief: str  # what to write inputs for
    count: int
    multi_turn: bool = False


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_messages(request: Request, existing: list[str]) -> list[Message]:
    if request.multi_turn:
        shape = (
            f"Reply with a JSON array of {request.count} conversations. Each conversation is an "
            "array of 2 to 4 strings: the user's first message, then the user's later messages "
            "in order. Write only the user's side; the later messages should depend on what "
            "was said before (a correction, a follow-up, a pronoun, a change of mind)."
        )
    else:
        shape = (
            f"Reply with a JSON array of {request.count} strings, one input each. Vary the "
            "wording, length and difficulty; include an input with a detail missing or "
            "ambiguous and one a careless user would type."
        )
    sample = [_short(e, EXAMPLE_CHARS) for e in existing[:SAMPLE_EXISTING]]
    avoid = (
        "\n\nInputs the test set already has (write different ones, not rewordings of these):\n"
        + "\n".join(f"- {s}" for s in sample)
        if sample
        else ""
    )
    return [
        Message(role="system", content=SYSTEM),
        Message(role="user", content=f"{request.brief}\n\n{shape}{avoid}"),
    ]


def strip_fence(text: str) -> str:
    text = text.strip()
    match = _FENCE.match(text)
    return match.group(1).strip() if match else text


def parse_reply(text: str, *, multi_turn: bool) -> list[list[str]] | None:
    """The inputs in ``text``, each as ``[input, *turns]``, or ``None`` when the reply is not
    what was asked for: not JSON, not an array, an empty array, or an element of the wrong type.
    Empty strings are dropped; a conversation with fewer than two messages is not multi-turn and
    is dropped too."""
    try:
        data = json.loads(strip_fence(text))
    except (json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(data, list) or not data:
        return None
    out: list[list[str]] = []
    for item in data:
        if multi_turn:
            if not isinstance(item, list) or not all(isinstance(m, str) for m in item):
                return None
            messages = [m.strip() for m in item if m.strip()]
            if len(messages) >= 2:
                out.append(messages)
        else:
            if not isinstance(item, str):
                return None
            if item.strip():
                out.append([item.strip()])
    return out


def key_of(messages: list[str]) -> str:
    """What makes two drafted inputs the same input: case, spacing and a trailing stop aside."""
    return " ␟ ".join(" ".join(m.casefold().split()).rstrip(".!? ") for m in messages)


def ask(
    provider: Provider,
    params: GenerationParams,
    request: Request,
    existing: list[str],
) -> tuple[list[list[str]] | None, str | None]:
    """The parsed inputs for ``request``, or ``(None, reason)`` after a malformed reply and one
    retry. A provider error is raised, not caught: the key, the endpoint or the model is wrong."""
    messages = build_messages(request, existing)
    room = max(params.max_tokens, MIN_REPLY_TOKENS, TOKENS_PER_INPUT * request.count * 3)
    limited = params.model_copy(update={"max_tokens": room})
    reply = ""
    for attempt in range(2):
        reply = provider.chat(messages, limited).text
        parsed = parse_reply(reply, multi_turn=request.multi_turn)
        if parsed is not None:
            return parsed, None
        if attempt == 0:  # the retry differs from the first call, so it is not a cache hit
            messages = [
                *messages,
                Message(role="assistant", content=reply),
                Message(role="user", content=RETRY),
            ]
    return (
        None,
        f"the model's reply was not a JSON array of the inputs asked for: {_short(reply, 80)!r}",
    )


# --- what each gap tells the model --------------------------------------------------------------


def _function_at(path: Path, line: int) -> dict[str, str] | None:
    """The function in ``path`` that contains ``line``: its name, signature and docstring."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return None
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = node.end_lineno or node.lineno
            if node.lineno <= line <= end and (best is None or node.lineno >= best.lineno):
                best = node
    if best is None:
        return None
    returns = f" -> {ast.unparse(best.returns)}" if best.returns is not None else ""
    return {
        "name": best.name,
        "signature": f"{best.name}({ast.unparse(best.args)}){returns}",
        "doc": (ast.get_docstring(best) or "")[:CONTEXT_CHARS],
    }


def _prompts_in(findings: list[dict[str, Any]], path: str | None = None) -> list[str]:
    return [
        _short(str(f["excerpt"]), CONTEXT_CHARS)
        for f in findings
        if f.get("kind") == "prompt"
        and f.get("excerpt")
        and (path is None or f.get("path") == path)
    ]


def tool_request(tool: dict[str, Any], count: int, findings: list[dict[str, Any]]) -> Request:
    name = str(tool.get("name", ""))
    lines = [f"The application can call a tool named `{name}`."]
    if tool.get("signature"):
        lines.append(f"Its signature: {tool['signature']}")
    if tool.get("doc"):
        lines.append(f"Its description: {_short(str(tool['doc']), CONTEXT_CHARS)}")
    prompts = _prompts_in(findings)[:1]
    if prompts:
        lines.append(f"The application's instructions begin: {prompts[0]}")
    lines.append(
        f"Write {count} different user messages that should make the application use `{name}`, "
        "some that ask for it directly and some that need it without naming it."
    )
    return Request("tool_unchecked", f"tool_unchecked {name}", "\n".join(lines), count)


def multi_turn_request(
    evidence: list[dict[str, Any]], count: int, findings: list[dict[str, Any]]
) -> Request:
    lines = ["The application keeps a conversation going across several user messages."]
    for f in evidence[:3]:
        lines.append(
            f"Evidence in its code: {f.get('detail', '')} ({f.get('path')}:{f.get('line')})"
        )
    prompts = _prompts_in(findings)[:1]
    if prompts:
        lines.append(f"The application's instructions begin: {prompts[0]}")
    lines.append(f"Write {count} different conversations with it.")
    return Request("no_multi_turn_cases", "no_multi_turn_cases", "\n".join(lines), count, True)


def code_request(
    root: Path, path: str, line: int, count: int, findings: list[dict[str, Any]]
) -> Request:
    lines = [
        f"The application has code in `{path}` that calls a model, builds an agent or "
        "defines a tool."
    ]
    fn = _function_at(root / path, line)
    if fn:
        lines.append(f"The code is the function `{fn['signature']}`.")
        if fn["doc"]:
            lines.append(f"Its docstring: {fn['doc']}")
    in_file = [f for f in findings if f.get("path") == path and f.get("kind") in ("agent", "tool")]
    for f in in_file[:4]:
        lines.append(f"It contains: {f.get('kind')} {f.get('detail', '')}")
    prompts = _prompts_in(findings, path)[:2]
    for text in prompts:
        lines.append(f"A prompt in that file: {text}")
    lines.append(f"Write {count} different user messages that would reach this code.")
    return Request(
        "uncovered_code", f"uncovered_code {posixpath.basename(path)}", "\n".join(lines), count
    )


def few_cases_request(suite: dict[str, Any], count: int, findings: list[dict[str, Any]]) -> Request:
    lines = [
        f"A test set named {suite.get('name')!r} has {suite.get('cases')} inputs, and it needs "
        f"at least {suite.get('min_cases')} to be a useful gate.",
        f"The application under test is `{suite.get('target')}`.",
    ]
    prompts = _prompts_in(findings)[:1]
    if prompts:
        lines.append(f"The application's instructions begin: {prompts[0]}")
    lines.append(f"Write {count} new inputs of the same kind as the ones already there.")
    return Request("few_cases", f"few_cases {suite.get('path')}", "\n".join(lines), count)
