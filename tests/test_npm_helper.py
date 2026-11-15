"""The JavaScript helper in ``npm/lib`` against the Python side it talks to.

The unit tests of the helper itself are ``npm/test`` (``npm test``). These start ``node`` for real
and read what it writes with evalproof's own code: a ``command:`` target through
``CommandSystem``, an ``http://`` target through ``HttpSystem``, and the LangChain converter's
trace against the Python adapter's. Skipped when ``node`` is not installed. Nothing here calls a
model or leaves localhost.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from evalproof.adapters.langchain import from_messages
from evalproof.config import TargetConfig
from evalproof.models import Case, Trace
from evalproof.systems import CommandSystem, HttpSystem

NODE = shutil.which("node")
LIB = (Path(__file__).resolve().parent.parent / "npm" / "lib" / "index.js").as_posix()

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

MESSAGES: list[dict[str, Any]] = [
    {"type": "human", "content": "Find the capital of France"},
    {
        "type": "ai",
        "content": "",
        "tool_calls": [{"name": "search", "args": {"q": "capital of France"}, "id": "call_1"}],
        "usage_metadata": {
            "input_tokens": 12,
            "output_tokens": 4,
            "input_token_details": {"cache_read": 3},
        },
        "response_metadata": {"model_name": "model-a"},
    },
    {"type": "tool", "tool_call_id": "call_1", "name": "search", "content": "Paris"},
    {
        "type": "ai",
        "content": "The capital is Paris.",
        "usage_metadata": {"input_tokens": 20, "output_tokens": 6},
        "response_metadata": {"model_name": "model-a"},
    },
    {"type": "human", "content": "And of Italy?"},
    {
        "type": "ai",
        "content": [{"type": "text", "text": "Checking."}],
        "tool_calls": [{"name": "search", "args": {"q": "capital of Italy"}, "id": "call_2"}],
    },
    {
        "type": "tool",
        "tool_call_id": "call_2",
        "name": "search",
        "content": "no such page",
        "status": "error",
    },
    {"type": "ai", "content": "I could not find it."},
]


def run_node(script: str, stdin: str = "") -> str:
    assert NODE
    done = subprocess.run(
        [NODE, "-e", script],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_langchain_converter_trace_equals_the_python_adapter() -> None:
    script = f"""
    const {{ fromLangChain }} = require({json.dumps(LIB)});
    let text = '';
    process.stdin.on('data', (c) => (text += c)).on('end', () =>
      console.log(JSON.stringify(fromLangChain(JSON.parse(text)))));
    """
    js = json.loads(run_node(script, json.dumps(MESSAGES)))
    py = from_messages(MESSAGES)
    py_trace = py["trace"]
    assert isinstance(py_trace, Trace)
    assert js["trace"] == py_trace.model_dump(mode="json")
    assert js["output"] == py["output"] == "I could not find it."
    assert (js["tokens_in"], js["tokens_out"]) == (32, 10)


def test_converter_output_is_accepted_as_a_trace() -> None:
    script = f"""
    const {{ fromVercelAI }} = require({json.dumps(LIB)});
    const call = {{ toolCallId: 'c1', toolName: 'lookup', input: {{ id: 7 }} }};
    console.log(JSON.stringify(fromVercelAI({{
      text: 'found it',
      steps: [
        {{ text: '', toolCalls: [call],
           toolResults: [{{ ...call, output: {{ ok: true }} }}],
           usage: {{ inputTokens: 5, outputTokens: 2 }}, response: {{ modelId: 'm' }} }},
        {{ text: 'found it', toolCalls: [], toolResults: [],
           usage: {{ inputTokens: 8, outputTokens: 3 }}, response: {{ modelId: 'm' }} }},
      ],
    }})));
    """
    got = json.loads(run_node(script))
    trace = Trace.model_validate(got["trace"])
    assert trace.responses == ["found it"]
    assert [(c.name, c.args, c.result, c.ok, c.id) for c in trace.tool_calls] == [
        ("lookup", {"id": 7}, {"ok": True}, True, "c1")
    ]
    assert trace.rounds == 2
    assert [(u.model, u.tokens_in, u.tokens_out) for u in trace.usage] == [("m", 13, 5)]
    assert (got["tokens_in"], got["tokens_out"]) == (13, 5)


def test_a_command_target_written_with_runcommand_runs_through_evalproof(tmp_path: Path) -> None:
    (tmp_path / "system.js").write_text(
        f"""
const {{ runCommand, fromLangChain }} = require({json.dumps(LIB)});
runCommand(async (request) => {{
  console.log('this line is a log, not the answer');
  const reply = fromLangChain([
    {{ type: 'human', content: request.input.question }},
    {{ type: 'ai', content: '', tool_calls: [{{ name: 'calc', args: {{ expr: '2+2' }}, id: 'k' }}],
       usage_metadata: {{ input_tokens: 9, output_tokens: 3 }} }},
    {{ type: 'tool', tool_call_id: 'k', name: 'calc', content: '4' }},
    {{ type: 'ai', content: 'The answer is 4.' }},
  ]);
  return {{ ...reply, retrieved: ['doc-1'] }};
}});
""",
        encoding="utf-8",
    )
    system = CommandSystem(
        TargetConfig(target="command:node system.js", timeout_s=60), base_dir=str(tmp_path)
    )
    out = system.run(Case(id="c1", input={"question": "What is 2 + 2?"}))
    assert out.output == "The answer is 4."
    assert out.retrieved == ["doc-1"]
    assert (out.tokens_in, out.tokens_out) == (9, 3)
    assert out.trace is not None
    assert [(c.name, c.args, c.result) for c in out.trace.tool_calls] == [
        ("calc", {"expr": "2+2"}, "4")
    ]


def test_a_failing_handler_is_a_case_error(tmp_path: Path) -> None:
    throw = "() => { throw new Error('model unavailable'); }"
    (tmp_path / "system.js").write_text(
        f"require({json.dumps(LIB)}).runCommand({throw});", encoding="utf-8"
    )
    system = CommandSystem(
        TargetConfig(target="command:node system.js", timeout_s=60), base_dir=str(tmp_path)
    )
    out = system.run(Case(id="c1", input="hi"))
    assert out.error is not None
    assert "exited with code 1" in out.error
    assert "model unavailable" in out.error


def test_serve_speaks_the_http_target_protocol() -> None:
    assert NODE
    script = f"""
    const {{ serve }} = require({json.dumps(LIB)});
    const server = serve(async (request) => ({{
      output: 'echo ' + request.input + ' trial=' + request.trial,
      tokens_in: 4,
      tokens_out: 2,
      trace: {{ responses: ['echo'], rounds: 1 }},
    }}), {{ port: 0 }});
    server.on('listening', () => console.log(server.address().port));
    """
    proc = subprocess.Popen(
        [NODE, "-e", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout
        port = int(proc.stdout.readline().strip())
        system = HttpSystem(
            TargetConfig(target=f"http://127.0.0.1:{port}/", timeout_s=30), trials=2
        )
        try:
            out = system.run(Case(id="c1", input="ping"), trial=1)
        finally:
            system.close()
        assert out.output == "echo ping trial=1"
        assert (out.tokens_in, out.tokens_out) == (4, 2)
        assert out.trace is not None and out.trace.rounds == 1
    finally:
        proc.terminate()
        proc.communicate(timeout=30)
