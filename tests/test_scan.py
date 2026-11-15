"""``evalproof scan``: what a repository's LLM code is and what no suite tests, read from the
files alone. Every repository here is built in ``tmp_path``; nothing is imported or run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalproof import experimental
from evalproof.cli import main
from evalproof.config import ConfigError
from evalproof.scan import ScanReport, render_text, report_json, scan_path
from evalproof.scan.graph import ModuleIndex, reachable
from evalproof.scan.pyscan import analyze_python
from evalproof.scan.tsscan import analyze_typescript

SUITE = """\
name: support
dataset: support.jsonl
system:
  target: {target}
scorers:
{scorers}
"""
EXACT = "  - name: exact\n    type: exact_match\n"
CHECKS = """\
  - name: tools
    type: checks
    params:
      checks:
        - {type: tool_called, name: lookup_order}
"""


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def suite(target: str = "python:app.llm:answer", scorers: str = EXACT, extra: str = "") -> str:
    return SUITE.format(target=target, scorers=scorers) + extra


def cases(n: int, **extra: object) -> str:
    rows = [{"id": f"c{i}", "input": f"question {i}", "reference": "x", **extra} for i in range(n)]
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def kinds(report: ScanReport, kind: str) -> list[tuple[str, str | None]]:
    return [(f.path, f.framework) for f in report.findings if f.kind == kind]


def gap_kinds(report: ScanReport) -> list[str]:
    return [g.kind for g in report.gaps]


# --- model calls -------------------------------------------------------------------------------

MODEL_CALLS = [
    ("anthropic", "import anthropic\nc = anthropic.Anthropic()\nc.messages.create(model='m')\n"),
    (
        "anthropic",
        "from anthropic import AsyncAnthropic\nasync def f(c):\n    await c.messages.stream()\n",
    ),
    ("openai", "import openai\nc = openai.OpenAI()\nc.chat.completions.create(model='m')\n"),
    ("openai", "from openai import OpenAI\nOpenAI().responses.create(model='m')\n"),
    ("litellm", "import litellm\nlitellm.completion(model='m', messages=[])\n"),
    ("litellm", "from litellm import acompletion\nacompletion(model='m')\n"),
    ("google-genai", "from google import genai\nc = genai.Client()\nc.models.generate_content()\n"),
    (
        "google-genai",
        "import google.generativeai as g\nm = g.GenerativeModel('x')\nm.generate_content('hi')\n",
    ),
    ("mistral", "from mistralai import Mistral\nMistral().chat.complete(model='m')\n"),
    ("cohere", "import cohere\nco = cohere.ClientV2()\nco.chat(model='m')\n"),
    ("ollama", "import ollama\nollama.chat(model='m', messages=[])\n"),
    ("langchain", "from langchain_openai import ChatOpenAI\nllm = ChatOpenAI(model='m')\n"),
    ("langchain", "from langchain_anthropic import ChatAnthropic as C\nllm = C()\n"),
    ("langchain", "from langchain.chat_models import init_chat_model\ninit_chat_model('x')\n"),
    (
        "bedrock",
        "import boto3\nc = boto3.client('bedrock-runtime')\nc.converse(modelId='m', messages=[])\n",
    ),
    (
        "bedrock",
        "import boto3\nc = boto3.client('bedrock-runtime')\nc.invoke_model(modelId='m', body='')\n",
    ),
]


@pytest.mark.parametrize(("framework", "source"), MODEL_CALLS)
def test_model_call_is_decided_from_the_imports(framework: str, source: str) -> None:
    info = analyze_python("app.py", source)
    calls = [f for f in info.findings if f.kind == "model_call"]
    assert [f.framework for f in calls] == [framework]
    assert calls[0].line == len(source.splitlines())  # the line of the call


def test_the_same_call_in_a_file_without_the_import_is_not_a_model_call() -> None:
    source = (
        "def f(c):\n    c.messages.create()\n    c.chat.completions.create()\n    c.converse()\n"
    )
    assert analyze_python("app.py", source).findings == []


def test_langchain_templates_and_a_local_name_are_not_chat_models() -> None:
    source = (
        "from langchain_core.prompts import ChatPromptTemplate\n"
        "ChatPromptTemplate.from_messages([])\n"
        "def ChatThing():\n    pass\nChatThing()\n"
    )
    assert analyze_python("app.py", source).findings == []


def test_a_bare_completion_call_counts_only_when_imported_from_the_library() -> None:
    source = "import litellm\ndef completion():\n    pass\ncompletion()\n"
    assert analyze_python("app.py", source).findings == []


def test_model_call_detail_names_the_model_literal() -> None:
    info = analyze_python(
        "a.py", "import anthropic\nanthropic.Anthropic().messages.create(model='claude-x')\n"
    )
    assert "model='claude-x'" in info.findings[0].detail


# --- agents ------------------------------------------------------------------------------------

AGENTS = [
    ("openai-agents", "from agents import Agent\nAgent(name='a', instructions='x')\n"),
    ("adk", "from google.adk.agents import LlmAgent\nLlmAgent(name='a', model='m')\n"),
    ("adk", "from google.adk import agents\nagents.Agent(name='a')\n"),
    ("pydantic_ai", "from pydantic_ai import Agent\nAgent('openai:gpt-4o')\n"),
    ("crewai", "from crewai import Agent, Crew\nCrew(agents=[])\n"),
    ("langgraph", "from langgraph.prebuilt import create_react_agent\ncreate_react_agent(m, [])\n"),
    ("langgraph", "from langgraph.graph import StateGraph\nStateGraph(dict)\n"),
    ("langchain", "from langchain.agents import create_agent\ncreate_agent(m, [])\n"),
    (
        "autogen",
        "from autogen_agentchat.agents import AssistantAgent\n"
        "AssistantAgent('a', model_client=m)\n",
    ),
    ("llama_index", "from llama_index.core.agent import ReActAgent\nReActAgent.from_tools([])\n"),
    ("dspy", "import dspy\ndspy.ReAct('q -> a', tools=[])\n"),
    ("haystack", "from haystack.components.agents import Agent\nAgent(chat_generator=g)\n"),
    (
        "semantic_kernel",
        "from semantic_kernel.agents import ChatCompletionAgent\nChatCompletionAgent()\n",
    ),
    ("claude_agent_sdk", "from claude_agent_sdk import query\nquery(prompt='x')\n"),
    ("claude_agent_sdk", "import claude_agent_sdk as sdk\nsdk.ClaudeSDKClient()\n"),
]


@pytest.mark.parametrize(("framework", "source"), AGENTS)
def test_agent_constructors(framework: str, source: str) -> None:
    info = analyze_python("app.py", source)
    assert [f.framework for f in info.findings if f.kind == "agent"] == [framework]


def test_an_agent_named_like_a_framework_but_defined_locally_is_not_one() -> None:
    assert analyze_python("a.py", "class Agent:\n    pass\nAgent()\n").findings == []


# --- tools -------------------------------------------------------------------------------------


def test_tools_from_decorators_take_the_name_override_or_the_function_name() -> None:
    source = (
        "from agents import function_tool\n"
        "from langchain_core.tools import tool\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('x')\n"
        "@function_tool\n"
        "def lookup(order_id: str) -> str:\n"
        '    """Find an order.\n\n    More."""\n'
        "@function_tool(name_override='refund_order')\n"
        "def refund(order_id: str) -> str:\n    pass\n"
        "@tool('search_docs')\n"
        "def search(q: str) -> str:\n    pass\n"
        "@mcp.tool()\n"
        "async def weather(city: str) -> str:\n    pass\n"
    )
    tools = [f for f in analyze_python("t.py", source).findings if f.kind == "tool"]
    assert [f.extra["name"] for f in tools] == ["lookup", "refund_order", "search_docs", "weather"]
    assert tools[0].extra["signature"] == "lookup(order_id: str) -> str"
    assert tools[0].extra["doc"].startswith("Find an order.")
    assert tools[0].detail == "lookup: Find an order."
    assert tools[2].framework == "langchain"


def test_agent_tool_decorators_need_an_agent_framework_import() -> None:
    plain = "app = App()\n@app.tool\ndef f():\n    pass\n"
    assert analyze_python("a.py", plain).findings == []
    agent = (
        "from pydantic_ai import Agent\nagent = Agent('m')\n@agent.tool\ndef f(ctx):\n    pass\n"
    )
    agent += "@agent.tool_plain\ndef g():\n    pass\n"
    tools = [f for f in analyze_python("a.py", agent).findings if f.kind == "tool"]
    assert [f.extra["name"] for f in tools] == ["f", "g"]


def test_kernel_function_and_a_locally_defined_tool_decorator() -> None:
    source = (
        "from semantic_kernel.functions import kernel_function\n"
        "def tool(fn):\n    return fn\n"
        "class P:\n    @kernel_function(name='add')\n    def add(self, a, b):\n        pass\n"
        "@tool\ndef mine():\n    pass\n"
    )
    tools = [f for f in analyze_python("a.py", source).findings if f.kind == "tool"]
    assert [f.extra["name"] for f in tools] == ["add"]


def test_function_tool_constructors() -> None:
    source = (
        "from google.adk.tools import FunctionTool\n"
        "from llama_index.core.tools import FunctionTool as FT\n"
        "def a():\n    '''Doc.'''\n"
        "def b():\n    pass\n"
        "FunctionTool(a)\n"
        "FT.from_defaults(b, name='bee')\n"
    )
    tools = [f for f in analyze_python("a.py", source).findings if f.kind == "tool"]
    assert [f.extra["name"] for f in tools] == ["a", "bee"]
    assert tools[0].extra["doc"] == "Doc."


def test_a_tool_listed_in_an_agent_is_the_tool_that_defines_it() -> None:
    source = (
        "from agents import Agent, function_tool\n"
        "@function_tool\ndef lookup():\n    pass\n"
        "Agent(name='a', tools=[lookup, other])\n"
    )
    tools = [f for f in analyze_python("a.py", source).findings if f.kind == "tool"]
    assert [(f.extra["name"], "listed" in f.extra) for f in tools] == [
        ("lookup", False),
        ("other", True),
    ]


def test_a_listed_tool_defined_in_another_file_is_one_tool_in_the_report(tmp_path: Path) -> None:
    root = write(
        tmp_path,
        {
            "tools.py": (
                "from agents import function_tool\n@function_tool\ndef lookup():\n    pass\n"
            ),
            "agent.py": (
                "from agents import Agent\nfrom tools import lookup, other\n"
                "Agent(name='a', tools=[lookup, other, other])\n"
            ),
        },
    )
    report = scan_path(root)
    tools = [(f.path, f.extra["name"]) for f in report.findings if f.kind == "tool"]
    assert tools == [("agent.py", "other"), ("tools.py", "lookup")]
    assert all("listed" not in f.extra for f in report.findings)


# --- prompts and multi-turn --------------------------------------------------------------------

LONG = "You are a careful support assistant. Answer only from the order data you are given."


def test_prompts_from_module_names_and_keywords() -> None:
    source = (
        "import anthropic\nfrom textwrap import dedent\n"
        f"SYSTEM_PROMPT = '{LONG}'\n"
        "SHORT_PROMPT = 'too short'\n"
        f"INSTRUCTIONS = dedent('''{LONG}''')\n"
        "OTHER = 'x' * 3\n"
        f"anthropic.Anthropic().messages.create(model='m', system='{LONG}')\n"
        f"f(instructions=f'{LONG} {{name}}')\n"
    )
    info = analyze_python("p.py", source)
    prompts = [(f.line, f.extra["name"]) for f in info.findings if f.kind == "prompt"]
    assert prompts == [
        (3, "SYSTEM_PROMPT"),
        (5, "INSTRUCTIONS"),
        (7, "system"),
        (8, "instructions"),
    ]
    assert info.findings[0].extra["excerpt"].startswith("You are a careful")


def test_prompt_keywords_count_only_in_a_file_that_calls_a_model_or_agent() -> None:
    source = f"configure(system='{LONG}', instructions='{LONG}')\n"
    assert analyze_python("c.py", source).findings == []


def test_prompt_files_under_a_prompts_directory(tmp_path: Path) -> None:
    root = write(
        tmp_path,
        {
            "prompts/system.md": "Be brief.",
            "prompts/nested/ask.j2": "Question: {{ q }}",
            "prompts/empty.txt": "  \n",
            "prompts/data.bin": "x",
            "docs/prompts.md": "not in a prompts directory",
        },
    )
    report = scan_path(root)
    assert sorted(p for p, _ in kinds(report, "prompt")) == [
        "prompts/nested/ask.j2",
        "prompts/system.md",
    ]
    assert report.findings[0].extra["excerpt"]


def test_multi_turn_evidence() -> None:
    source = (
        "import anthropic\n"
        "from langgraph.checkpoint.memory import MemorySaver\n"
        "saver = MemorySaver()\n"
        "history = []\n"
        "history.append({'role': 'user', 'content': 'hi'})\n"
        "history.append({'role': 'assistant', 'content': 'yo'})\n"
        "agent.run(q, message_history=history)\n"
        "chat = model.start_chat()\n"
        "cfg = {'configurable': {'thread_id': '1'}}\n"
        "anthropic.Anthropic().messages.create(model='m')\n"
    )
    info = analyze_python("m.py", source)
    turns = [f.detail for f in info.findings if f.kind == "multi_turn"]
    assert len(turns) == 5  # the repeated append is one
    assert any("MemorySaver" in t for t in turns)
    assert any("message_history=" in t for t in turns)
    assert any("start_chat" in t for t in turns)
    assert any("thread_id" in t for t in turns)
    assert any("grows" in t for t in turns)


def test_generic_multi_turn_evidence_needs_a_model_call_in_the_file() -> None:
    source = "h = []\nh.append({'role': 'user'})\nrun(history=h)\n"
    assert analyze_python("m.py", source).findings == []
    named = "from langchain.memory import ConversationBufferMemory\nConversationBufferMemory()\n"
    assert [f.kind for f in analyze_python("m.py", named).findings] == ["multi_turn"]


# --- TypeScript and JavaScript -----------------------------------------------------------------


def test_typescript_calls_and_imports() -> None:
    source = (
        "import OpenAI from 'openai';\n"
        "import Anthropic from '@anthropic-ai/sdk'\n"
        "import { generateText } from 'ai';\n"
        "import { ChatOpenAI } from '@langchain/openai';\n"
        "import { GoogleGenAI } from '@google/genai';\n"
        "// await client.messages.create({})\n"
        "const a = await openai.chat.completions.create({});\n"
        "const b = await anthropic.messages.create({});\n"
        "const c = await generateText({ model });\n"
        "const d = new ChatOpenAI({});\n"
    )
    found = {(f.framework, f.line) for f in analyze_typescript("a.ts", source)}
    assert found == {
        ("openai", 7),
        ("anthropic", 8),
        ("vercel-ai", 9),
        ("langchain", 10),
        ("google-genai", 5),  # imported, no call found: reported at the import
    }
    assert all(f.language == "typescript" for f in analyze_typescript("a.ts", source))


def test_typescript_agent_frameworks() -> None:
    source = (
        "const { Agent } = require('@openai/agents');\n"
        "import { Agent as M } from '@mastra/core/agent';\n"
        "import { createReactAgent } from '@langchain/langgraph/prebuilt';\n"
        "const a = new Agent({ name: 'a' });\n"
        "const r = createReactAgent({ llm });\n"
    )
    found = {(f.kind, f.framework) for f in analyze_typescript("a.js", source)}
    assert ("agent", "openai-agents") in found
    assert ("agent", "mastra") in found
    assert ("agent", "langchain") in found


def test_typescript_files_in_a_scan_are_marked_and_skip_declarations(tmp_path: Path) -> None:
    root = write(
        tmp_path,
        {
            "src/chat.ts": "import OpenAI from 'openai';\nawait c.responses.create({});\n",
            "src/types.d.ts": "import OpenAI from 'openai';\n",
            "src/plain.mjs": "export const x = 1;\n",
        },
    )
    report = scan_path(root)
    assert [(f.path, f.line, f.language) for f in report.findings] == [
        ("src/chat.ts", 2, "typescript")
    ]
    # TypeScript findings never become uncovered_code: no python target can reach them.
    assert "uncovered_code" not in gap_kinds(report)


# --- the import graph --------------------------------------------------------------------------


def test_imports_resolve_relative_absolute_and_src_layout() -> None:
    files = {
        "src/pkg/__init__.py": "from .agent import run\n",
        "src/pkg/agent.py": "from . import tools\nfrom .llm import ask\n",
        "src/pkg/tools.py": "import os\n",
        "src/pkg/llm.py": "from pkg.util import clean\n",
        "src/pkg/util.py": "",
        "scripts/job.py": "from helper import go\nimport pkg.agent as a\n",
        "scripts/helper.py": "",
    }
    infos = {p: analyze_python(p, src) for p, src in files.items()}
    index = ModuleIndex(infos)
    edges = {p: index.edges(i) for p, i in infos.items()}
    assert edges["src/pkg/agent.py"] == {
        "src/pkg/tools.py",
        "src/pkg/llm.py",
        "src/pkg/__init__.py",
    }
    assert edges["scripts/job.py"] == {
        "scripts/helper.py",
        "src/pkg/agent.py",
        "src/pkg/__init__.py",
    }
    assert reachable("scripts/job.py", edges) == [
        "scripts/helper.py",
        "scripts/job.py",
        "src/pkg/__init__.py",
        "src/pkg/agent.py",
        "src/pkg/llm.py",
        "src/pkg/tools.py",
        "src/pkg/util.py",
    ]
    assert index.resolve("pkg.agent") == "src/pkg/agent.py"
    assert index.resolve("os") is None


def test_a_relative_import_above_the_package_resolves_to_nothing() -> None:
    files = {"a/__init__.py": "", "a/b.py": "from ... import x\nfrom .. import y\n", "y.py": ""}
    infos = {p: analyze_python(p, src) for p, src in files.items()}
    index = ModuleIndex(infos)
    assert index.edges(infos["a/b.py"]) == {"a/__init__.py"}  # never y.py


# --- suites and gaps ---------------------------------------------------------------------------

LLM = (
    "import anthropic\n"
    "client = anthropic.Anthropic()\n"
    "def answer(q):\n"
    "    return client.messages.create(model='m', messages=[{'role': 'user', 'content': q}])\n"
)


def shop(tmp_path: Path, **overrides: str) -> Path:
    files = {
        "app/__init__.py": "",
        "app/llm.py": LLM,
        "evals/support.yaml": suite(),
        "evals/support.jsonl": cases(3),
    }
    files.update(overrides)
    return write(tmp_path, {k: v for k, v in files.items() if v is not None})


def test_a_suite_is_found_with_its_dataset_cases_and_reach(tmp_path: Path) -> None:
    report = scan_path(shop(tmp_path))
    [s] = report.suites
    assert (s.path, s.name, s.dataset, s.cases) == (
        "evals/support.yaml",
        "support",
        "evals/support.jsonl",
        3,
    )
    assert s.target == "python:app.llm:answer"
    assert s.reachability == "known"
    assert s.reaches == ["app/__init__.py", "app/llm.py"]
    assert s.min_cases == 20
    assert kinds(report, "suite") == [("evals/support.yaml", None)]
    assert "uncovered_code" not in gap_kinds(report)  # app/llm.py is what the suite runs


def test_a_suite_that_load_config_rejects_is_counted_and_explained(tmp_path: Path) -> None:
    report = scan_path(shop(tmp_path, **{"evals/bad.yaml": suite() + "bogus: 1\n"}))
    assert [s.path for s in report.suites] == ["evals/support.yaml"]
    assert report.invalid_suites == 1
    assert any("evals/bad.yaml" in n and "bogus" in n for n in report.notes)


def test_yaml_that_is_not_a_suite_is_ignored(tmp_path: Path) -> None:
    other = "scorers: nothing\nsystem: also nothing\nversion: [1, 2\n"  # not even valid YAML
    report = scan_path(shop(tmp_path, **{"ci.yml": "on: push\n", "x.yaml": other}))
    assert report.invalid_suites == 0
    assert len(report.suites) == 1


def test_no_suite_is_a_gap_and_every_code_file_is_uncovered(tmp_path: Path) -> None:
    report = scan_path(shop(tmp_path, **{"evals/support.yaml": None, "evals/support.jsonl": None}))
    assert report.suites == []
    assert gap_kinds(report) == ["no_suite", "uncovered_code"]
    assert report.gaps[0].suggest == "evalproof init"
    assert report.gaps[1].path == "app/llm.py"
    assert report.gaps[1].line == 4
    assert "--file app/llm.py" in report.gaps[1].suggest
    assert "--target MODEL" in report.gaps[1].suggest  # no suite, so no judge to default to


def test_code_no_suite_target_reaches_is_uncovered(tmp_path: Path) -> None:
    root = shop(tmp_path, **{"jobs/batch.py": LLM, "tests/test_llm.py": LLM})
    gaps = [g for g in scan_path(root).gaps if g.kind == "uncovered_code"]
    assert [g.path for g in gaps] == ["jobs/batch.py"]  # test files are not code to cover


def test_reach_follows_imports_through_other_files(tmp_path: Path) -> None:
    root = shop(
        tmp_path,
        **{
            "evals/support.yaml": suite("python:app.agent:run"),
            "app/agent.py": "from .llm import answer\ndef run(q):\n    return answer(q)\n",
        },
    )
    report = scan_path(root)
    assert report.suites[0].reaches == ["app/__init__.py", "app/agent.py", "app/llm.py"]
    assert "uncovered_code" not in gap_kinds(report)


@pytest.mark.parametrize(
    "target", ["http://localhost:8000/ask", "command:python run.py", "anthropic:claude-x"]
)
def test_a_target_that_is_not_python_makes_reachability_unknown(
    target: str, tmp_path: Path
) -> None:
    root = shop(tmp_path, **{"evals/support.yaml": suite(target), "jobs/batch.py": LLM})
    report = scan_path(root)
    assert report.suites[0].reachability == "unknown"
    assert "not a python target" in str(report.suites[0].reason)
    assert "uncovered_code" not in gap_kinds(report)
    assert "not known" in report.not_computed["uncovered_code"]
    assert "not computed" in render_text(report)


def test_a_python_target_outside_the_repository_is_unknown_too(tmp_path: Path) -> None:
    root = shop(tmp_path, **{"evals/support.yaml": suite("python:elsewhere.mod:fn")})
    report = scan_path(root)
    assert report.suites[0].reachability == "unknown"
    assert "elsewhere.mod" in report.not_computed["uncovered_code"]


TOOLS = (
    "from agents import Agent, function_tool\n"
    "@function_tool\ndef lookup_order(order_id: str) -> str:\n    '''Find an order.'''\n"
    "@function_tool\ndef refund(order_id: str) -> str:\n    '''Refund it.'''\n"
)


def test_tool_unchecked_names_the_tools_no_check_mentions(tmp_path: Path) -> None:
    root = shop(
        tmp_path,
        **{"app/tools.py": TOOLS, "evals/support.yaml": suite("python:app.tools:x", CHECKS)},
    )
    gaps = [g for g in scan_path(root).gaps if g.kind == "tool_unchecked"]
    assert [(g.extra["tool"], g.path, g.line) for g in gaps] == [("refund", "app/tools.py", 6)]
    assert "--gap tool_unchecked --tool refund" in gaps[0].suggest


def test_a_check_in_a_case_or_inside_any_of_counts(tmp_path: Path) -> None:
    case = {
        "id": "c0",
        "input": "q",
        "expect": [
            {
                "type": "any_of",
                "checks": [
                    {"type": "tool_called", "name": "refund"},
                    {"type": "not", "check": {"type": "tool_called", "name": "other"}},
                ],
            },
            {"type": "tool_order", "calls": ["lookup_order", {"name": "refund"}]},
        ],
    }
    root = shop(
        tmp_path,
        **{
            "app/tools.py": TOOLS,
            "evals/support.yaml": suite("python:app.tools:x", "  - name: t\n    type: checks\n"),
            "evals/support.jsonl": json.dumps(case) + "\n",
        },
    )
    assert "tool_unchecked" not in gap_kinds(scan_path(root))


def test_an_unreadable_dataset_means_the_case_gaps_are_not_computed(tmp_path: Path) -> None:
    root = shop(tmp_path, **{"app/tools.py": TOOLS, "evals/support.jsonl": "not json\n"})
    report = scan_path(root)
    assert report.suites[0].cases is None
    assert "tool_unchecked" in report.not_computed
    assert "tool_unchecked" not in gap_kinds(report)
    assert any("could not be read" in n for n in report.notes)


MULTI = "import anthropic\nfrom langgraph.checkpoint.memory import MemorySaver\nMemorySaver()\n"


def test_no_multi_turn_cases_when_the_code_keeps_a_conversation(tmp_path: Path) -> None:
    gap = [g for g in scan_path(shop(tmp_path, **{"app/chat.py": MULTI})).gaps]
    assert "no_multi_turn_cases" in [g.kind for g in gap]
    one = next(g for g in gap if g.kind == "no_multi_turn_cases")
    assert (one.path, one.line) == ("app/chat.py", 3)


def test_a_case_with_turns_closes_the_multi_turn_gap(tmp_path: Path) -> None:
    root = shop(
        tmp_path,
        **{"app/chat.py": MULTI, "evals/support.jsonl": cases(2, turns=["and then?"])},
    )
    assert "no_multi_turn_cases" not in gap_kinds(scan_path(root))


def test_few_cases_compares_with_the_gate_minimum(tmp_path: Path) -> None:
    few = [g for g in scan_path(shop(tmp_path)).gaps if g.kind == "few_cases"]
    assert [(g.extra["cases"], g.extra["min_cases"]) for g in few] == [(3, 20)]
    assert "--gap few_cases --suite evals/support.yaml" in few[0].suggest
    gate = "gate:\n  metric: exact\n  min_cases: 3\n"
    ok = shop(tmp_path / "ok", **{"evals/support.yaml": suite(extra=gate)})
    assert "few_cases" not in gap_kinds(scan_path(ok))


def test_the_suggested_draft_command_names_the_scan_file_and_the_judge(tmp_path: Path) -> None:
    judge = "judge:\n  target: ollama:qwen3\n"
    root = shop(
        tmp_path,
        **{"app/tools.py": TOOLS, "evals/support.yaml": suite("python:app.tools:x", extra=judge)},
    )
    gap = next(
        g for g in scan_path(root, scan_ref="out/scan.json").gaps if g.kind == "tool_unchecked"
    )
    assert gap.suggest.startswith(
        "evalproof draft --scan out/scan.json --out evals/drafts.jsonl --gap"
    )
    assert "--target" not in gap.suggest


# --- test cases --------------------------------------------------------------------------------


def test_test_cases_in_other_formats_come_with_the_import_command(tmp_path: Path) -> None:
    root = write(
        tmp_path,
        {
            "data/qa.jsonl": json.dumps({"input": "q", "expected_output": "a"}) + "\n",
            "data/rag.jsonl": json.dumps({"question": "q", "ground_truth": "a"}) + "\n",
            "data/both.jsonl": json.dumps({"input": "q", "target": "a", "expected": "b"}) + "\n",
            "data/a b.evalset.json": json.dumps({"eval_cases": []}),
            "promptfooconfig.yaml": "prompts: [x]\n",
            "package.json": json.dumps({"name": "x", "version": "1"}),
            "data/notes.json": "{ not json",
        },
    )
    report = scan_path(root)
    found = {f.path: f for f in report.findings if f.kind == "test_cases"}
    assert set(found) == {
        "data/qa.jsonl",
        "data/rag.jsonl",
        "data/both.jsonl",
        "data/a b.evalset.json",
        "promptfooconfig.yaml",
    }
    assert found["data/qa.jsonl"].extra["command"] == (
        "evalproof import-dataset deepeval data/qa.jsonl --out evals/qa.jsonl"
    )
    assert found["data/rag.jsonl"].framework == "ragas"
    assert found["data/a b.evalset.json"].extra["command"] == (
        "evalproof import-dataset adk 'data/a b.evalset.json' --out evals/a_b.jsonl"
    )
    assert found["promptfooconfig.yaml"].extra["format"] == "promptfoo"
    ambiguous = found["data/both.jsonl"]
    assert ambiguous.extra["format"] is None
    assert ambiguous.extra["candidates"] == ["inspect", "braintrust"]
    gaps = {g.path: g for g in report.gaps if g.kind == "test_cases_not_imported"}
    assert set(gaps) == set(found)
    assert gaps["data/qa.jsonl"].suggest == found["data/qa.jsonl"].extra["command"]
    assert "<format>" in gaps["data/both.jsonl"].suggest


def test_a_suites_own_dataset_is_not_reported_as_test_cases(tmp_path: Path) -> None:
    row = json.dumps({"id": "a", "input": "q", "expected_output": "a"}) + "\n"
    root = shop(tmp_path, **{"evals/support.jsonl": row})
    assert kinds(scan_path(root), "test_cases") == []


def test_pytest_files_over_code_with_a_model_call(tmp_path: Path) -> None:
    test = (
        "import app.llm\n"
        "import app.agent as agent_mod\n"
        "from app.llm import answer\n"
        "from app import llm\n"
        "import json\n"
        "def test_a():\n    assert 'x' in answer('q')\n"
    )
    root = shop(
        tmp_path,
        **{
            "app/agent.py": "from .llm import answer\n",
            "tests/test_llm.py": test,
            "tests/test_unrelated.py": "def test_b():\n    assert 1\n",
            "tests/test_agent_only.py": "from app.agent import answer\n",
        },
    )
    report = scan_path(root)
    found = [f for f in report.findings if f.kind == "test_cases"]
    assert [f.path for f in found] == ["tests/test_agent_only.py", "tests/test_llm.py"]
    pytest_file = next(f for f in found if f.path == "tests/test_llm.py")
    assert pytest_file.extra["format"] == "pytest"
    assert pytest_file.extra["imports"] == {
        "app.llm": "app/llm.py",
        "answer": "app/llm.py::answer",
        "llm": "app/llm.py",
        "agent_mod": "app/agent.py",
    }
    assert "--from-tests-only" in pytest_file.extra["command"]
    assert any(
        g.kind == "test_cases_not_imported" and g.path == "tests/test_llm.py" for g in report.gaps
    )


# --- walking, skipping and failing safely ------------------------------------------------------


def test_skipped_directories_and_excludes(tmp_path: Path) -> None:
    root = write(
        tmp_path,
        {
            "app/llm.py": LLM,
            "node_modules/x/llm.py": LLM,
            ".venv/lib/llm.py": LLM,
            "venv/llm.py": LLM,
            "env/llm.py": LLM,
            "build/llm.py": LLM,
            "dist/llm.py": LLM,
            ".hidden/llm.py": LLM,
            "x/__pycache__/llm.py": LLM,
            "lib/site-packages/llm.py": LLM,
            "pkg.egg-info/llm.py": LLM,
            "vendor/llm.py": LLM,
            "scratch/old_llm.py": LLM,
        },
    )
    assert {p for p, _ in kinds(scan_path(root), "model_call")} == {
        "app/llm.py",
        "vendor/llm.py",
        "scratch/old_llm.py",
    }
    excluded = scan_path(root, excludes=["vendor", "scratch/*", "./nothing/"])
    assert {p for p, _ in kinds(excluded, "model_call")} == {"app/llm.py"}
    by_name = scan_path(root, excludes=["old_*.py", "vendor/"])
    assert {p for p, _ in kinds(by_name, "model_call")} == {"app/llm.py"}


def test_unparsable_and_oversized_files_are_counted_and_skipped(tmp_path: Path) -> None:
    root = write(
        tmp_path,
        {
            "app/llm.py": LLM,
            "app/broken.py": "def (:\n",
            "app/nul.py": "x = 1\x00\n",
            "app/huge.py": "#" * 1_000_001,
        },
    )
    (root / "app" / "latin.py").write_bytes(b"# \xff\xfe\n")
    report = scan_path(root)
    assert (report.unparsable, report.too_large) == (3, 1)
    assert kinds(report, "model_call") == [("app/llm.py", "anthropic")]
    assert "3 file(s) that did not parse" in render_text(report)
    assert "1 file(s) over 1 MB" in render_text(report)


def test_a_missing_or_file_path_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no such directory"):
        scan_path(tmp_path / "nope")
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a file"):
        scan_path(tmp_path / "f.py")


def test_an_empty_repository_reports_nothing_and_is_not_an_error(tmp_path: Path) -> None:
    report = scan_path(tmp_path)
    assert report.findings == [] and report.gaps == []
    assert "Nothing found" in render_text(report)


def test_the_scan_never_imports_the_code_it_reads(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    root = write(
        tmp_path / "repo",
        {
            "app/llm.py": LLM + f"open({str(marker)!r}, 'w').write('x')\n",
            "evals/support.yaml": suite(),
            "evals/support.jsonl": cases(1),
        },
    )
    scan_path(root)
    assert not marker.exists()


# --- output ------------------------------------------------------------------------------------


def test_the_json_has_the_documented_shape(tmp_path: Path) -> None:
    report = scan_path(shop(tmp_path, **{"app/tools.py": TOOLS}))
    data = json.loads(report_json(report))
    assert set(data) == {
        "format_version",
        "root",
        "findings",
        "suites",
        "gaps",
        "not_computed",
        "notes",
        "skipped",
    }
    assert data["format_version"] == 1
    assert data["skipped"] == {"unparsable": 0, "too_large": 0, "invalid_suites": 0}
    finding = next(f for f in data["findings"] if f["kind"] == "tool")
    assert set(finding) >= {"kind", "path", "line", "framework", "detail", "language", "name"}
    assert finding["path"] == "app/tools.py" and "\\" not in finding["path"]
    assert set(data["suites"][0]) == {
        "path",
        "name",
        "dataset",
        "cases",
        "target",
        "judge",
        "min_cases",
        "reachability",
        "reaches",
        "reason",
    }
    assert set(data["gaps"][0]) == {"kind", "path", "line", "message", "suggest"} | set(
        data["gaps"][0]
    ) - {"tool", "suite", "cases", "min_cases", "format"}


def test_the_text_groups_by_kind_and_caps_each_group(tmp_path: Path) -> None:
    files = {f"prompts/p{i:02}.txt": f"prompt {i}" for i in range(25)}
    files["app/llm.py"] = LLM
    text = render_text(scan_path(write(tmp_path, files)))
    assert text.startswith("evalproof scan: ")
    assert "model_call (1)\n  app/llm.py:4  [anthropic]  client.messages.create(model='m')" in text
    assert "prompt (25)" in text
    assert "  ... and 5 more" in text
    assert text.count("prompts/p") == 20
    assert "gaps (" in text and "next: evalproof draft" in text


def test_the_text_says_when_there_are_no_gaps(tmp_path: Path) -> None:
    gate = "gate:\n  metric: exact\n  min_cases: 1\n"
    root = shop(tmp_path, **{"evals/support.yaml": suite(extra=gate)})
    assert "gaps (0)\n  none found" in render_text(scan_path(root))


def test_suite_flags_replace_discovery(tmp_path: Path) -> None:
    root = shop(tmp_path, **{"other/also.yaml": suite().replace("name: support", "name: also")})
    assert len(scan_path(root).suites) == 2
    only = scan_path(root, suite_files=[root / "other" / "also.yaml"])
    assert [s.name for s in only.suites] == ["also"]
    with pytest.raises(ConfigError, match="not found"):
        scan_path(root, suite_files=[root / "missing.yaml"])


# --- the command -------------------------------------------------------------------------------


def test_scan_prints_text_and_exits_zero_with_gaps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["scan", str(shop(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "model_call (1)" in out and "few_cases" in out


def test_scan_json_and_out(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = shop(tmp_path / "repo")
    out_file = tmp_path / "reports" / "scan.json"
    assert main(["scan", str(root), "--json", "--out", str(out_file)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == json.loads(out_file.read_text(encoding="utf-8"))
    assert f"wrote {out_file}" in captured.err
    gap = next(g for g in json.loads(captured.out)["gaps"] if g["kind"] == "few_cases")
    assert f"--scan {out_file}" in gap["suggest"]


def test_scan_defaults_to_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(shop(tmp_path))
    assert main(["scan", "--json", "--exclude", "evals"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["root"] == "." and data["suites"] == []


@pytest.mark.parametrize("bad", ["missing-dir", "FILE"])
def test_scan_usage_errors_exit_two(
    bad: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "f.py"
    target.write_text("x = 1\n", encoding="utf-8")
    path = str(target) if bad == "FILE" else str(tmp_path / bad)
    assert main(["scan", path]) == 2
    assert "evalproof: error:" in capsys.readouterr().err


def test_scan_with_an_unwritable_out_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["scan", str(tmp_path), "--out", str(tmp_path)]) == 2
    assert "cannot write the scan file" in capsys.readouterr().err


def test_scan_says_once_that_it_is_experimental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experimental, "_noticed", set())
    main(["scan", str(tmp_path)])
    first = capsys.readouterr().err
    main(["scan", str(tmp_path)])
    second = capsys.readouterr().err
    assert first.startswith("evalproof: scan is experimental: ")
    assert first.count("experimental") == 1
    assert second == ""
