"""``evalproof scan`` on TypeScript and JavaScript: the tools and agents it finds, and which files a
``command:`` target that starts node, tsx, bun or deno reaches. Every repository is built in
``tmp_path`` from fixture text; nothing is run and no model is called."""

from __future__ import annotations

import json
import posixpath
import shlex
from pathlib import Path

import pytest

from evalproof.draft.ask import code_request, tool_request
from evalproof.draft.scanfile import load_scan
from evalproof.scan import ScanReport, render_text, report_json, scan_path
from evalproof.scan.tsgraph import TsGraph, node_command
from evalproof.scan.tsscan import analyze_typescript, read_typescript

SUITE = """\
name: support
dataset: support.jsonl
system:
  target: "{target}"
scorers:
  - name: exact
    type: exact_match
"""
SUITE_WITH_CHECKS = SUITE + (
    "  - name: tools\n    type: checks\n    params:\n      checks:\n"
    "        - {type: tool_called, name: lookup_order}\n"
)
CASES = "".join(
    json.dumps({"id": f"c{i}", "input": f"question {i}", "reference": "x"}) + "\n" for i in range(3)
)
SYSTEM = (
    'import { runCommand } from "evalproof";\n'
    'import { answer } from "../src/agent.js";\n'
    "await runCommand(({ input }) => answer(input));\n"
)
AGENT = (
    'import { generateText } from "ai";\n'
    "export async function answer(q) {\n"
    "  return generateText({ model, prompt: q });\n"
    "}\n"
)
OPENAI = (
    'import OpenAI from "openai";\n'
    "export const ask = (c, q) => c.chat.completions.create({ model: 'm', messages: q });\n"
)


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def repo(
    tmp_path: Path,
    files: dict[str, str],
    target: str = "command:node system.mjs",
    suite: str = SUITE,
) -> ScanReport:
    base = {
        "evals/support.yaml": suite.replace("{target}", target),
        "evals/support.jsonl": CASES,
    }
    return scan_path(write(tmp_path, {**base, **files}))


def gap_paths(report: ScanReport, kind: str) -> list[str | None]:
    return [g.path for g in report.gaps if g.kind == kind]


def found(source: str, kind: str, path: str = "a.ts") -> list[tuple[str | None, str, int]]:
    return [
        (f.framework, str(f.extra.get("name", f.detail)), f.line)
        for f in analyze_typescript(path, source)
        if f.kind == kind
    ]


# --- tools --------------------------------------------------------------------------------------


def test_a_vercel_tool_is_named_by_its_key_in_tools() -> None:
    source = (
        'import { generateText, tool } from "ai";\n'
        'import { z } from "zod";\n'
        "const result = await generateText({\n"
        "  model,\n"
        "  tools: {\n"
        "    lookupOrder: tool({\n"
        "      description: 'Find an order by its id',\n"
        "      inputSchema: z.object({ name: z.string().describe('name: of the customer') }),\n"
        "      execute: async ({ id }) => db.find(id),\n"
        "    }),\n"
        '    "send-refund": tool({ description: "Refund it", inputSchema: z.object({}) }),\n'
        "  },\n"
        "});\n"
    )
    tools = [f for f in analyze_typescript("a.ts", source) if f.kind == "tool"]
    assert [(f.extra["name"], f.line, f.framework, f.language) for f in tools] == [
        ("lookupOrder", 6, "vercel-ai", "typescript"),
        ("send-refund", 11, "vercel-ai", "typescript"),
    ]
    assert tools[0].extra["doc"] == "Find an order by its id"
    assert tools[0].detail == "lookupOrder: Find an order by its id"
    assert tools[1].extra["doc"] == "Refund it"


def test_a_vercel_tool_kept_in_a_variable_takes_the_key_it_is_listed_under() -> None:
    source = (
        'import { tool as defineTool, generateText } from "ai";\n'
        "export const getWeather = defineTool({ description: 'Weather for a city' });\n"
        "export const getTime = defineTool({ description: 'Time in a city' });\n"
        "await generateText({ model, tools: { weather: getWeather, getTime } });\n"
    )
    assert found(source, "tool") == [
        ("vercel-ai", "weather", 2),  # listed as weather: that is the name the model sees
        ("vercel-ai", "getTime", 3),  # listed by shorthand: the variable's own name
    ]


def test_a_namespace_import_and_a_require_find_the_vercel_tool_too() -> None:
    namespaced = 'import * as ai from "ai";\nconst tools = { a: ai.tool({ description: "A" }) };\n'
    required = 'const { tool } = require("ai");\nconst tools = { b: tool({ description: "B" }) };\n'
    assert found(namespaced, "tool") == [("vercel-ai", "a", 2)]
    assert found(required, "tool", "a.js") == [("vercel-ai", "b", 2)]


def test_a_tool_function_that_is_not_the_sdks_is_not_a_tool() -> None:
    source = (
        'import { generateText } from "ai";\n'
        "function tool(x) { return x; }\n"
        "const t = { a: tool({ description: 'not the sdk' }) };\n"
        'import { tool as lc } from "some-other-package";\n'
        "const u = { b: lc({ description: 'nor this' }) };\n"
    )
    assert found(source, "tool") == []


def test_a_langchain_tool_takes_its_name_and_description_from_its_options() -> None:
    source = (
        'import { tool } from "@langchain/core/tools";\n'
        'import { DynamicStructuredTool } from "@langchain/core/tools";\n'
        "export const search = tool(\n"
        "  async ({ q }) => {\n"
        "    const meta = { name: 'inner', description: 'not this one' };\n"
        "    return run(q, meta);\n"
        "  },\n"
        "  {\n"
        "    name: 'search_docs',\n"
        '    description: "Search the " + "docs",\n'
        "    schema: z.object({ q: z.string() }),\n"
        "  },\n"
        ");\n"
        "export const calc = new DynamicStructuredTool({\n"
        "  name: 'calculator',\n"
        "  description: `Do arithmetic`,\n"
        "  func: async () => '1',\n"
        "});\n"
    )
    tools = [f for f in analyze_typescript("a.ts", source) if f.kind == "tool"]
    assert [(f.extra["name"], f.extra["doc"], f.framework) for f in tools] == [
        ("search_docs", "Search the docs", "langchain"),
        ("calculator", "Do arithmetic", "langchain"),
    ]
    assert [f.line for f in tools] == [3, 14]


def test_langchain_v1_imports_from_the_langchain_package() -> None:
    source = (
        "import {\n  createAgent,\n  tool,\n} from 'langchain';\n"
        "const t = tool(fn, { name: 'ping', description: 'Ping' });\n"
        "const agent = createAgent({ model, tools: [t] });\n"
    )
    infos = analyze_typescript("a.ts", source)
    assert [(f.kind, f.framework, f.line) for f in infos] == [
        ("tool", "langchain", 5),
        ("agent", "langchain", 6),
    ]


def test_a_tool_whose_name_is_not_a_literal_is_not_reported() -> None:
    source = (
        'import { tool } from "@langchain/core/tools";\n'
        "const a = tool(fn, { name: NAME, description: 'x' });\n"
        "const b = tool(fn, { description: 'no name' });\n"
        "const c = tool(fn, { name });\n"
        'import { tool as vtool } from "ai";\n'
        "register(vtool({ description: 'anonymous' }));\n"
    )
    assert found(source, "tool") == []


def test_a_description_that_is_not_a_plain_string_is_left_out() -> None:
    source = (
        'import { tool } from "ai";\n'
        "const t = { a: tool({ description: `Looks up ${kind} orders` }) };\n"
        "const u = { b: tool({ description: DESCRIPTION }) };\n"
    )
    tools = [f for f in analyze_typescript("a.ts", source) if f.kind == "tool"]
    assert [(f.extra["name"], "doc" in f.extra) for f in tools] == [("a", False), ("b", False)]
    assert tools[0].detail == "a: no description"


def test_commented_out_code_is_not_read() -> None:
    source = (
        'import { generateText, tool } from "ai";\n'
        "// const t = { a: tool({ description: 'x' }) };\n"
        "/* const u = { b: tool({ description: 'y' }),\n"
        "   stopWhen: stepCountIs(3) } */\n"
        "const url = 'http://example.com'; const t = { c: tool({ description: 'z' }) };\n"
    )
    assert found(source, "tool") == [("vercel-ai", "c", 5)]
    assert found(source, "agent") == []


# --- agents -------------------------------------------------------------------------------------


def test_vercel_agents_are_the_agent_class_and_a_step_limit() -> None:
    source = (
        'import { Experimental_Agent as Agent, generateText, stepCountIs } from "ai";\n'
        "const agent = new Agent({ model, tools, stopWhen: stepCountIs(5) });\n"
        "await generateText({ model, prompt, maxSteps: 3 });\n"
    )
    assert [
        (f.framework, f.line) for f in analyze_typescript("a.ts", source) if f.kind == "agent"
    ] == [
        ("vercel-ai", 2),
        ("vercel-ai", 2),
        ("vercel-ai", 3),
    ]
    details = [f.detail for f in analyze_typescript("a.ts", source) if f.kind == "agent"]
    assert details[1:] == [
        "stopWhen: a multi-step tool loop",
        "maxSteps: a multi-step tool loop",
    ]


def test_an_agent_class_from_another_package_is_not_a_vercel_agent() -> None:
    source = (
        'import { generateText } from "ai";\n'
        'import { Agent } from "./my-agent";\n'
        "const a = new Agent({});\n"
        "await generateText({ model, prompt });\n"
    )
    assert found(source, "agent") == []


def test_langchain_agent_constructors_are_found_under_any_langchain_module() -> None:
    source = (
        'import { createReactAgent } from "@langchain/langgraph/prebuilt";\n'
        "const agent = createReactAgent({ llm, tools });\n"
    )
    assert [(f.kind, f.framework, f.line) for f in analyze_typescript("a.ts", source)] == [
        ("agent", "langchain", 2)
    ]


def test_a_file_that_only_defines_tools_has_no_import_only_finding() -> None:
    source = (
        'import { tool } from "@langchain/core/tools";\nexport const t = tool(f, { name: "t" });\n'
    )
    assert [f.kind for f in analyze_typescript("a.ts", source)] == ["tool"]
    only_import = 'import { HumanMessage } from "@langchain/core/messages";\n'
    assert [f.detail for f in analyze_typescript("a.ts", only_import)] == [
        "imports @langchain/core/messages; no call found"
    ]


# --- what a file imports ------------------------------------------------------------------------


def test_the_modules_a_file_loads_are_read_from_every_import_form() -> None:
    source = (
        "import a from './a';\n"
        "import { b, c } from '../b.js'\n"
        "import {\n  d,\n  e,\n} from './multi/line';\n"
        "import * as f from './f';\n"
        "import './side-effect';\n"
        "export { g } from './g';\n"
        "export * from './h';\n"
        "import type { T } from './types-only';\n"
        "export type { U } from './types-only-too';\n"
        "const i = require('./i');\n"
        "const j = await import('./j');\n"
        "import k = require('./k');\n"
        "// import l from './commented-out';\n"
        "import m from 'openai';\n"
    )
    info = read_typescript("src/x.ts", source)
    assert info.imports == [
        "./a",
        "../b.js",
        "./multi/line",
        "./f",
        "./side-effect",
        "./g",
        "./h",
        "./i",
        "./j",
        "./k",
        "openai",
    ]
    assert info.dynamic is False


@pytest.mark.parametrize(
    "line",
    ["await import(name)", "require(path.join(dir, 'x'))", "await import(`./x/${n}`)"],
)
def test_a_module_loaded_by_a_computed_name_is_marked(line: str) -> None:
    assert read_typescript("a.ts", f"async function f() {{ return {line}; }}\n").dynamic is True


def test_a_literal_import_and_createRequire_are_not_computed() -> None:
    source = (
        "import { createRequire } from 'node:module';\n"
        "const require = createRequire(import.meta.url);\n"
        "const x = await import('./x.js');\n"
        "const y = require.resolve('./y');\n"
    )
    assert read_typescript("a.mjs", source).dynamic is False


def graph(files: list[str]) -> TsGraph:
    code = [f for f in files if not f.endswith(".d.ts")]
    return TsGraph({f: read_typescript(f, "") for f in code}, [f for f in files if f not in code])


SCANNED = [
    "src/util.ts",
    "src/lib/index.ts",
    "src/esm.mts",
    "src/plain.js",
    "src/view.tsx",
    "src/cjs.cjs",
    "src/types.d.ts",
    "top.ts",
]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("./util", "src/util.ts"),
        ("./util.js", "src/util.ts"),  # a TypeScript project writes the compiled name
        ("./lib", "src/lib/index.ts"),
        ("./esm.mjs", "src/esm.mts"),
        ("./plain", "src/plain.js"),
        ("./view", "src/view.tsx"),
        ("./cjs", "src/cjs.cjs"),
        ("./types", "src/types.d.ts"),
        ("../top", "top.ts"),
        ("./gone", None),
        ("../../outside", None),
    ],
)
def test_a_relative_import_resolves_like_node_and_typescript_do(
    spec: str, expected: str | None
) -> None:
    assert graph(SCANNED).resolve_path(posixpath.join("src", spec)) == expected


def test_a_path_that_leaves_the_scanned_directory_or_is_absolute_resolves_to_nothing() -> None:
    assert graph(SCANNED).resolve_path("/src/util.ts") is None
    assert graph(SCANNED).resolve_path("../src/util.ts") is None


def test_a_graph_follows_imports_and_says_what_it_could_not_follow() -> None:
    infos = {
        "evals/system.mjs": read_typescript(
            "evals/system.mjs", "import a from '../src/a.js'; import 'evalproof';"
        ),
        "src/a.ts": read_typescript(
            "src/a.ts",
            "import b from './b'; import './data.json'; import 'node:fs';"
            " import x from './missing'; import y from '@/lib/y'; import z from '/abs/z';",
        ),
        "src/b.ts": read_typescript("src/b.ts", "import c from './c'; export default 1;"),
        "src/c.ts": read_typescript("src/c.ts", "export const c = 1;"),
        "src/unused.ts": read_typescript("src/unused.ts", ""),
    }
    reached, problems = TsGraph(infos).reach(["evals/system.mjs"])
    assert reached == ["evals/system.mjs", "src/a.ts", "src/b.ts", "src/c.ts"]
    assert problems == [
        "src/a.ts imports './missing', which is not a scanned file",
        "src/a.ts imports '/abs/z', an absolute path",
        "src/a.ts imports '@/lib/y', a path alias or workspace package this scan does not follow",
    ]


def test_a_cycle_does_not_loop() -> None:
    infos = {
        "a.ts": read_typescript("a.ts", "import './b'"),
        "b.ts": read_typescript("b.ts", "import './a'"),
    }
    assert TsGraph(infos).reach(["a.ts"]) == (["a.ts", "b.ts"], [])


# --- the command a suite runs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "scripts"),
    [
        ("node system.mjs", ["system.mjs"]),
        ("node ./system.mjs --port 80 -e not-an-option-of-node", ["./system.mjs"]),
        ("node --no-warnings --env-file .env system.mjs", ["system.mjs"]),
        ("node --import ./register.mjs system.mjs", ["system.mjs", "./register.mjs"]),
        ("node -r ./setup.cjs --import tsx system.mjs", ["system.mjs", "./setup.cjs"]),
        ("node --import=./register.mjs system.mjs", ["system.mjs", "./register.mjs"]),
        ("/usr/local/bin/node system.mjs", ["system.mjs"]),
        ("tsx system.ts", ["system.ts"]),
        ("tsx watch system.ts", ["system.ts"]),
        ("tsx --tsconfig tsconfig.eval.json system.ts", ["system.ts"]),
        ("./node_modules/.bin/tsx system.ts", ["system.ts"]),
        ("npx tsx system.ts", ["system.ts"]),
        ("npx --yes tsx system.ts", ["system.ts"]),
        ("bun system.ts", ["system.ts"]),
        ("bun run system.ts", ["system.ts"]),
        ("deno run --allow-net --allow-env system.ts", ["system.ts"]),
        ("deno system.ts", ["system.ts"]),
        ("deno run -A --config deno.json system.ts", ["system.ts"]),
        ("node -- system.mjs", ["system.mjs"]),
    ],
)
def test_the_script_a_runner_starts_is_found(command: str, scripts: list[str]) -> None:
    parsed = node_command(shlex.split(command))
    assert parsed is not None and parsed.problem is None
    assert parsed.scripts == scripts


@pytest.mark.parametrize(
    ("command", "problem"),
    [
        ("node -e 'run()'", "inline script"),
        ("node --print 1", "inline script"),
        ("node -", "standard input"),
        ("node", "names no script"),
        ("deno task eval", "deno task"),
        ("deno test", "deno test"),
        ("bun test", "bun test"),
        ("bun x something", "bun x"),
        ("bun run", "names no script"),
    ],
)
def test_a_command_with_no_script_to_follow_says_why(command: str, problem: str) -> None:
    parsed = node_command(shlex.split(command))
    assert parsed is not None
    assert parsed.problem is not None and problem in parsed.problem


@pytest.mark.parametrize(
    "command",
    [
        "python run.py",
        "npx prettier x.ts",
        "pnpm exec tsx system.ts",
        "./run.sh",
        "java -jar a.jar",
    ],
)
def test_any_other_command_is_not_followed(command: str) -> None:
    assert node_command(shlex.split(command)) is None


# --- reachability and gaps in a scan ------------------------------------------------------------


def test_a_node_command_reaches_what_its_script_imports(tmp_path: Path) -> None:
    report = repo(
        tmp_path,
        {
            "evals/system.mjs": SYSTEM,
            "src/agent.ts": AGENT + "import { lookup } from './tools/index';\n",
            "src/tools/index.ts": (
                'import { tool } from "ai";\n'
                "export const lookup = { lookup_order: tool({ description: 'Find an order' }) };\n"
            ),
            "src/other.ts": OPENAI,
        },
    )
    suite = report.suites[0]
    assert (suite.reachability, suite.reason) == ("known", None)
    assert suite.reaches == ["evals/system.mjs", "src/agent.ts", "src/tools/index.ts"]
    assert gap_paths(report, "uncovered_code") == ["src/other.ts"]
    assert report.not_computed == {}


def test_a_gap_for_typescript_names_the_file_and_the_draft_command(tmp_path: Path) -> None:
    report = repo(tmp_path, {"evals/system.mjs": SYSTEM, "src/agent.ts": AGENT, "src/o.ts": OPENAI})
    gap = next(g for g in report.gaps if g.kind == "uncovered_code")
    assert (gap.path, gap.line) == ("src/o.ts", 2)
    assert "--gap uncovered_code --file src/o.ts" in gap.suggest


@pytest.mark.parametrize(
    ("target", "script"),
    [
        ("command:node system.mjs", "evals/system.mjs"),
        ("command:npx tsx system.ts", "evals/system.ts"),
        ("command:tsx watch system.ts", "evals/system.ts"),
        ("command:bun run system.ts", "evals/system.ts"),
        ("command:deno run --allow-net system.ts", "evals/system.ts"),
        ("command:node --import ./register.mjs ../main.js", "main.js"),
    ],
)
def test_the_runners_the_scan_follows(tmp_path: Path, target: str, script: str) -> None:
    helper = posixpath.join(posixpath.dirname(script), "helper.ts")
    files = {
        script: "import './helper';\n",
        helper: AGENT,
        "evals/register.mjs": "export {};\n",
    }
    report = repo(tmp_path, files, target=target)
    suite = report.suites[0]
    assert suite.reachability == "known", suite.reason
    assert script in suite.reaches and helper in suite.reaches


def test_an_import_the_scan_cannot_resolve_makes_the_reach_unknown(tmp_path: Path) -> None:
    report = repo(
        tmp_path,
        {
            "evals/system.mjs": "import { answer } from '../src/agent.js';\n",
            "src/agent.ts": AGENT + "import { client } from './generated/client';\n",
            "src/other.ts": OPENAI,
        },
    )
    suite = report.suites[0]
    assert suite.reachability == "unknown" and suite.reaches == []
    assert "src/agent.ts imports './generated/client', which is not a scanned file" in str(
        suite.reason
    )
    assert "uncovered_code" not in [g.kind for g in report.gaps]
    assert "not known" in report.not_computed["uncovered_code"]
    assert "not computed" in render_text(report)


@pytest.mark.parametrize(
    ("files", "unknown"),
    [
        ({"src/agent.ts": AGENT + "import { u } from '@/lib/u';\n"}, True),
        ({"src/agent.ts": AGENT + "import { u } from '~/lib/u';\n"}, True),
        ({"src/agent.ts": AGENT + "import { u } from '#internal/u';\n"}, True),
        ({"src/agent.ts": AGENT + "import { u } from '@app/u';\n"}, False),  # a package
        (
            {
                "src/agent.ts": AGENT + "import { u } from '@app/u';\n",
                "tsconfig.json": (
                    '{\n  // aliases\n  "compilerOptions": {\n'
                    '    "paths": {"@app/*": ["./src/*"]}\n  }\n}\n'
                ),
            },
            True,
        ),
        (
            {
                "src/agent.ts": AGENT + "import { u } from '@acme/shared/u';\n",
                "packages/shared/package.json": '{"name": "@acme/shared"}',
            },
            True,
        ),
        ({"src/agent.ts": AGENT + "import { u } from '@acme/shared/u';\n"}, False),
        ({"src/agent.ts": AGENT + "const m = await import(moduleName);\n"}, True),
        ({"src/agent.ts": AGENT + "const m = await import('./sibling.js');\n"}, True),  # missing
        (
            {
                "src/agent.ts": AGENT + "const m = await import('./sibling.js');\n",
                "src/sibling.ts": "export const x = 1;\n",
            },
            False,
        ),
        ({"src/agent.ts": AGENT + "import cfg from './config.json';\nimport 'node:fs';\n"}, False),
    ],
)
def test_what_the_scan_follows_and_what_leaves_the_reach_unknown(
    tmp_path: Path, files: dict[str, str], unknown: bool
) -> None:
    report = repo(tmp_path, {"evals/system.mjs": SYSTEM, **files})
    assert (report.suites[0].reachability == "unknown") is unknown, report.suites[0].reason


def test_a_script_the_scan_cannot_find_leaves_the_reach_unknown(tmp_path: Path) -> None:
    report = repo(tmp_path, {"src/agent.ts": AGENT}, target="command:node dist/system.js")
    suite = report.suites[0]
    assert suite.reachability == "unknown"
    assert "'dist/system.js'" in str(suite.reason) and "not among the scanned files" in str(
        suite.reason
    )
    assert "uncovered_code" in report.not_computed


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        ("command:node -e 'run()'", "inline script"),
        ("command:deno task eval", "deno task"),
        ("command:bun run", "names no script"),
    ],
)
def test_a_node_command_with_no_script_is_unknown_not_covered(
    tmp_path: Path, target: str, reason: str
) -> None:
    report = repo(tmp_path, {"src/agent.ts": AGENT}, target=target)
    assert report.suites[0].reachability == "unknown"
    assert reason in str(report.suites[0].reason)
    assert "uncovered_code" not in [g.kind for g in report.gaps]


@pytest.mark.parametrize("target", ["command:pnpm exec tsx system.ts", "command:python run.py"])
def test_a_command_that_is_not_a_node_runner_stays_unknown(tmp_path: Path, target: str) -> None:
    report = repo(tmp_path, {"src/agent.ts": AGENT}, target=target)
    assert report.suites[0].reachability == "unknown"
    assert "not a python target" in str(report.suites[0].reason)


def test_typescript_tests_and_import_only_files_are_not_untested_code(tmp_path: Path) -> None:
    report = repo(
        tmp_path,
        {
            "evals/system.mjs": SYSTEM,
            "src/agent.ts": AGENT,
            "src/agent.test.ts": OPENAI,
            "src/agent.spec.js": OPENAI,
            "src/__tests__/helper.ts": OPENAI,
            "src/types.ts": 'import { HumanMessage } from "@langchain/core/messages";\n',
            "src/real.ts": OPENAI,
        },
    )
    assert gap_paths(report, "uncovered_code") == ["src/real.ts"]
    # they are still found, so the report says what the code is
    paths = {f.path for f in report.findings if f.kind == "model_call"}
    assert {"src/agent.test.ts", "src/agent.spec.js", "src/__tests__/helper.ts"} <= paths


def test_a_python_suite_and_a_node_suite_each_reach_their_own_code(tmp_path: Path) -> None:
    files = {
        "app/__init__.py": "",
        "app/llm.py": (
            "import anthropic\nc = anthropic.Anthropic()\n"
            "def answer(q):\n    return c.messages.create(model='m', messages=[])\n"
        ),
        "src/agent.ts": AGENT,
        "evals/system.mjs": "import { answer } from '../src/agent.js';\n",
        "evals/py.yaml": SUITE.format(target="python:app.llm:answer"),
        "evals/py.jsonl": CASES,
    }
    both = repo(tmp_path, files)  # evals/support.yaml runs the node script, evals/py.yaml python
    assert {s.path: s.reachability for s in both.suites} == {
        "evals/support.yaml": "known",
        "evals/py.yaml": "known",
    }
    assert [g for g in both.gaps if g.kind == "uncovered_code"] == []


def test_a_python_only_suite_leaves_typescript_model_calls_untested(tmp_path: Path) -> None:
    files = {
        "app/__init__.py": "",
        "app/llm.py": (
            "import anthropic\nc = anthropic.Anthropic()\n"
            "def answer(q):\n    return c.messages.create(model='m', messages=[])\n"
        ),
        "src/agent.ts": AGENT,
    }
    report = repo(tmp_path, files, target="python:app.llm:answer")
    assert report.suites[0].reachability == "known"
    assert gap_paths(report, "uncovered_code") == ["src/agent.ts"]


def test_a_typescript_tool_no_check_names_is_a_tool_gap(tmp_path: Path) -> None:
    report = repo(
        tmp_path,
        {
            "evals/system.mjs": SYSTEM,
            "src/agent.ts": AGENT + "import './tools';\n",
            "src/tools.ts": (
                'import { tool } from "ai";\n'
                "export const tools = {\n"
                "  lookup_order: tool({ description: 'Find an order' }),\n"
                "  refund: tool({ description: 'Refund an order' }),\n"
                "};\n"
            ),
        },
        suite=SUITE_WITH_CHECKS,
    )
    tool_gaps = [g for g in report.gaps if g.kind == "tool_unchecked"]
    assert [(g.extra["tool"], g.path, g.line) for g in tool_gaps] == [("refund", "src/tools.ts", 4)]
    assert "--gap tool_unchecked --tool refund" in tool_gaps[0].suggest


def test_typescript_tools_and_agents_are_in_the_json_and_the_text(tmp_path: Path) -> None:
    report = repo(
        tmp_path,
        {
            "evals/system.mjs": SYSTEM,
            "src/agent.ts": AGENT
            + (
                "import { tool } from 'ai';\n"
                "export const t = { lookup_order: tool({ description: 'Find an order' }) };\n"
                "export const a = createAgent({ model });\n"
                "import { createAgent } from 'langchain';\n"
            ),
        },
    )
    data = json.loads(report_json(report))
    tool = next(f for f in data["findings"] if f["kind"] == "tool")
    assert tool["name"] == "lookup_order" and tool["doc"] == "Find an order"
    assert (tool["framework"], tool["language"], tool["path"]) == (
        "vercel-ai",
        "typescript",
        "src/agent.ts",
    )
    assert {f["framework"] for f in data["findings"] if f["kind"] == "agent"} == {"langchain"}
    assert "lookup_order" in render_text(report)
    assert data["suites"][0]["reachability"] == "known"


def test_draft_reads_a_typescript_scan_without_trying_to_parse_the_code(tmp_path: Path) -> None:
    root = tmp_path / "shop"
    report = repo(
        root,
        {
            "evals/system.mjs": SYSTEM,
            "src/agent.ts": AGENT,
            "src/tools.ts": (
                'import { tool } from "ai";\n'
                "export const t = { lookup_order: tool({ description: 'Find an order' }) };\n"
            ),
        },
    )
    out = tmp_path / "scan.json"
    out.write_text(report_json(report), encoding="utf-8")
    scan = load_scan(out)
    tool = scan.findings_of("tool")[0]
    request = tool_request(tool, 3, scan.findings)
    assert "`lookup_order`" in request.brief and "Find an order" in request.brief
    code = code_request(scan.root, "src/tools.ts", 2, 3, scan.findings)
    assert "`src/tools.ts`" in code.brief
    assert "It contains: tool lookup_order: Find an order" in code.brief
