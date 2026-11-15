"""Read a TypeScript or JavaScript file with regular expressions: the model SDKs and agent
frameworks it imports, and the line of the call when one is found. Nothing is parsed or run, and
only model calls and agents are reported (no tools, prompts or reachability for these files).
"""

from __future__ import annotations

import re

from evalproof.scan.model import Finding

# import x from "m"; import "m"; export ... from "m"; require("m"); import("m")
_IMPORT = re.compile(
    r"""(?:\bimport\s+(?:[^'";]*?\s+from\s+)?|\bexport\s+[^'";]*?\s+from\s+|\brequire\(\s*|\bimport\(\s*)"""
    r"""['"]([^'"]+)['"]"""
)

# (module matcher, framework, kind of the import itself, the calls to look for)
_RULES: tuple[tuple[re.Pattern[str], str, str, tuple[re.Pattern[str], ...]], ...] = (
    (
        re.compile(r"openai(/.*)?"),
        "openai",
        "model_call",
        (
            re.compile(r"\.chat\.completions\.(create|parse|stream)\("),
            re.compile(r"\.responses\.(create|parse|stream)\("),
            re.compile(r"\.completions\.create\("),
        ),
    ),
    (
        re.compile(r"@anthropic-ai/sdk(/.*)?"),
        "anthropic",
        "model_call",
        (re.compile(r"\.messages\.(create|stream)\("),),
    ),
    (
        re.compile(r"ai(/.*)?"),
        "vercel-ai",
        "model_call",
        (re.compile(r"\b(generateText|streamText|generateObject|streamObject)\("),),
    ),
    (
        re.compile(r"@langchain/.+"),
        "langchain",
        "model_call",
        (re.compile(r"\bnew\s+(Azure)?Chat[A-Z]\w*\("), re.compile(r"\binitChatModel\(")),
    ),
    (
        re.compile(r"@google/genai(/.*)?"),
        "google-genai",
        "model_call",
        (re.compile(r"\.models\.generateContent(Stream)?\("),),
    ),
    (
        re.compile(r"@openai/agents(/.*)?"),
        "openai-agents",
        "agent",
        (re.compile(r"\bnew\s+Agent\("),),
    ),
    (
        re.compile(r"@mastra/core(/.*)?"),
        "mastra",
        "agent",
        (re.compile(r"\bnew\s+Agent\("),),
    ),
)
# LangChain's agent constructors are agents even though the module is a model module.
_LANGCHAIN_AGENT = re.compile(r"\b(createReactAgent|createAgent)\(")


def _is_comment(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith(("//", "*", "/*"))


def analyze_typescript(path: str, source: str) -> list[Finding]:
    """The model-call and agent findings of one TypeScript or JavaScript file."""
    lines = source.splitlines()
    frameworks: dict[str, tuple[str, str, tuple[re.Pattern[str], ...], int]] = {}
    for lineno, line in enumerate(lines, start=1):
        if _is_comment(line):
            continue
        for match in _IMPORT.finditer(line):
            module = match.group(1)
            for pattern, framework, kind, calls in _RULES:
                if pattern.fullmatch(module) and framework not in frameworks:
                    frameworks[framework] = (kind, module, calls, lineno)
    findings: list[Finding] = []
    for framework, (kind, module, calls, import_line) in frameworks.items():
        found = False
        for lineno, line in enumerate(lines, start=1):
            if _is_comment(line):
                continue
            for call in calls:
                hit = call.search(line)
                if hit:
                    found = True
                    text = hit.group(0).lstrip(". ").rstrip("(")
                    findings.append(
                        Finding(kind, path, lineno, framework, f"{text}(...)", "typescript")
                    )
            agent = _LANGCHAIN_AGENT.search(line) if framework == "langchain" else None
            if agent:
                found = True
                findings.append(
                    Finding(
                        "agent", path, lineno, framework, f"{agent.group(1)}(...)", "typescript"
                    )
                )
        if not found:
            findings.append(
                Finding(
                    kind,
                    path,
                    import_line,
                    framework,
                    f"imports {module}; no call found",
                    "typescript",
                )
            )
    return sorted(findings, key=lambda f: (f.line, f.kind, f.detail))
