"""Read a TypeScript or JavaScript file with regular expressions: the model SDKs and agent
frameworks it imports and the line of the call when one is found, the tools it defines (the Vercel
AI SDK's ``tool()``, LangChain.js's ``tool()`` and ``DynamicStructuredTool``), and the modules it
imports. Nothing is parsed or run, so a name or a description that is not a string literal in the
file is not reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from evalproof.scan.model import Finding

# Comments and strings, in the order a reader meets them: a ``//`` inside a string is not a comment.
_TOKEN = re.compile(
    r"//[^\n]*|/\*.*?\*/|'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"|`(?:\\.|[^`\\])*`", re.DOTALL
)
_STRING = re.compile(r"'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"|`(?:\\.|[^`\\])*`", re.DOTALL)

_CLAUSE = r"(?:[\w$]+(?:\s*,\s*(?:\*\s*as\s+[\w$]+|\{[^}]*\}))?|\*\s*as\s+[\w$]+|\{[^}]*\})"
# import x from "m"; import "m"; export {x} from "m"; export * from "m"; require("m"); import("m")
_IMPORT = re.compile(
    rf"""(?:\bimport\s+(?:type\s+)?{_CLAUSE}\s*from\s*|\bimport\s*"""
    r"""|\bexport\s+(?:type\s+)?(?:\*(?:\s*as\s+[\w$]+)?|\{[^}]*\})\s*from\s*"""
    r"""|\brequire\(\s*|\bimport\(\s*)"""
    r"""['"]([^'"]+)['"]"""
)
# ``import type {...}`` and ``export type {...} from`` load nothing at run time.
_TYPE_ONLY = re.compile(r"(?:import|export)\s+type\s+(?!from\b)")
# require(x) or import(x) with an argument that is not a string literal: a module named at run time
_DYNAMIC = re.compile(r"""(?<![\w$.])(?:require|import)\s*\(\s*(?!['"])(?=[^\s)])""")

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
        re.compile(r"(@langchain/.+|langchain(/.*)?)"),
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

# What a file imports to define tools and agents, and from which modules.
_VERCEL_MODULE = re.compile(r"ai(/.*)?")
_LANGCHAIN_MODULE = re.compile(r"(@langchain/.+|langchain(/.*)?)")
_VERCEL_TOOLS = ("tool", "dynamicTool")
_VERCEL_AGENTS = ("Experimental_Agent", "Agent", "ToolLoopAgent")
_LANGCHAIN_TOOL_CLASSES = ("DynamicStructuredTool", "DynamicTool")
_LOOP_OPTION = re.compile(r"\b(stopWhen|maxSteps)\s*:")

_NAMED = re.compile(
    r"""\bimport\s+(?!type\s)(?:[\w$]+\s*,\s*)?\{([^}]*)\}\s*from\s*['"]([^'"]+)['"]"""
    r"""|\b(?:const|let|var)\s*\{([^}]*)\}\s*=\s*(?:await\s+)?"""
    r"""(?:require|import)\(\s*['"]([^'"]+)['"]\s*\)"""
)
_NAMESPACE = re.compile(
    r"""\bimport\s+\*\s*as\s+([\w$]+)\s+from\s*['"]([^'"]+)['"]"""
    r"""|\b(?:const|let|var)\s+([\w$]+)\s*=\s*(?:await\s+)?"""
    r"""(?:require|import)\(\s*['"]([^'"]+)['"]\s*\)"""
)
_ITEM = re.compile(r"([\w$]+)(?:\s+as\s+([\w$]+)|\s*:\s*([\w$]+))?")
_NAME_KEY = r"""[A-Za-z_$][\w$]*|'[^'\n]+'|"[^"\n]+\""""
_KEY_BEFORE = re.compile(rf"""(?:^|[\s,{{(])({_NAME_KEY})\s*:\s*$""")
_VAR_BEFORE = re.compile(
    r"""\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=;]+?)?=\s*(?:await\s+)?$"""
)
_PROP = re.compile(rf"""\s*({_NAME_KEY})\s*:\s*""")
_ALIAS_PIECE = re.compile(rf"""\s*({_NAME_KEY})\s*:\s*([A-Za-z_$][\w$]*)\s*""")
_TEST_FILE = re.compile(r"\.(?:test|spec)\.[cm]?[jt]sx?$")

IMPORT_ONLY = "; no call found"


@dataclass
class TsInfo:
    """What one TypeScript or JavaScript file holds: its findings and the modules it loads."""

    path: str
    findings: list[Finding]
    imports: list[str] = field(default_factory=list)  # module specifiers, in source order
    dynamic: bool = False  # it loads a module by a name only known at run time
    is_test: bool = False


def is_ts_test_file(path: str) -> bool:
    return bool(_TEST_FILE.search(path)) or "__tests__" in path.split("/")


def is_import_only(finding: Finding) -> bool:
    """A finding that only says an SDK is imported and no call was found."""
    return finding.language == "typescript" and finding.detail.endswith(IMPORT_ONLY)


def _strip_comments(source: str) -> str:
    """``source`` with every comment blanked and every line break kept, so offsets and line
    numbers stay the same."""

    def blank(match: re.Match[str]) -> str:
        text = match.group(0)
        if text.startswith(("//", "/*")):
            return re.sub(r"[^\n]", " ", text)
        return text

    return _TOKEN.sub(blank, source)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


# --- reading a call -----------------------------------------------------------------------------


def _region(text: str, start: int, limit: int = 20_000) -> tuple[list[str], int] | None:
    """The comma-separated top-level pieces of the bracketed text that opens at ``text[start]``,
    and the offset after its closing bracket. ``None`` when it does not close within ``limit``
    characters. A trailing comma adds no piece."""
    depth = 0
    pieces: list[str] = []
    piece_start = start + 1
    i = start
    end = min(len(text), start + limit)
    while i < end:
        c = text[i]
        if c in "'\"`":
            match = _STRING.match(text, i)
            i = match.end() if match else i + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                last = text[piece_start:i]
                if last.strip() or not pieces:  # a trailing comma leaves nothing after it
                    pieces.append(last)
                return pieces, i + 1
        elif c == "," and depth == 1:
            pieces.append(text[piece_start:i])
            piece_start = i + 1
        i += 1
    return None


_UNESCAPE = re.compile(r"\\(.)", re.DOTALL)


def _literal(text: str) -> str | None:
    """The value of ``"a" + 'b'`` (string literals joined with ``+``), or ``None`` when the text is
    anything else, such as a variable or a template with ``${...}``."""
    text = text.strip()
    parts: list[str] = []
    i = 0
    while True:
        match = _STRING.match(text, i)
        if match is None:
            return None
        body = match.group(0)[1:-1]
        if match.group(0).startswith("`") and "${" in body:
            return None
        parts.append(_UNESCAPE.sub(lambda m: " " if m.group(1) in "nt" else m.group(1), body))
        i = match.end()
        rest = text[i:].lstrip()
        if not rest:
            return " ".join("".join(parts).split())
        if not rest.startswith("+"):
            return None
        i = len(text) - len(rest) + 1
        while i < len(text) and text[i].isspace():
            i += 1


def _props(obj: str) -> dict[str, str | None]:
    """The properties of the object literal ``obj``, with the string value of each one that is a
    literal and ``None`` for any other value."""
    obj = obj.strip()
    if not obj.startswith("{"):
        return {}
    got = _region(obj, 0)
    if got is None:
        return {}
    out: dict[str, str | None] = {}
    for piece in got[0]:
        match = _PROP.match(piece)
        if match:
            out.setdefault(match.group(1).strip("'\""), _literal(piece[match.end() :]))
    return out


def _bound_names(clean: str, module: re.Pattern[str]) -> tuple[dict[str, set[str]], set[str]]:
    """What ``clean`` binds from the modules ``module`` accepts: ``{exported name: local names}``
    from ``import {a as b}`` and ``const {a: b} = require()``, and the names bound to the whole
    module (``import * as ns``)."""
    names: dict[str, set[str]] = {}
    spaces: set[str] = set()
    for match in _NAMED.finditer(clean):
        group, mod = (match.group(1), match.group(2)) if match.group(2) else match.group(3, 4)
        if group is None or mod is None or not module.fullmatch(mod):
            continue
        for item in group.split(","):
            item = item.strip()
            parsed = None if item.startswith("type ") else _ITEM.fullmatch(item)
            if parsed:
                local = parsed.group(2) or parsed.group(3) or parsed.group(1)
                names.setdefault(parsed.group(1), set()).add(local)
    for match in _NAMESPACE.finditer(clean):
        name, mod = (match.group(1), match.group(2)) if match.group(2) else match.group(3, 4)
        if name and mod and module.fullmatch(mod):
            spaces.add(name)
    return names, spaces


def _calls(
    clean: str, names: dict[str, set[str]], spaces: set[str], exported: str, *, new: bool = False
) -> list[tuple[int, int]]:
    """Where ``exported`` is called (or constructed, with ``new``) under any name it is bound to:
    ``(offset of the call, offset of its opening bracket)``."""
    options = [re.escape(local) for local in sorted(names.get(exported, ()))]
    options += [re.escape(space) + r"\." + re.escape(exported) for space in sorted(spaces)]
    if not options:
        return []
    lead = r"\bnew\s+" if new else r"(?<![\w$.])"
    pattern = re.compile(lead + "(?:" + "|".join(options) + r")\s*(?:<[^()]*?>)?\s*\(")
    return [(m.start(), m.end() - 1) for m in pattern.finditer(clean)]


def _first_line(text: str | None) -> str:
    stripped = (text or "").strip()
    line = stripped.splitlines()[0].strip() if stripped else ""
    return line if len(line) <= 80 else line[:77] + "..."


def _tool_finding(
    path: str, line: int, framework: str, name: str, description: str | None
) -> Finding:
    extra: dict[str, object] = {"name": name}
    if description:
        extra["doc"] = description
    detail = f"{name}: {_first_line(description) or 'no description'}"
    return Finding("tool", path, line, framework, detail, "typescript", extra)


def _tool_keys(clean: str) -> dict[str, str]:
    """``{variable: key}`` for each ``tools: { key: variable }`` entry: the Vercel AI SDK names a
    tool by its key, so a tool kept in a variable first is known by the key it is given there."""
    keys: dict[str, str] = {}
    for match in re.finditer(r"\btools\s*:\s*\{", clean):
        got = _region(clean, match.end() - 1)
        if got is None:
            continue
        for piece in got[0]:
            entry = _ALIAS_PIECE.fullmatch(piece)
            if entry:
                keys[entry.group(2)] = entry.group(1).strip("'\"")
    return keys


def _vercel(path: str, clean: str) -> list[Finding]:
    names, spaces = _bound_names(clean, _VERCEL_MODULE)
    if not names and not spaces:
        return []
    out: list[Finding] = []
    keys = _tool_keys(clean)
    for exported in _VERCEL_TOOLS:
        for start, bracket in _calls(clean, names, spaces, exported):
            got = _region(clean, bracket)
            if got is None or not got[0]:
                continue
            before = clean[max(0, start - 300) : start]
            key = _KEY_BEFORE.search(before)
            var = _VAR_BEFORE.search(before)
            name = key.group(1).strip("'\"") if key else keys.get(var.group(1)) if var else None
            if name is None and var:
                name = var.group(1)
            if name:
                description = _props(got[0][0]).get("description")
                out.append(
                    _tool_finding(path, _line_of(clean, start), "vercel-ai", name, description)
                )
    for exported in _VERCEL_AGENTS:
        for start, _ in _calls(clean, names, spaces, exported, new=True):
            out.append(
                Finding(
                    "agent",
                    path,
                    _line_of(clean, start),
                    "vercel-ai",
                    f"new {exported}(...)",
                    "typescript",
                )
            )
    for match in _LOOP_OPTION.finditer(clean):
        out.append(
            Finding(
                "agent",
                path,
                _line_of(clean, match.start()),
                "vercel-ai",
                f"{match.group(1)}: a multi-step tool loop",
                "typescript",
            )
        )
    return out


def _langchain(path: str, clean: str) -> list[Finding]:
    names, spaces = _bound_names(clean, _LANGCHAIN_MODULE)
    if not names and not spaces:
        return []
    out: list[Finding] = []
    for start, bracket in _calls(clean, names, spaces, "tool"):
        got = _region(clean, bracket)
        if got is None or len(got[0]) < 2:  # tool(fn, {name, description, schema})
            continue
        props = _props(got[0][-1])
        if props.get("name"):
            out.append(
                _tool_finding(
                    path,
                    _line_of(clean, start),
                    "langchain",
                    str(props["name"]),
                    props.get("description"),
                )
            )
    for exported in _LANGCHAIN_TOOL_CLASSES:
        for start, bracket in _calls(clean, names, spaces, exported, new=True):
            got = _region(clean, bracket)
            props = _props(got[0][0]) if got and got[0] else {}
            if props.get("name"):
                out.append(
                    _tool_finding(
                        path,
                        _line_of(clean, start),
                        "langchain",
                        str(props["name"]),
                        props.get("description"),
                    )
                )
    return out


# --- the file -----------------------------------------------------------------------------------


def read_typescript(path: str, source: str) -> TsInfo:
    """One file: its model-call, agent and tool findings, and the modules it imports."""
    clean = _strip_comments(source)
    frameworks: dict[str, tuple[str, str, tuple[re.Pattern[str], ...], int]] = {}
    imports: list[str] = []
    for match in _IMPORT.finditer(clean):
        module = match.group(1)
        if not _TYPE_ONLY.match(clean, match.start()):
            imports.append(module)
        for pattern, framework, kind, calls in _RULES:
            if pattern.fullmatch(module) and framework not in frameworks:
                frameworks[framework] = (kind, module, calls, _line_of(clean, match.start()))
    defined = [*_vercel(path, clean), *_langchain(path, clean)]
    has_defined = {f.framework for f in defined}
    lines = clean.split("\n")
    findings: list[Finding] = list(defined)
    for framework, (kind, module, calls, import_line) in frameworks.items():
        found = framework in has_defined
        for lineno, line in enumerate(lines, start=1):
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
                    f"imports {module}{IMPORT_ONLY}",
                    "typescript",
                )
            )
    findings.sort(key=lambda f: (f.line, f.kind, f.detail))
    return TsInfo(
        path,
        findings,
        imports,
        dynamic=bool(_DYNAMIC.search(clean)),
        is_test=is_ts_test_file(path),
    )


def analyze_typescript(path: str, source: str) -> list[Finding]:
    """The model-call, agent and tool findings of one TypeScript or JavaScript file."""
    return read_typescript(path, source).findings
