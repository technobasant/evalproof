"""Read one Python file with ``ast``: its model calls, agents, tools, prompts and the evidence of
multi-turn use. The code is parsed, never imported or run.

What counts as a model call is decided from the module's own imports: a ``.messages.create(...)``
is an Anthropic call only in a file that imports ``anthropic``. A client built in another module
and imported here is not seen; that is the cost of not running anything.
"""

from __future__ import annotations

import ast
import posixpath
import re
import warnings
from dataclasses import dataclass, field

from evalproof.scan.model import Finding

PROMPT_MIN_CHARS = 40
EXCERPT_CHARS = 1500
DOC_CHARS = 600

# --- what to look for --------------------------------------------------------------------------

# (framework, the modules the file must import, the trailing names of a call on any object)
_SUFFIX_RULES: tuple[tuple[str, tuple[str, ...], tuple[tuple[str, ...], ...]], ...] = (
    (
        "anthropic",
        ("anthropic",),
        (("messages", "create"), ("messages", "stream")),
    ),
    (
        "openai",
        ("openai",),
        (
            ("chat", "completions", "create"),
            ("chat", "completions", "parse"),
            ("chat", "completions", "stream"),
            ("responses", "create"),
            ("responses", "parse"),
            ("responses", "stream"),
            ("completions", "create"),
        ),
    ),
    (
        "litellm",
        ("litellm",),
        (("completion",), ("acompletion",), ("responses",), ("aresponses",)),
    ),
    (
        "google-genai",
        ("google.genai", "google.generativeai", "vertexai"),
        (("generate_content",), ("generate_content_stream",), ("generate_content_async",)),
    ),
    (
        "mistral",
        ("mistralai",),
        (
            ("chat", "complete"),
            ("chat", "stream"),
            ("chat", "complete_async"),
            ("chat", "stream_async"),
        ),
    ),
    ("cohere", ("cohere",), (("chat",), ("chat_stream",))),
    ("ollama", ("ollama",), (("chat",), ("generate",))),
)
_BEDROCK_CALLS = (
    ("converse",),
    ("converse_stream",),
    ("invoke_model",),
    ("invoke_model_with_response_stream",),
)
_CHAT_CTOR = re.compile(r"(Azure)?Chat[A-Z][A-Za-z0-9]*")
_CHAT_DENY = frozenset(
    {
        "ChatPromptTemplate",
        "ChatMessageHistory",
        "ChatMessage",
        "ChatMessagePromptTemplate",
        "ChatPromptValue",
        "ChatGeneration",
        "ChatResult",
        "ChatSession",
        "ChatMessageChunk",
        "ChatGenerationChunk",
    }
)

_ADK_AGENTS = frozenset({"Agent", "LlmAgent", "SequentialAgent", "ParallelAgent", "LoopAgent"})
_LLAMA_AGENTS = frozenset(
    {
        "ReActAgent",
        "FunctionAgent",
        "AgentWorkflow",
        "OpenAIAgent",
        "AgentRunner",
        "FunctionCallingAgent",
    }
)
_DSPY_MODULES = frozenset({"ReAct", "Predict", "ChainOfThought", "ProgramOfThought", "CodeAct"})
_LANGGRAPH_CTORS = frozenset(
    {"create_react_agent", "StateGraph", "create_supervisor", "create_swarm"}
)

_TOOL_DECOS = frozenset({"tool", "function_tool", "tool_plain", "kernel_function"})
# The framework a tool or an agent belongs to, by the module it comes from; first match wins.
_FRAMEWORKS: tuple[tuple[str, str], ...] = (
    ("pydantic_ai", "pydantic_ai"),
    ("fastmcp", "mcp"),
    ("mcp", "mcp"),
    ("agents", "openai-agents"),
    ("google.adk", "adk"),
    ("crewai", "crewai"),
    ("semantic_kernel", "semantic_kernel"),
    ("autogen_agentchat", "autogen"),
    ("autogen_core", "autogen"),
    ("llama_index", "llama_index"),
    ("claude_agent_sdk", "claude_agent_sdk"),
    ("smolagents", "smolagents"),
    ("langgraph", "langgraph"),
    ("langchain_core", "langchain"),
    ("langchain", "langchain"),
    ("haystack", "haystack"),
    ("dspy", "dspy"),
    ("strands", "strands"),
    ("openai", "openai"),
)
_TOOL_MODULES = tuple(prefix for prefix, _ in _FRAMEWORKS)

_PROMPT_NAME = re.compile(r"prompt|instructions", re.IGNORECASE)
_PROMPT_KWARGS = frozenset(
    {"system_prompt", "instructions", "instruction", "system", "system_message"}
)

_MEMORY_NAMES = frozenset(
    {
        "ConversationBufferMemory",
        "ConversationBufferWindowMemory",
        "ConversationSummaryMemory",
        "ConversationChain",
        "ChatMessageHistory",
        "RunnableWithMessageHistory",
        "MemorySaver",
        "InMemorySaver",
        "SqliteSaver",
        "AsyncSqliteSaver",
        "PostgresSaver",
        "AsyncPostgresSaver",
        "InMemorySessionService",
        "DatabaseSessionService",
        "SQLiteSession",
        "ChatMemoryBuffer",
        "ChatSummaryMemoryBuffer",
        "ClaudeSDKClient",
        "MessagesPlaceholder",
    }
)
_HISTORY_KWARGS = frozenset(
    {
        "message_history",
        "previous_response_id",
        "chat_history",
        "checkpointer",
        "session",
        "history",
        "conversation_id",
        "thread_id",
    }
)
_ROLES = frozenset({"user", "assistant", "tool", "system"})

FuncDef = ast.FunctionDef | ast.AsyncFunctionDef


# --- results ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportRef:
    """One import statement. ``names`` pairs what the file calls a name with what it imported:
    ``from a.b import c as d`` is ``("d", "c")``; ``import a.b as q`` is ``("q", "")`` and plain
    ``import a.b`` is ``("a.b", "")``. ``level`` counts the dots of a relative import."""

    kind: str  # "import" or "from"
    module: str
    level: int
    names: tuple[tuple[str, str], ...]
    line: int


@dataclass
class PyInfo:
    path: str
    findings: list[Finding]
    imports: list[ImportRef]
    is_test: bool
    extra: dict[str, object] = field(default_factory=dict)


# --- helpers ------------------------------------------------------------------------------------


def _chain(node: ast.AST) -> tuple[list[str], bool]:
    """The names along ``a.b.c``, and whether the chain starts at a plain name. A chain that
    starts at a call (``Anthropic().messages.create``) gives only the names after it."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    rooted = isinstance(node, ast.Name)
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts[::-1], rooted


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _text(node: ast.AST | None) -> str | None:
    """The literal text of a string expression: a string, an f-string's literal parts, or one of
    those under ``textwrap.dedent(...)`` or ``.strip()``. ``None`` for anything computed."""
    if node is None:
        return None
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
    if isinstance(node, ast.Call):
        if node.args and _chain(node.func)[0][-1:] == ["dedent"]:
            return _text(node.args[0])
        if isinstance(node.func, ast.Attribute) and node.func.attr == "strip":
            return _text(node.func.value)
        return None
    return _const_str(node)


def _suffix(parts: list[str], suffixes: tuple[tuple[str, ...], ...]) -> bool:
    return any(len(parts) >= len(s) and tuple(parts[-len(s) :]) == s for s in suffixes)


def _signature(fn: FuncDef) -> str:
    returns = f" -> {ast.unparse(fn.returns)}" if fn.returns is not None else ""
    return f"{fn.name}({ast.unparse(fn.args)}){returns}"


def _doc(fn: FuncDef | None) -> str:
    return (ast.get_docstring(fn) or "")[:DOC_CHARS] if fn is not None else ""


def _short(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _agent_framework(qual: str) -> str | None:
    """The framework whose agent ``qual`` (a resolved dotted name) constructs, or ``None``."""
    head, _, last = qual.rpartition(".")
    if last.startswith("from_") and head.rsplit(".", 1)[-1][:1].isupper():
        qual, last = head, head.rsplit(".", 1)[-1]  # Agent.from_tools(...) builds an Agent
    if qual == "agents.Agent":
        return "openai-agents"
    if qual.startswith("google.adk") and last in _ADK_AGENTS:
        return "adk"
    if qual.startswith("pydantic_ai") and last == "Agent":
        return "pydantic_ai"
    if qual.startswith("crewai") and last in ("Agent", "Crew"):
        return "crewai"
    if qual.startswith("langgraph") and last in _LANGGRAPH_CTORS:
        return "langgraph"
    if qual.startswith("langchain.agents") and (
        last.startswith("create_") or last == "AgentExecutor"
    ):
        return "langchain"
    if qual.startswith("autogen_agentchat") and last.endswith("Agent"):
        return "autogen"
    if qual.startswith("llama_index") and last in _LLAMA_AGENTS:
        return "llama_index"
    if qual.startswith("dspy") and last in _DSPY_MODULES:
        return "dspy"
    if qual.startswith("haystack") and last == "Agent":
        return "haystack"
    if qual.startswith("semantic_kernel") and last.endswith("Agent"):
        return "semantic_kernel"
    if qual.startswith("claude_agent_sdk") and last in ("query", "ClaudeSDKClient"):
        return "claude_agent_sdk"
    return None


def is_test_file(path: str) -> bool:
    name = posixpath.basename(path)
    return name.startswith("test_") or name.endswith("_test.py")


# --- the analyzer -------------------------------------------------------------------------------


class _Analyzer:
    def __init__(self, path: str, tree: ast.Module) -> None:
        self.path = path
        self.tree = tree
        self.aliases: dict[str, str] = {}  # a name in this file -> the dotted name it stands for
        self.imported: set[str] = set()  # every module this file imports, in full
        self.refs: list[ImportRef] = []
        self.functions: dict[str, FuncDef] = {}
        self.bedrock = False
        self.findings: list[Finding] = []
        self._gated: list[Finding] = []  # kept only when the file calls a model or an agent
        self._index()

    # -- indexing

    def _index(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
                        binding = alias.asname
                    else:
                        top = alias.name.split(".")[0]
                        self.aliases[top] = top
                        binding = alias.name
                    self.imported.add(alias.name)
                    self.refs.append(
                        ImportRef("import", alias.name, 0, ((binding, ""),), node.lineno)
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = tuple((a.asname or a.name, a.name) for a in node.names if a.name != "*")
                if node.level == 0 and module:
                    for binding, name in names:
                        self.aliases[binding] = f"{module}.{name}"
                    self.imported.add(module)
                    self.imported.update(f"{module}.{name}" for _, name in names)
                self.refs.append(ImportRef("from", module, node.level, names, node.lineno))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self.functions.setdefault(node.name, node)
            elif isinstance(node, ast.Constant) and node.value == "bedrock-runtime":
                self.bedrock = True

    def _imports(self, prefixes: tuple[str, ...]) -> bool:
        return any(m == p or m.startswith(p + ".") for m in self.imported for p in prefixes)

    def _qual(self, parts: list[str], rooted: bool) -> str | None:
        if rooted and parts and parts[0] in self.aliases:
            return ".".join([self.aliases[parts[0]], *parts[1:]])
        return None

    def _framework(self, qual: str | None) -> str | None:
        if qual:
            for prefix, framework in _FRAMEWORKS:
                if qual == prefix or qual.startswith(prefix + "."):
                    return framework
        for prefix, framework in _FRAMEWORKS:
            if self._imports((prefix,)):
                return framework
        return None

    # -- recording

    def _add(
        self,
        kind: str,
        line: int,
        framework: str | None,
        detail: str,
        extra: dict[str, object] | None = None,
        *,
        gated: bool = False,
    ) -> None:
        finding = Finding(kind, self.path, line, framework, detail, "python", extra or {})
        (self._gated if gated else self.findings).append(finding)

    def _tool(
        self,
        name: str,
        line: int,
        framework: str | None,
        fn: FuncDef | None,
        *,
        listed_in: str | None = None,
    ) -> None:
        extra: dict[str, object] = {"name": name}
        if fn is not None:
            extra["signature"] = _signature(fn)
            if _doc(fn):
                extra["doc"] = _doc(fn)
        if listed_in is not None:
            extra["listed"] = True
            detail = f"{name} listed in tools= of {listed_in}"
        else:
            first = (_doc(fn).strip().splitlines() or ["no docstring"])[0] if fn else "no docstring"
            detail = f"{name}: {_short(first)}"
        self._add("tool", line, framework, detail, extra)

    # -- calls

    def _model_framework(self, parts: list[str], qual: str | None) -> str | None:
        if qual:
            last = qual.rsplit(".", 1)[-1]
            if qual.split(".", 1)[0].startswith("langchain") and (
                last == "init_chat_model" or (_CHAT_CTOR.fullmatch(last) and last not in _CHAT_DENY)
            ):
                return "langchain"
        for framework, prefixes, suffixes in _SUFFIX_RULES:
            if not self._imports(prefixes) or not _suffix(parts, suffixes):
                continue
            # A bare ``completion(...)`` counts only when it was imported from the library.
            if len(parts) > 1 or (qual and any(qual.startswith(p) for p in prefixes)):
                return framework
        if self.bedrock and len(parts) > 1 and _suffix(parts, _BEDROCK_CALLS):
            return "bedrock"
        return None

    def _call(self, call: ast.Call) -> None:
        parts, rooted = _chain(call.func)
        if not parts:
            return
        qual = self._qual(parts, rooted)
        framework = self._model_framework(parts, qual)
        if framework:
            model = next((_const_str(k.value) for k in call.keywords if k.arg == "model"), None)
            shown = ast.unparse(call.func)
            detail = f"{shown}(model={model!r})" if model else f"{shown}(...)"
            self._add("model_call", call.lineno, framework, _short(detail))
        agent = _agent_framework(qual) if qual else None
        if agent:
            label = parts[-1]
            name = next((_const_str(k.value) for k in call.keywords if k.arg == "name"), None)
            self._add(
                "agent", call.lineno, agent, f"{label}(name={name!r})" if name else f"{label}(...)"
            )
            self._listed_tools(call, agent, label)
        self._function_tool(call, parts, qual)
        self._multi_turn(call, parts, rooted, qual)
        for kw in call.keywords:
            if kw.arg in _PROMPT_KWARGS:
                text = _text(kw.value)
                if text is not None and len(text.strip()) >= PROMPT_MIN_CHARS:
                    self._add(
                        "prompt",
                        kw.value.lineno,
                        None,
                        f"{kw.arg}= ({len(text.strip())} characters)",
                        {"name": kw.arg, "excerpt": text.strip()[:EXCERPT_CHARS]},
                        gated=True,
                    )

    def _listed_tools(self, call: ast.Call, framework: str, label: str) -> None:
        for kw in call.keywords:
            if kw.arg != "tools" or not isinstance(kw.value, ast.List | ast.Tuple):
                continue
            for el in kw.value.elts:
                fn: FuncDef | None = None
                if isinstance(el, ast.Name):
                    name = el.id
                    fn = self.functions.get(name)
                elif isinstance(el, ast.Attribute):
                    name = el.attr
                    fn = self.functions.get(name)
                elif isinstance(el, ast.Call):
                    name = _chain(el.func)[0][-1] if _chain(el.func)[0] else ""
                else:
                    continue
                if name:
                    self._tool(name, el.lineno, framework, fn, listed_in=f"{label}(...)")

    def _function_tool(self, call: ast.Call, parts: list[str], qual: str | None) -> None:
        if not qual:
            return
        resolved = qual.split(".")
        constructor = resolved[-1] == "FunctionTool" or (
            resolved[-1] == "from_defaults"
            and len(resolved) >= 2
            and resolved[-2] == "FunctionTool"
        )
        if not constructor:
            return
        ref = next(
            (
                k.value.id
                for k in call.keywords
                if k.arg in ("func", "fn") and isinstance(k.value, ast.Name)
            ),
            None,
        )
        if ref is None and call.args and isinstance(call.args[0], ast.Name):
            ref = call.args[0].id
        name = next((_const_str(k.value) for k in call.keywords if k.arg == "name"), None) or ref
        if name:
            fn = self.functions.get(ref) if ref else None
            self._tool(name, call.lineno, self._framework(qual), fn)

    def _multi_turn(self, call: ast.Call, parts: list[str], rooted: bool, qual: str | None) -> None:
        last = parts[-1]
        if last in _MEMORY_NAMES and qual is not None:
            self._add(
                "multi_turn", call.lineno, None, f"{last} keeps the conversation across turns"
            )
        elif last == "start_chat" or (len(parts) >= 2 and parts[-2:] == ["chats", "create"]):
            self._add(
                "multi_turn", call.lineno, None, f"{ast.unparse(call.func)}() opens a chat session"
            )
        for kw in call.keywords:
            if kw.arg in _HISTORY_KWARGS:
                self._add(
                    "multi_turn",
                    call.lineno,
                    None,
                    f"{kw.arg}= passed to {_short(ast.unparse(call.func), 40)}",
                    gated=True,
                )
        if last == "append" and call.args and isinstance(call.args[0], ast.Dict):
            for key, value in zip(call.args[0].keys, call.args[0].values, strict=False):
                if _const_str(key) == "role" and _const_str(value) in _ROLES:
                    self._add(
                        "multi_turn",
                        call.lineno,
                        None,
                        "a message list grows by one role message at a time",
                        gated=True,
                    )
                    break

    # -- decorators, module-level prompts, dict keys

    def _tool_decorator(self, deco: ast.expr) -> tuple[str | None, str | None] | None:
        call = deco if isinstance(deco, ast.Call) else None
        parts, rooted = _chain(call.func if call else deco)
        if not parts or parts[-1] not in _TOOL_DECOS:
            return None
        qual = self._qual(parts, rooted)
        if len(parts) == 1:
            if qual is None:  # a decorator this file defines itself
                return None
        elif qual is None and not self._imports(_TOOL_MODULES):
            return None
        explicit: str | None = None
        if call is not None:
            explicit = next(
                (
                    _const_str(k.value)
                    for k in call.keywords
                    if k.arg in ("name", "name_override") and _const_str(k.value)
                ),
                None,
            )
            if explicit is None and call.args:
                explicit = _const_str(call.args[0])
        return self._framework(qual), explicit

    def _decorated(self) -> None:
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for deco in node.decorator_list:
                hit = self._tool_decorator(deco)
                if hit is not None:
                    self._tool(hit[1] or node.name, node.lineno, hit[0], node)
                    break

    def _module_prompts(self) -> None:
        for stmt in self.tree.body:
            if isinstance(stmt, ast.Assign):
                names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                value: ast.expr | None = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                names, value = [stmt.target.id], stmt.value
            else:
                continue
            text = _text(value)
            if text is None or len(text.strip()) < PROMPT_MIN_CHARS:
                continue
            for name in names:
                if _PROMPT_NAME.search(name):
                    self._add(
                        "prompt",
                        stmt.lineno,
                        None,
                        f"{name} ({len(text.strip())} characters)",
                        {"name": name, "excerpt": text.strip()[:EXCERPT_CHARS]},
                    )

    def _thread_keys(self) -> None:
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Dict):
                continue
            for key in node.keys:
                name = _const_str(key)
                if key is not None and name in ("thread_id", "session_id", "conversation_id"):
                    self._add("multi_turn", key.lineno, None, f'"{name}" in a config', gated=True)

    # -- run

    def run(self) -> list[Finding]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                self._call(node)
        self._decorated()
        self._module_prompts()
        self._thread_keys()
        if any(f.kind in ("model_call", "agent") for f in self.findings):
            self.findings.extend(self._gated)
        # A tool listed in an agent's ``tools=`` that this file also defines is one tool.
        defined = {
            f.extra["name"] for f in self.findings if f.kind == "tool" and "listed" not in f.extra
        }
        seen: set[tuple[str, str]] = set()
        kept: list[Finding] = []
        for f in sorted(self.findings, key=lambda f: (f.line, f.kind, f.detail)):
            if f.kind == "tool":
                name = str(f.extra["name"])
                if "listed" in f.extra and name in defined:
                    continue
                key = ("tool", name)
            elif f.kind == "multi_turn":
                key = ("multi_turn", f.detail)
            else:
                key = (f.kind, f"{f.line}:{f.detail}")
            if key in seen:
                continue
            seen.add(key)
            kept.append(f)
        return kept


def analyze_python(path: str, source: str) -> PyInfo:
    """Read ``source`` (the file at ``path``). Raises ``SyntaxError``, ``ValueError`` (a null byte)
    or ``RecursionError`` when it cannot be parsed; the caller counts it and moves on."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an invalid escape in a string is a warning, not a failure
        tree = ast.parse(source)
    analyzer = _Analyzer(path, tree)
    findings = analyzer.run()
    return PyInfo(path, findings, analyzer.refs, is_test_file(path))
