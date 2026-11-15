"""Which TypeScript and JavaScript files a ``command:`` target starts and what they import.

Only static imports with a relative path are followed (``./x``, ``../x``). A bare package name
(``openai``, ``node:fs``) is a dependency and is not followed. Anything else that could name code in
this repository is a gap in what the scan knows, not a file it skipped on purpose: an import that
resolves to no scanned file, a path alias, a workspace package and a module loaded by a computed
name. ``TsGraph.reach`` reports each of these, and the caller then treats the reach as unknown
instead of reporting files as untested.
"""

from __future__ import annotations

import posixpath
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from evalproof.scan.tsscan import TsInfo

# Tried in this order after the path as written; a TypeScript project imports "./x.js" for x.ts.
EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
_SWAP = {".js": (".ts", ".tsx"), ".jsx": (".tsx",), ".mjs": (".mts",), ".cjs": (".cts",)}
# Not code: an import of one of these cannot lead to a model call.
_ASSETS = frozenset(
    {
        ".json",
        ".css",
        ".scss",
        ".svg",
        ".png",
        ".jpg",
        ".txt",
        ".md",
        ".html",
        ".wasm",
        ".node",
        ".yaml",
        ".yml",
        ".csv",
        ".jsonl",
        ".sql",
        ".graphql",
        ".gql",
        ".prompt",
    }
)
_ALIAS_PREFIXES = ("@/", "~/", "#")


def _candidates(base: str) -> Iterator[str]:
    yield base
    stem, ext = posixpath.splitext(base)
    for swapped in _SWAP.get(ext, ()):
        yield stem + swapped
    for e in EXTENSIONS:
        yield base + e
    for e in EXTENSIONS:
        yield posixpath.join(base, "index" + e) if base else "index" + e
    yield base + ".d.ts"
    yield posixpath.join(base, "index.d.ts") if base else "index.d.ts"


class TsGraph:
    """The import graph of the scanned TypeScript and JavaScript files."""

    def __init__(
        self,
        infos: dict[str, TsInfo],
        declarations: Iterable[str] = (),
        workspace_names: Iterable[str] = (),
        alias_prefixes: Iterable[str] = (),
    ) -> None:
        self.infos = infos
        self.files = set(infos) | set(declarations)
        self.workspace = set(workspace_names)
        self.aliases = tuple(alias_prefixes)

    def resolve_path(self, path: str) -> str | None:
        """The scanned file ``path`` names: as written, with an extension, or as a directory with
        an ``index`` file. ``path`` is relative to the scanned directory."""
        base = posixpath.normpath(path)
        if base == ".":
            base = ""
        if base == ".." or base.startswith("../") or base.startswith("/"):
            return None
        for candidate in _candidates(base):
            if candidate in self.files:
                return candidate
        return None

    def _is_alias(self, spec: str) -> bool:
        if spec == "~" or spec.startswith(_ALIAS_PREFIXES):
            return True
        for alias in self.aliases:  # a tsconfig "paths" key: "@app/*" is a prefix, "lib" is exact
            if spec.startswith(alias[:-1]) if alias.endswith("*") else spec == alias:
                return True
        return any(spec == name or spec.startswith(name + "/") for name in self.workspace)

    def imports_of(self, info: TsInfo) -> tuple[set[str], list[str]]:
        """The scanned files ``info`` imports, and what it imports that this scan cannot follow."""
        found: set[str] = set()
        problems: list[str] = []
        here = posixpath.dirname(info.path)
        for spec in info.imports:
            if spec == "." or spec == ".." or spec.startswith(("./", "../")):
                target = self.resolve_path(posixpath.join(here, spec))
                if target is not None:
                    found.add(target)
                elif posixpath.splitext(spec)[1].lower() not in _ASSETS:
                    problems.append(f"{info.path} imports {spec!r}, which is not a scanned file")
            elif spec.startswith(("/", "file:")):
                problems.append(f"{info.path} imports {spec!r}, an absolute path")
            elif self._is_alias(spec):
                problems.append(
                    f"{info.path} imports {spec!r}, a path alias or workspace package "
                    "this scan does not follow"
                )
        if info.dynamic:
            problems.append(f"{info.path} loads a module by a name only known at run time")
        found.discard(info.path)
        return found, problems

    def reach(self, entries: Iterable[str]) -> tuple[list[str], list[str]]:
        """The entry files and every file they import, directly or through other files (sorted),
        and the imports on the way that could not be followed (sorted, once each)."""
        seen = set(entries)
        stack = sorted(seen)
        problems: set[str] = set()
        while stack:
            info = self.infos.get(stack.pop())
            if info is None:  # a declaration file holds no code
                continue
            files, issues = self.imports_of(info)
            problems.update(issues)
            for nxt in sorted(files):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return sorted(seen), sorted(problems)


# --- the command a suite runs -------------------------------------------------------------------

_VALUE_OPTIONS = frozenset(
    {
        "-r",
        "--require",
        "--import",
        "--loader",
        "--experimental-loader",
        "--preload",
        "--env-file",
        "--env-file-if-exists",
        "--tsconfig",
        "--tsconfig-override",
        "-C",
        "--conditions",
        "--input-type",
        "--title",
        "--cwd",
        "--config",
        "-c",
        "--import-map",
        "--lock",
        "--cert",
        "--port",
    }
)
_PRELOAD_OPTIONS = frozenset({"-r", "--require", "--import", "--loader", "--experimental-loader"})
_PRELOAD_OPTIONS_BUN = frozenset({"-r", "--preload", "--require", "--import"})
_SUBCOMMANDS = {"tsx": {"watch"}, "bun": {"run"}, "deno": {"run", "serve"}}
_INLINE = frozenset({"-e", "--eval", "-p", "--print"})
_BUN_SUBCOMMANDS = frozenset(
    {"x", "test", "install", "add", "remove", "create", "init", "build", "repl", "pm", "update"}
)
_DENO_SUBCOMMANDS = frozenset(
    {
        "task",
        "eval",
        "test",
        "bench",
        "repl",
        "install",
        "compile",
        "fmt",
        "lint",
        "check",
        "bundle",
        "info",
        "doc",
        "add",
        "remove",
        "publish",
        "init",
        "upgrade",
    }
)


@dataclass
class NodeCommand:
    """A command line that starts a script under node, tsx, bun or deno: the script, and any module
    it is told to load first (``--import ./register.mjs``). ``problem`` says why the script cannot
    be named (an inline script, a package script, a subcommand that is not a run)."""

    runner: str
    scripts: list[str] = field(default_factory=list)
    problem: str | None = None


def _program(word: str) -> str:
    name = posixpath.basename(word.replace("\\", "/")).lower()
    for suffix in (".exe", ".cmd"):
        name = name.removesuffix(suffix)
    return name


def node_command(argv: list[str]) -> NodeCommand | None:
    """The script a ``node``, ``tsx`` (or ``npx tsx``), ``bun`` or ``deno`` command line starts, or
    ``None`` when the command is none of these."""
    runner = _program(argv[0])
    rest = argv[1:]
    if runner == "npx":
        rest = list(rest)
        while rest and rest[0].startswith("-"):
            rest.pop(0)
        if not rest or _program(rest[0]) != "tsx":
            return None
        runner, rest = "tsx", rest[1:]
    if runner not in ("node", "tsx", "bun", "deno"):
        return None
    command = NodeCommand(runner)
    preload = _PRELOAD_OPTIONS_BUN if runner == "bun" else _PRELOAD_OPTIONS
    positional: list[str] = []
    i = 0
    while i < len(rest):
        word = rest[i]
        i += 1
        if word == "--":
            positional.extend(rest[i:])
            break
        if word == "-":
            command.problem = "reads its script from standard input"
            return command
        if not word.startswith("-"):
            positional.append(word)
            if len(positional) >= (2 if positional[0] in _SUBCOMMANDS.get(runner, ()) else 1):
                break  # what follows the script belongs to the script
            continue
        option, equals, value = word.partition("=")
        if option in _INLINE:
            command.problem = f"runs an inline script ({option})"
            return command
        if option in _VALUE_OPTIONS and not equals:
            value = rest[i] if i < len(rest) else ""
            i += 1
        if option in preload and value.startswith("."):
            command.scripts.append(value)
    script = _script(command, positional)
    if script is not None:
        command.scripts.insert(0, script)
    return command


def _script(command: NodeCommand, positional: list[str]) -> str | None:
    """The script among the positional words, or ``None`` with ``command.problem`` set."""
    if command.problem:
        return None
    words = list(positional)
    first = words[0] if words else ""
    if command.runner == "deno":
        if first in _DENO_SUBCOMMANDS:
            command.problem = f"runs `deno {first}`, which is not a script this scan can follow"
            return None
        if first in ("run", "serve"):
            words = words[1:]
    elif command.runner == "bun":
        if first in _BUN_SUBCOMMANDS:
            command.problem = f"runs `bun {first}`, which is not a script this scan can follow"
            return None
        if first == "run":
            words = words[1:]
    elif command.runner == "tsx" and first == "watch":
        words = words[1:]
    if not words:
        command.problem = "names no script"
        return None
    return words[0]
