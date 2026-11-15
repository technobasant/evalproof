"""Test cases hidden in pytest files: a call with a literal input, and an assertion on its result.

``result = ask("What is the refund window?")`` followed by ``assert "30 days" in result`` is a
case: the input, and an expected answer a person wrote in a test. Only those shapes are read, and
only calls of functions the scan says reach a model call. No model is called and nothing is run:
the test file is parsed, never imported.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Methods that only reshape the text the assertion then compares.
_RESHAPE = frozenset({"lower", "upper", "strip", "casefold", "lstrip", "rstrip"})
_INPUT_KWARGS = frozenset(
    {"question", "query", "prompt", "input", "text", "message", "user_input", "q", "content"}
)
_FLAGS = {"I": "i", "IGNORECASE": "i", "M": "m", "MULTILINE": "m", "S": "s", "DOTALL": "s"}


@dataclass
class Harvested:
    """What the tests say about one input."""

    input: str
    source: str  # path::test_name:line of the first call
    reference: str | None = None
    accept: list[str] = field(default_factory=list)
    pattern: str | None = None

    def fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.reference is not None:
            out["reference"] = self.reference
        if self.accept:
            out["accept"] = list(self.accept)
        if self.pattern is not None:
            out["pattern"] = self.pattern
        return out


@dataclass
class HarvestResult:
    cases: list[Harvested] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tests: int = 0


def _str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _chain(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


def _literal_input(call: ast.Call) -> str | None:
    """The one text the call passes as the input: its first argument, or the keyword that names
    it. Anything computed is not a literal and gives ``None``."""
    if call.args:
        return _str(call.args[0])
    literal = [(k.arg, _str(k.value)) for k in call.keywords if k.arg and _str(k.value) is not None]
    named = [text for name, text in literal if name in _INPUT_KWARGS]
    if named:
        return named[0]
    return literal[0][1] if len(literal) == 1 and len(call.keywords) == 1 else None


def _flags(node: ast.AST | None) -> str | None:
    """The inline flags (``"i"``, ``"im"``, ...) for a ``re`` flags argument, ``""`` for none,
    ``None`` when the argument is anything this does not understand."""
    if node is None:
        return ""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = _flags(node.left), _flags(node.right)
        return None if left is None or right is None else left + right
    chain = _chain(node)
    if chain and len(chain) == 2 and chain[0] == "re" and chain[1] in _FLAGS:
        return _FLAGS[chain[1]]
    return None


class _Test:
    """One test function: the calls it makes with literal inputs, and what it asserts."""

    def __init__(
        self,
        path: str,
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        imports: dict[str, str],
    ) -> None:
        self.path = path
        self.fn = fn
        self.modules = {b for b, target in imports.items() if "::" not in target}
        self.functions = {b for b, target in imports.items() if "::" in target}
        self.assigned: dict[
            str, list[tuple[int, str, int]]
        ] = {}  # name -> (line, input, call line)

    def _is_target(self, call: ast.Call) -> bool:
        chain = _chain(call.func)
        if not chain:
            return False
        if len(chain) == 1:
            return chain[0] in self.functions
        return any(".".join(chain[:k]) in self.modules for k in range(1, len(chain)))

    def _call_of(self, node: ast.AST) -> ast.Call | None:
        while isinstance(node, ast.Await):
            node = node.value
        return node if isinstance(node, ast.Call) and self._is_target(node) else None

    def _unwrap(self, node: ast.AST) -> ast.AST:
        """The expression under ``.lower()``, ``.strip()`` and the like."""
        while True:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RESHAPE
                and not node.args
            ):
                node = node.func.value
            elif isinstance(node, ast.Await):
                node = node.value
            else:
                return node

    def _input_of(self, node: ast.AST, line: int) -> tuple[str, int] | None:
        """The literal input of the call whose result ``node`` is (a call, or a name that was
        assigned one before ``line``), and the line of that call."""
        node = self._unwrap(node)
        call = self._call_of(node)
        if call is not None:
            text = _literal_input(call)
            return (text, call.lineno) if text is not None else None
        if isinstance(node, ast.Name):
            earlier = [a for a in self.assigned.get(node.id, []) if a[0] <= line]
            if earlier:
                _, text, call_line = max(earlier, key=lambda a: a[0])
                return text, call_line
        return None

    def collect(self) -> list[tuple[str, int, dict[str, Any]]]:
        """``(input, call line, {field: value})`` for each recognised assertion."""
        for node in ast.walk(self.fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                call = self._call_of(node.value)
                if isinstance(target, ast.Name) and call is not None:
                    text = _literal_input(call)
                    if text is not None:
                        self.assigned.setdefault(target.id, []).append(
                            (node.lineno, text, call.lineno)
                        )
        found: list[tuple[str, int, dict[str, Any]]] = []
        for node in ast.walk(self.fn):
            if isinstance(node, ast.Assert):
                found.extend(self._assertion(node.test, node.lineno))
        return found

    def _assertion(self, test: ast.expr, line: int) -> list[tuple[str, int, dict[str, Any]]]:
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            return [hit for value in test.values for hit in self._assertion(value, line)]
        # assert re.search(pattern, result) [is not None]
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            test = test.left
        if isinstance(test, ast.Call):
            return self._pattern(test, line)
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return []
        op, left, right = test.ops[0], test.left, test.comparators[0]
        if isinstance(op, ast.Eq):
            for result, literal in ((left, right), (right, left)):
                want = _str(literal)
                got = self._input_of(result, line) if want is not None else None
                if want is not None and got is not None:
                    return [(got[0], got[1], {"reference": want})]
        elif isinstance(op, ast.In):
            want = _str(left)
            got = self._input_of(right, line) if want is not None else None
            if want is not None and got is not None:
                return [(got[0], got[1], {"accept": want})]
        return []

    def _pattern(self, call: ast.Call, line: int) -> list[tuple[str, int, dict[str, Any]]]:
        if _chain(call.func) != ["re", "search"] or len(call.args) < 2:
            return []
        pattern = _str(call.args[0])
        flags = _flags(call.args[2] if len(call.args) > 2 else None)
        got = self._input_of(call.args[1], line)
        if pattern is None or flags is None or got is None:
            return []
        return [(got[0], got[1], {"pattern": f"(?{flags}){pattern}" if flags else pattern})]


def _tests(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test")
    ]


def harvest(root: Path, test_file: str, imports: dict[str, str]) -> HarvestResult:
    """The cases ``test_file`` (relative to ``root``) holds. ``imports`` maps each name the file
    imports from a module that reaches a model call to that file (``file::name`` for a name inside
    it), as the scan recorded it."""
    result = HarvestResult()
    try:
        tree = ast.parse((root / test_file).read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError) as exc:
        result.warnings.append(f"{test_file}: could not be read as Python ({exc}); skipped")
        return result
    by_input: dict[str, Harvested] = {}
    for fn in sorted(_tests(tree), key=lambda n: n.lineno):
        result.tests += 1
        for text, line, got in _Test(test_file, fn, imports).collect():
            case = by_input.setdefault(text, Harvested(text, f"{test_file}::{fn.name}:{line}"))
            if "reference" in got:
                if case.reference is None:
                    case.reference = got["reference"]
                elif case.reference != got["reference"]:
                    result.warnings.append(
                        f"{case.source}: input {text!r} has two different exact answers in "
                        "the tests; kept the first"
                    )
            if "accept" in got and got["accept"] not in case.accept:
                case.accept.append(got["accept"])
            if "pattern" in got and case.pattern is None:
                case.pattern = got["pattern"]
    result.cases = list(by_input.values())
    return result
