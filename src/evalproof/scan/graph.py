"""Which of the repository's Python files import which, and what a suite's target reaches.

A file's imports are resolved against source roots: the repository root, ``src/``, the directory
above each top-level package, and each suite's own directory. An import that names something
outside the repository (the standard library, a dependency) resolves to nothing.
"""

from __future__ import annotations

import posixpath
from collections.abc import Iterable

from evalproof.scan.pyscan import ImportRef, PyInfo


def _ancestors(path: str) -> list[str]:
    """The directories above ``path``, nearest first, ending with "" (the repository root)."""
    out: list[str] = []
    here = posixpath.dirname(path)
    while here:
        out.append(here)
        here = posixpath.dirname(here)
    out.append("")
    return out


def _package_root(path: str, files: set[str]) -> str:
    """The directory that holds ``path``'s top-level package: the first ancestor with no
    ``__init__.py``. A script next to no package is its own root."""
    here = posixpath.dirname(path)
    while here and f"{here}/__init__.py" in files:
        here = posixpath.dirname(here)
    return here


def _module_name(path: str, root: str) -> str:
    rel = path[len(root) + 1 :] if root else path
    name = rel[: -len(".py")].replace("/", ".")
    return name.removesuffix(".__init__")


class ModuleIndex:
    """Every Python file by the dotted module names it can be imported as, per source root."""

    def __init__(self, files: Iterable[str], extra_roots: Iterable[str] = ()) -> None:
        self.files = set(files)
        self.own_root = {f: _package_root(f, self.files) for f in self.files}
        self.roots: list[str] = list(
            dict.fromkeys(["", "src", *extra_roots, *sorted(set(self.own_root.values()))])
        )
        known = set(self.roots)
        self.by_root: dict[str, dict[str, str]] = {r: {} for r in self.roots}
        for path in sorted(self.files):
            for root in _ancestors(path):
                if root not in known:
                    continue
                name = _module_name(path, root)
                if name and name != "__init__":
                    self.by_root[root].setdefault(name, path)

    def _order(self, from_file: str | None, first: Iterable[str] = ()) -> list[str]:
        own = [self.own_root[from_file]] if from_file in self.own_root else []
        return list(dict.fromkeys([*first, *own, *self.roots]))

    def resolve(
        self, module: str, from_file: str | None = None, first: Iterable[str] = ()
    ) -> str | None:
        """The file ``import module`` loads, or ``None`` when it is not in the repository."""
        for root in self._order(from_file, first):
            found = self.by_root.get(root, {}).get(module)
            if found:
                return found
        return None

    def _prefixes(self, module: str, from_file: str | None) -> list[str]:
        """The files an ``import a.b.c`` runs: ``a``, ``a.b`` and ``a.b.c``, where they exist."""
        parts = module.split(".")
        out = []
        for i in range(1, len(parts) + 1):
            found = self.resolve(".".join(parts[:i]), from_file)
            if found:
                out.append(found)
        return out

    def _relative_base(self, ref: ImportRef, path: str) -> str | None:
        root = self.own_root[path]
        name = _module_name(path, root)
        package = (
            name.split(".")
            if path.endswith("/__init__.py") or path == "__init__.py"
            else (name.split(".")[:-1])
        )
        keep = len(package) - (ref.level - 1)
        if keep < 0 or (keep == 0 and package):  # above the top-level package
            return None
        return ".".join(package[:keep])

    def targets(self, ref: ImportRef, path: str) -> list[tuple[str, str, str | None]]:
        """For each name an import binds: ``(binding, imported name, file it comes from)``. The
        file is the module itself, or the submodule when the imported name is one (the imported
        name is then ``""``)."""
        if ref.kind == "import":
            file = self.resolve(ref.module, path)
            return [(binding, "", file) for binding, _ in ref.names]
        out: list[tuple[str, str, str | None]] = []
        if ref.level:
            base = self._relative_base(ref, path)
            if base is None:
                return [(binding, name, None) for binding, name in ref.names]
            module = ".".join(p for p in (base, ref.module) if p)
            lookup = self.by_root[self.own_root[path]]
            for binding, name in ref.names:
                sub = lookup.get(f"{module}.{name}" if module else name)
                out.append((binding, "" if sub else name, sub or lookup.get(module)))
            return out
        for binding, name in ref.names:
            sub = self.resolve(f"{ref.module}.{name}", path)
            out.append((binding, "" if sub else name, sub or self.resolve(ref.module, path)))
        return out

    def _parents(self, path: str) -> list[str]:
        """The ``__init__.py`` files of the packages ``path`` sits in: importing a module runs
        them first."""
        root = self.own_root.get(path, "")
        out: list[str] = []
        here = posixpath.dirname(path)
        while here and here != root and (here + "/__init__.py") in self.files:
            out.append(here + "/__init__.py")
            here = posixpath.dirname(here)
        return [p for p in out if p != path]

    def edges(self, info: PyInfo) -> set[str]:
        """The repository files ``info`` imports (itself excluded), with the ``__init__.py`` of the
        packages those files and this one sit in."""
        found: set[str] = set()
        for ref in info.imports:
            if ref.kind == "import" or not ref.level:
                found.update(self._prefixes(ref.module, info.path))
            for _, _, file in self.targets(ref, info.path):
                if file:
                    found.add(file)
        for file in [*found, info.path]:
            found.update(self._parents(file))
        found.discard(info.path)
        return found


def reachable(start: str, edges: dict[str, set[str]]) -> list[str]:
    """``start`` and every file it imports, directly or through other files, sorted."""
    seen = {start}
    stack = [start]
    while stack:
        for nxt in edges.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return sorted(seen)
