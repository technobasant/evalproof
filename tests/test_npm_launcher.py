"""The npm launcher in ``npm/``: its version follows the Python package, its LICENSE is a copy,
and its own Node tests pass (when Node is installed).

The launcher only starts the PyPI release with the same version, so a version that drifted from
``evalproof.__version__`` would run the wrong program. The Node tests use fake ``uvx``, ``pipx``
and ``evalproof`` executables on a temporary PATH: no model, network or package index is used.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import evalproof

ROOT = Path(__file__).resolve().parents[1]
NPM = ROOT / "npm"


def test_the_npm_version_is_the_python_version() -> None:
    package = json.loads((NPM / "package.json").read_text("utf-8"))
    assert package["version"] == evalproof.__version__


def test_the_npm_license_is_a_copy_of_the_repository_license() -> None:
    copy = NPM / "LICENSE"
    assert not copy.is_symlink()  # a copy, so a Windows checkout and `npm pack` both work
    assert copy.read_bytes() == (ROOT / "LICENSE").read_bytes()


def test_the_npm_package_ships_only_the_launcher_library_readme_and_license() -> None:
    package = json.loads((NPM / "package.json").read_text("utf-8"))
    assert package["files"] == ["bin", "lib", "README.md", "LICENSE"]
    assert package["bin"] == {"evalproof": "bin/evalproof.js"}
    assert package["main"] == "lib/index.js"
    assert package["types"] == "lib/index.d.ts"
    assert "dependencies" not in package
    for name in package["files"]:
        assert (NPM / name).exists(), name


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node on PATH")
def test_the_node_tests_pass() -> None:
    node = shutil.which("node")
    assert node is not None
    done = subprocess.run(
        [node, "--test", *(str(p) for p in sorted((NPM / "test").glob("*.test.js")))],
        cwd=NPM,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
