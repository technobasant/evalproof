"""``evalproof`` writes UTF-8 to stdout and stderr on Windows, where a redirect would not."""

from __future__ import annotations

import io
import sys

import pytest

from evalproof.cli import _utf8_streams


def test_a_stream_in_the_ansi_code_page_is_switched_to_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with pytest.raises(UnicodeEncodeError):
        out.write("✅")  # the report's check mark: not in cp1252
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    _utf8_streams()
    out.write("✅ → ×")
    err.write("⚠")
    out.flush()
    err.flush()
    assert out.buffer.getvalue().decode("utf-8").endswith("✅ → ×")  # type: ignore[attr-defined]
    assert err.buffer.getvalue().decode("utf-8") == "⚠"  # type: ignore[attr-defined]


def test_a_stream_that_cannot_be_reconfigured_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Plain:
        def write(self, text: str) -> int:
            return len(text)

    monkeypatch.setattr(sys, "stdout", Plain())
    monkeypatch.setattr(sys, "stderr", Plain())
    _utf8_streams()  # no reconfigure(): nothing to do, and no error
