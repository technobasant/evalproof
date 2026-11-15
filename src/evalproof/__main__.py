"""``python -m evalproof``: the same entry point as the ``evalproof`` command."""

from __future__ import annotations

import sys

from evalproof.cli import main

if __name__ == "__main__":
    sys.exit(main())
