"""Minimal runner for tests/test_domain_randomization.py (no pytest)."""

from __future__ import annotations

import sys
import traceback

import tests.test_domain_randomization as m


def _run(name: str) -> bool:
    fn = getattr(m, name)
    try:
        fn()
    except Exception:
        traceback.print_exc()
        print(f"[dr-test] FAIL {name}")
        return False
    print(f"[dr-test] ok   {name}")
    return True


def main() -> int:
    names = [n for n in dir(m) if n.startswith("test_")]
    failed = [n for n in names if not _run(n)]
    if failed:
        print(f"[dr-test] FAILED: {failed}")
        return 1
    print("[dr-test] all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
