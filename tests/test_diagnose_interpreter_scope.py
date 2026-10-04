"""Regression: dependency checks must state which interpreter they describe.

`mnemosyne diagnose` imports fastembed/sqlite_vec inside the CLI process. When
the CLI is a pipx install and the Hermes provider runs from its own venv, the
checks report the *CLI* environment as healthy while the runtime that actually
serves recall lacks the vector stack (#813). The false green is silent: recall
degrades to FTS-only and nothing says so.

These tests pin the contract the maintainer specified for #813:

- every dependency check carries the interpreter it describes, so a reader can
  never mistake a CLI-local result for the provider runtime;
- the scope is explicit in the machine-readable payload, not only in prose;
- no `HERMES_HOME`-style guessing is introduced to "fix" the mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mnemosyne.runtime_diagnostics import collect_runtime_diagnostics


def test_every_dep_check_declares_the_interpreter_it_ran_in():
    checks = collect_runtime_diagnostics()["checks"]
    dep_checks = [c for c in checks if c["category"] == "deps"]
    assert dep_checks, "expected dependency checks in the payload"

    for c in dep_checks:
        scope = c.get("scope")
        assert scope == "cli_interpreter", (
            f"dep check {c['check']!r} has scope={scope!r}; without it a green "
            "result reads as a statement about the runtime serving recall"
        )


def test_scope_is_machine_readable_not_only_prose():
    """The payload, not just the human print, must carry the scope."""
    payload = collect_runtime_diagnostics()
    assert payload["scope"] == "cli_interpreter"
    assert "executable" in payload


def test_executable_stays_redacted():
    """The repo already redacts sys.executable to a basename; keep it that way."""
    payload = collect_runtime_diagnostics()
    assert "/" not in payload["executable"], payload["executable"]


def test_no_guessed_runtime_interpreter(monkeypatch):
    """A heuristic that scans HERMES_HOME would only move the misleading report.

    Asserted behaviourally: with a stale HERMES_HOME pointing somewhere else, the
    scope still describes this interpreter and is not resolved from the env var.
    """
    monkeypatch.setenv("HERMES_HOME", "/nonexistent/hermes-home-that-is-not-this-env")
    payload = collect_runtime_diagnostics()
    assert payload["scope"] == "cli_interpreter"
    assert payload["executable"] == Path(sys.executable).name
    assert "nonexistent" not in json.dumps(payload)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))