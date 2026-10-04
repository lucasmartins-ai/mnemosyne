"""Pure runtime diagnostics shared by doctor and legacy diagnose.

This module intentionally has no logging, database, provider, repair, or CLI
dependencies.  Keeping it neutral lets ``mnemosyne.doctor`` report runtime
capabilities without importing the mutable ``mnemosyne.diagnose`` command.
"""

import importlib.metadata
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Dependency and capability checks run in whichever interpreter called us. Naming
# that scope is what keeps a green CLI result from being read as a statement
# about the runtime serving recall (#813).
DEP_SCOPE = "cli_interpreter"


def collect_runtime_diagnostics() -> dict[str, Any]:
    """Run pure runtime, dependency, and capability checks without a provider.

    Every dependency check below imports into *this* process. When the CLI lives
    in a different interpreter from the one serving recall -- the normal outcome
    of a pipx install beside a Hermes venv -- a green result here says nothing
    about that runtime, and reporting it unqualified is how a degraded vector
    stack stayed invisible (#813). So the payload declares the scope it actually
    measured. Guessing a runtime interpreter from HERMES_HOME would only move the
    misleading report somewhere else, so nothing here tries.
    """

    checks: list[dict[str, str]] = []

    def add(category: str, check: str, status: str, detail: str = "") -> None:
        entry = {"category": category, "check": check, "status": status, "detail": detail}
        # Every check in this module imports into the calling interpreter, so all
        # of them carry the scope. Stamping it at the source means a new check
        # cannot be added without it (#813).
        entry["scope"] = DEP_SCOPE
        checks.append(entry)

    add("env", "python_version", "OK", sys.version.split()[0])
    add("env", "platform", "OK", platform.platform())
    # Report only the executable name: an absolute interpreter path can reveal
    # a user's home directory or virtual-environment layout in diagnostics.
    executable = Path(sys.executable).name
    add("env", "python_executable", "OK", executable)
    add("env", "checks_scope", "OK", DEP_SCOPE)

    try:
        import mnemosyne

        version = getattr(mnemosyne, "__version__", None)
        if not version:
            version = importlib.metadata.version("mnemosyne-memory")
        add("package", "mnemosyne_version", "OK", str(version))
    except Exception:
        add("package", "mnemosyne_version", "ERROR", "package version unavailable")

    for name, module in {
        "fastembed": "fastembed",
        "sqlite_vec": "sqlite_vec",
        "numpy": "numpy",
        "huggingface_hub": "huggingface_hub",
    }.items():
        try:
            dependency = __import__(module)
            add("deps", name, "OK", f"version={getattr(dependency, '__version__', 'unknown')}")
        except ImportError:
            add("deps", name, "MISSING")
        except Exception:
            add("deps", name, "ERROR", "dependency import failed")

    try:
        dependency = __import__("ctransformers")
        add("deps", "ctransformers", "OK", f"version={getattr(dependency, '__version__', 'unknown')}")
    except ImportError:
        add("deps", "ctransformers", "OPTIONAL", "optional local-GGUF fallback dependency not installed")
    except Exception:
        add("deps", "ctransformers", "ERROR", "dependency import failed")

    try:
        from mnemosyne.core import embeddings as _embeddings

        add("core", "embeddings_available", "YES" if _embeddings.available() else "NO")
        add("core", "embeddings_model", "OK", _embeddings._DEFAULT_MODEL)
        # Surface the resolved dimension so operators can confirm their
        # MNEMOSYNE_EMBEDDING_DIM / model-table resolution via the doctor,
        # complementing the fail-loud unknown-model resolver.
        add("core", "embeddings_dim", "OK", str(_embeddings.EMBEDDING_DIM))
    except Exception:
        add("core", "embeddings_available", "ERROR", "embeddings capability unavailable")

    try:
        from mnemosyne.core.beam import _SQLITE_VEC_AVAILABLE

        vec_can_load = False
        if _SQLITE_VEC_AVAILABLE:
            try:
                import sqlite_vec

                test_conn = sqlite3.connect(":memory:")
                try:
                    test_conn.enable_load_extension(True)
                    sqlite_vec.load(test_conn)
                    vec_can_load = True
                finally:
                    test_conn.close()
            except Exception:
                vec_can_load = False
        add("core", "sqlite_vec_available", "YES" if vec_can_load else "NO")
        if _SQLITE_VEC_AVAILABLE and not vec_can_load:
            add("core", "sqlite_vec_warning", "NO", "extension loading unavailable")
    except Exception:
        add("core", "sqlite_vec", "ERROR", "sqlite-vec capability unavailable")

    statuses = {entry["status"] for entry in checks}
    overall = "unavailable" if "ERROR" in statuses else "warning" if statuses & {"MISSING", "NO"} else "ok"
    return {"status": overall, "checks": checks, "scope": DEP_SCOPE, "executable": executable}