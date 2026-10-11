"""Run a module after adding workspace-local dependencies to a CUDA environment."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m alphasql.runner.cuda_bootstrap <module> [args...]")
    module = sys.argv[1]
    root = Path(__file__).resolve().parents[2]
    dependencies = root / ".gpu-deps"
    if dependencies.exists():
        sys.path.insert(0, str(dependencies))
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
