"""Remove local build/test artifacts without touching datasets or experiment runs."""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLICIT_TARGETS = (
    ".coverage",
    "coverage.xml",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "build",
    "dist",
    "src/evoshift.egg-info",
)


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        raise RuntimeError(f"refusing to clean outside project root: {resolved}") from None
    if resolved == PROJECT_ROOT:
        raise RuntimeError("refusing to clean the project root")
    return resolved


def _remove(path: Path) -> bool:
    target = _inside_project(path)
    if not target.exists() and not target.is_symlink():
        return False
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink()
    return True


def main() -> None:
    targets = [PROJECT_ROOT / value for value in EXPLICIT_TARGETS]
    targets.extend(PROJECT_ROOT.rglob("__pycache__"))
    removed = []
    for target in sorted(set(targets), key=lambda item: len(item.parts), reverse=True):
        if _remove(target):
            removed.append(target.relative_to(PROJECT_ROOT))
    if removed:
        for path in sorted(removed):
            print(f"removed {path}")
    else:
        print("nothing to clean")


if __name__ == "__main__":
    main()
