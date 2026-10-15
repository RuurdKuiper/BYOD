#!/usr/bin/env python3
"""Export this checkout as a standalone source ZIP and clean directory.

No parent Git history, credentials, caches, models, or generated outputs are
copied. The exported directory can be initialized as a new Git repository.
"""
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parent
FILES = (
    ".env.example", ".gitignore", "README.md", "RELEASE_CHECKLIST.md",
    "REPRODUCIBILITY.md", "CITATION.cff", "LICENSE", "pyproject.toml",
    "train.py", "merge_adapter.py", "evaluate_benchmarks.py",
    "build_training_dataset.py", "prepare_release.py",
)
DIRECTORIES = ("src", "configs", "notebooks", "reference", "tests", ".github")


def release_files():
    for name in FILES:
        path = ROOT / name
        if path.is_file():
            yield path
    for name in DIRECTORIES:
        for path in sorted((ROOT / name).rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if any(
                p.startswith(".") or p == "__pycache__" or p.endswith(".egg-info")
                for p in path.relative_to(ROOT / name).parts
            ):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def main():
    dist = ROOT / "dist"
    destination = dist / "BYOD"
    archive = dist / "BYOD-source.zip"
    destination.mkdir(parents=True, exist_ok=True)
    files = list(release_files())
    # Refresh only files listed in the previous export; preserve a newly
    # initialized .git directory and never recursively delete user files.
    manifest = dist / "export-manifest.txt"
    if manifest.exists():
        current = {str(p.relative_to(ROOT)) for p in files}
        for name in manifest.read_text().splitlines():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
                raise SystemExit(f"Invalid export manifest entry: {name}")
            stale = destination / relative
            if name not in current and stale.is_file():
                stale.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for path in files:
            name = path.relative_to(ROOT)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            package.write(path, Path("BYOD") / name)
    manifest.write_text("".join(str(p.relative_to(ROOT)) + "\n" for p in files))
    print(f"Standalone checkout: {destination}")
    print(f"Source archive: {archive} ({len(files)} files)")
    if not (ROOT / "LICENSE").is_file():
        print("Release pending: choose a code license before publishing.")


if __name__ == "__main__":
    main()
