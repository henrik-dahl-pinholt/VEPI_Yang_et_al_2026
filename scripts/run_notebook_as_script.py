from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def notebook_to_python(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text())
    chunks: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines: list[str] = [f"# %% {notebook_path.name} cell {index}\n"]
        for line in source.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("%"):
                continue
            lines.append(line)
        if lines[-1] and not lines[-1].endswith("\n"):
            lines.append("\n")
        chunks.append("".join(lines))
    return "\n\n".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a notebook's code cells as a plain Python script."
    )
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output-script", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook_path = args.notebook.resolve()
    repo_root = Path.cwd().resolve()
    for path in (
        repo_root / "MS2Posterior",
        repo_root / "TwoLocusGPR",
        repo_root,
        notebook_path.parent,
    ):
        path_text = str(path)
        if path.exists() and path_text not in sys.path:
            sys.path.insert(0, path_text)
    script = notebook_to_python(notebook_path)
    if args.output_script is not None:
        args.output_script.parent.mkdir(parents=True, exist_ok=True)
        args.output_script.write_text(script)
    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
    }
    exec(compile(script, str(notebook_path), "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
