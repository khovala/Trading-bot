from __future__ import annotations

import argparse
from pathlib import Path

from src.research.baseline_snapshot import build_baseline_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create baseline metrics snapshot.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", default="reports/baseline_snapshot.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    output = workspace / args.output
    path = build_baseline_snapshot(workspace, output_path=output)
    print(path.relative_to(workspace))


if __name__ == "__main__":
    main()
