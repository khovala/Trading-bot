from __future__ import annotations

import argparse
from pathlib import Path

from src.training.pipeline.manifest import load_params, load_pipeline_manifest
from src.training.pipeline.registry import StageRegistry
from src.training.pipeline.runner import PipelineRunner
from src.training.stages.register import register_default_stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single pipeline stage.")
    parser.add_argument("--stage", help="Stage name from pipeline.yaml")
    parser.add_argument("--pipeline", default="pipeline.yaml")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--strict-inputs", action="store_true")
    parser.add_argument("--list-stages", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    pipeline_cfg = load_pipeline_manifest(workspace / args.pipeline)
    params = load_params(workspace / args.params)

    registry = StageRegistry()
    register_default_stages(registry)
    registry.register_from_manifest(pipeline_cfg)
    if args.list_stages:
        for stage_name in registry.list_names():
            print(stage_name)
        return

    if not args.stage:
        raise SystemExit("--stage is required unless --list-stages is used.")

    runner = PipelineRunner(
        workspace=workspace,
        manifest=pipeline_cfg,
        params=params,
        registry=registry,
    )
    result = runner.run_stage(args.stage, fail_on_missing_inputs=args.strict_inputs)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
