from __future__ import annotations

import json
import sys

from apps.trainer import run_stage


def test_run_stage_writes_json_report(tmp_path, monkeypatch) -> None:
    pipeline = tmp_path / "pipeline.yaml"
    params = tmp_path / "params.yaml"
    report = tmp_path / "reports" / "stage.json"

    pipeline.write_text(
        "stages:\n  download_market_data:\n    purpose: test\n    reports: ['reports/stage.json']\n",
        encoding="utf-8",
    )
    params.write_text("global:\n  seed: 42\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage.py",
            "--stage",
            "download_market_data",
            "--pipeline",
            "pipeline.yaml",
            "--params",
            "params.yaml",
            "--workspace",
            str(tmp_path),
        ],
    )

    run_stage.main()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["stage_name"] == "download_market_data"
    assert payload["success"] is True


def test_list_stages_includes_manifest_stage(tmp_path, monkeypatch, capsys) -> None:
    pipeline = tmp_path / "pipeline.yaml"
    params = tmp_path / "params.yaml"
    pipeline.write_text("stages:\n  run_retrain_pipeline:\n    purpose: test\n", encoding="utf-8")
    params.write_text("global:\n  seed: 42\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage.py",
            "--pipeline",
            "pipeline.yaml",
            "--params",
            "params.yaml",
            "--workspace",
            str(tmp_path),
            "--list-stages",
        ],
    )
    run_stage.main()
    out = capsys.readouterr().out
    assert "run_retrain_pipeline" in out
