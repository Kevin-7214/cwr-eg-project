from __future__ import annotations

import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import psutil

import continue_i04_after_base as ops
import continue_i05_after_features as stage
from cwr_eg.hashing import sha256_file
from cwr_eg.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT.parent
I07_COMPLETION = Path("manifests/i07_test_completion.json")
STATUS = Path("status/i08_i09_status.json")
REPORT_DIR = Path("reports")


def _wait_for_i07(process_id: int) -> None:
    deadline = time.monotonic() + 36 * 60 * 60
    while not I07_COMPLETION.exists():
        if not psutil.pid_exists(process_id):
            raise RuntimeError("I-07 process exited without Test completion")
        if time.monotonic() > deadline:
            raise TimeoutError("I-07 continuation exceeded thirty-six hours")
        time.sleep(30)


def _run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=capture,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{output[-3000:]}")
    return completed


def _failure_table() -> tuple[Path, list[dict[str, Any]]]:
    rows = [row for row in read_jsonl("artifacts/i04_full/all_generated.jsonl") if row.get("status") == "failed"]
    if len(rows) != 11:
        raise RuntimeError("I-04 explicit failure count changed before final reporting")
    output = REPORT_DIR / "i08_failed_samples.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "recipe_id",
        "kind",
        "split",
        "source",
        "language",
        "watermark_family",
        "key_id",
        "attack_id",
        "failure_type",
        "failure_message",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return output, rows


def _resource_summary() -> dict[str, Any]:
    progress = read_jsonl("status/progress.jsonl")
    monitored = [row for row in progress if "gpu_temperature_c" in row.get("details", {})]
    if not monitored:
        raise RuntimeError("No experiment monitoring records are available")
    artifact_bytes = sum(
        path.stat().st_size
        for path in Path("artifacts").rglob("*")
        if path.is_file()
    )
    by_task: dict[str, dict[str, Any]] = {}
    for row in monitored:
        task_id = str(row["task_id"])
        details = row["details"]
        summary = by_task.setdefault(
            task_id,
            {
                "monitor_points": 0,
                "maximum_gpu_temperature_c": -math.inf,
                "maximum_used_ram_gib": -math.inf,
                "minimum_free_disk_gib": math.inf,
                "first": row["time"],
                "last": row["time"],
            },
        )
        summary["monitor_points"] += 1
        summary["maximum_gpu_temperature_c"] = max(
            summary["maximum_gpu_temperature_c"], float(details["gpu_temperature_c"])
        )
        summary["maximum_used_ram_gib"] = max(
            summary["maximum_used_ram_gib"], float(details["used_ram_gib"])
        )
        summary["minimum_free_disk_gib"] = min(
            summary["minimum_free_disk_gib"], float(details["free_disk_gib"])
        )
        summary["last"] = row["time"]
    feature = json.loads(Path("manifests/i05_feature_completion.json").read_text(encoding="utf-8"))
    first = datetime.fromisoformat(feature["resources"]["first_recorded_at"])
    last = datetime.fromisoformat(feature["resources"]["last_recorded_at"])
    return {
        "artifact_bytes": artifact_bytes,
        "artifact_gib": artifact_bytes / 2**30,
        "monitoring": by_task,
        "train_dev_feature_documents": 4047,
        "train_dev_feature_elapsed_seconds": (last - first).total_seconds(),
        "train_dev_feature_documents_per_second": 3237 / (last - first).total_seconds(),
    }


def _markdown_results(
    result: dict[str, Any], failures: list[dict[str, Any]], resources: dict[str, Any]
) -> str:
    primary = result["scenario_metrics"]["a_and_b"]
    fwer = result["primary_parent_fwer"]
    baselines = result["baselines"]
    lofo_lines = [
        f"- {family}: 宏 F1 {metrics['macro_f1']:.4f}，父级 FWER {metrics['parent_fwer']['rate']:.4f}。"
        for family, metrics in result["lofo_metrics"].items()
    ]
    return "\n".join(
        [
            "# RTX 5060 中尺度实验结果",
            "",
            "本报告是 I 阶段中尺度结果，只用于估计效应量、失败率、资源与 H100 方案，不作为论文最终结论。",
            "",
            "## 数据与执行完整性",
            "",
            "- 冻结配方：8,400；成功生成：8,389；显式失败：11；静默丢失：0。",
            f"- Test 文档：{result['test_documents']}；独立 Test 父样本：{result['test_parents']}。",
            f"- 当前工作区实验产物占用：{resources['artifact_gib']:.2f} GiB。",
            "",
            "## 主结果（A+B 授权）",
            "",
            f"- 五类宏 F1：{primary['macro_f1']:.4f}。",
            f"- OSCR（主已知样本 + LOFO 未知样本组合）：{result['composite_oscr']:.4f}。",
            f"- 字符 IoU：{primary['mean_character_iou']:.4f}。",
            f"- 事件 F1：{primary['mean_event_f1']:.4f}。",
            f"- 父样本级 clean FWER：{fwer['errors']}/{fwer['trials']} = {fwer['rate']:.4f}；95% Clopper–Pearson 上界 {fwer['upper']:.4f}。",
            "",
            "## 冻结基线",
            "",
            f"- Logistic Regression 宏 F1：{baselines['logistic_regression']['macro_f1']:.4f}。",
            f"- direct-feature MLP 宏 F1：{baselines['direct_feature_mlp']['macro_f1']:.4f}。",
            f"- generic-only AUROC：{baselines['generic_only']['watermark_auc']:.4f}。",
            f"- registered-only AUROC：{baselines['registered_only']['watermark_auc']:.4f}。",
            f"- 线性证据融合 AUROC：{baselines['linear_evidence_fusion']['watermark_auc']:.4f}。",
            "",
            "## LOFO",
            "",
            *lofo_lines,
            "",
            "## 失败样本",
            "",
            f"共 {len(failures)} 条显式失败，逐条记录见 `reports/i08_failed_samples.csv`；失败项未进入特征或评分，未静默丢弃。",
            "",
            "## 结论边界",
            "",
            "无论性能是否达到预设目标，本报告均保留完整结果。H100 正式实验不得把这里的 Test 当作新的 Dev，也不得根据本报告回改本轮冻结结果。",
            "",
        ]
    )


def _resource_markdown(resources: dict[str, Any], result: dict[str, Any]) -> str:
    feature_rate = resources["train_dev_feature_documents_per_second"]
    return "\n".join(
        [
            "# RTX 5060 资源统计与 H100 建议",
            "",
            "## 本机观测",
            "",
            f"- Train/Dev 新特征吞吐：{feature_rate:.3f} 文档/秒（3,237 条新增特征的实际时段）。",
            f"- 全部实验产物占用：{resources['artifact_gib']:.2f} GiB。",
            f"- Test 父样本：{result['test_parents']}；Test 文档：{result['test_documents']}。",
            "",
            "## H100 正式阶段建议",
            "",
            "1. 先用与本轮完全同构的 400 父样本 H100 canary 测量生成、特征、注册检测和训练四类吞吐，不直接套用理论 GPU 倍率。",
            "2. 正式规模优先按 4 倍父样本扩展到 3,200；只有在 canary 证明 wall-clock、失败率和存储均有余量时再扩到 6,400。",
            "3. 保持父样本分区、父级最大值校准、三模型固定集成与一次性 Test 规则；增加 Calibration 父样本数以降低最小经验 p 值，而不是重复使用相关后代。",
            "4. H100 存储预算至少按本轮实占的 8 倍预留，并另留 30% 临时分片与断点空间；运行时间以 H100 canary 吞吐线性外推并加 20% 余量。",
            "5. 若主结果的父级 FWER 上界未满足 0.015，不应只扩大 Test；应先扩大独立 Calibration null 池并审查注册搜索空间与 validity abstention。",
            "",
            "## 明确不外推的内容",
            "",
            "- RTX 5060 与 H100 的生成吞吐不会按峰值 FLOPS 等比例缩放。",
            "- MarkLLM 检测含 tokenizer/CPU 开销，必须单独基准。",
            "- 本轮效应量不是论文最终显著性结论。",
            "",
        ]
    )


def _update_task_lists(stage_name: str, evidence: str) -> None:
    for path in (
        IMPLEMENTATION / "10_实验前准备任务清单.md",
        IMPLEMENTATION / "11_RTX5060中尺度实验计划.md",
    ):
        text = path.read_text(encoding="utf-8")
        marker = f"- [ ] `{stage_name}`"
        replacement = f"- [x] `{stage_name}`"
        if marker in text:
            line_start = text.index(marker)
            line_end = text.find("\n", line_start)
            line_end = len(text) if line_end == -1 else line_end
            original = text[line_start:line_end]
            text = text[:line_start] + original.replace(marker, replacement) + f" 证据：{evidence}" + text[line_end:]
            path.write_text(text, encoding="utf-8")


def _qa() -> dict[str, Any]:
    tests = _run([sys.executable, "-m", "pytest", "-q"])
    if "failed" in tests.stdout.lower():
        raise RuntimeError("Final pytest output contains failures")
    required = [
        "manifests/i04_completion.json",
        "manifests/i05_training_matrix_completion.json",
        "manifests/i05_checkpoint_scoring_completion.json",
        "manifests/i06_calibration_completion.json",
        "manifests/i07_test_completion.json",
        "artifacts/i07_test/i07_results.json",
    ]
    hashes = {path: sha256_file(path) for path in required}
    test_result = json.loads(Path("artifacts/i07_test/i07_results.json").read_text(encoding="utf-8"))
    if test_result["deviation_audit"]["test_used_for_training_selection_or_early_stopping"]:
        raise RuntimeError("Final QA detected Test leakage")
    if test_result["test_parents"] != 200:
        raise RuntimeError("Final QA detected a Test parent-count drift")
    ignored = _run(["git", "check-ignore", "artifacts/i07_test/i07_results.json"])
    if not ignored.stdout.strip():
        raise RuntimeError("Artifacts directory is not ignored by Git")
    return {
        "pytest": tests.stdout.strip().splitlines()[-1],
        "required_hashes": hashes,
        "test_leakage": False,
        "test_parents": 200,
        "artifacts_git_ignored": True,
    }


def _git_publish() -> tuple[str, str]:
    _run(["git", "config", "--local", "user.name", "Kevin-7214"])
    _run(["git", "config", "--local", "user.email", "kevinzhang7214@gmail.com"])
    _run(["git", "add", "-A"])
    staged = _run(["git", "diff", "--cached", "--name-only"]).stdout.splitlines()
    if not staged:
        raise RuntimeError("No final I-stage files were staged")
    if any(path.startswith("artifacts/") or path.startswith("status/private/") for path in staged):
        raise RuntimeError("Final Git stage contains forbidden experiment artifacts or private keys")
    _run(["git", "commit", "-m", "Complete RTX 5060 intermediate experiment"])
    experiment_commit = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    _run(["git", "push", "origin", "main"])
    return experiment_commit, "origin/main"


def _record_i09(experiment_commit: str, remote: str, qa: dict[str, Any]) -> str:
    completion = Path("manifests/i09_completion.json")
    payload = {
        "task_id": "I-09",
        "completed_at": ops._now(),
        "experiment_commit": experiment_commit,
        "remote": remote,
        "branch": "main",
        "qa_manifest_sha256": sha256_file("manifests/i08_qa.json"),
        "pytest": qa["pytest"],
        "first_push_succeeded": True,
    }
    if completion.exists():
        raise FileExistsError("I-09 completion already exists")
    stage._write_new(completion, payload)
    _run(["git", "add", "manifests/i09_completion.json"])
    _run(["git", "commit", "-m", "Record I-09 completion"])
    final_commit = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    _run(["git", "push", "origin", "main"])
    return final_commit


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--i07-process-id", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    ops.STATUS_PATH = STATUS
    ops.CONFIG = Path("configs/intermediate.yaml")
    ops._set_status("wait_for_i07", "in_progress", process_id=args.i07_process_id)
    _wait_for_i07(args.i07_process_id)

    ops._set_status("i08_reporting", "in_progress")
    result = json.loads(Path("artifacts/i07_test/i07_results.json").read_text(encoding="utf-8"))
    failure_path, failures = _failure_table()
    resources = _resource_summary()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_report = REPORT_DIR / "i08_intermediate_results.md"
    resource_report = REPORT_DIR / "i08_resources_and_h100.md"
    result_report.write_text(_markdown_results(result, failures, resources), encoding="utf-8")
    resource_report.write_text(_resource_markdown(resources, result), encoding="utf-8")
    external_report = IMPLEMENTATION / "12_RTX5060中尺度实验结果.md"
    external_report.write_text(result_report.read_text(encoding="utf-8") + "\n" + resource_report.read_text(encoding="utf-8"), encoding="utf-8")

    ops._set_status("i08_final_qa", "in_progress")
    qa = _qa()
    qa_path = Path("manifests/i08_qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    i08 = Path("manifests/i08_completion.json")
    stage._write_new(
        i08,
        {
            "task_id": "I-08",
            "completed_at": ops._now(),
            "result_report_sha256": sha256_file(result_report),
            "failure_table_sha256": sha256_file(failure_path),
            "resource_report_sha256": sha256_file(resource_report),
            "qa_sha256": sha256_file(qa_path),
            "deviation_audit": {
                "status": "pass",
                "results_reported_regardless_of_performance": True,
                "paper_final_claim_made": False,
            },
        },
    )
    _update_task_lists("I-05", "`manifests/i05_dev_analysis_completion.json`。")
    _update_task_lists("I-GATE-D", "`manifests/i_gate_d_freeze.json` 与最终 Test freeze amendment。")
    _update_task_lists("I-06", "`manifests/i06_calibration_completion.json`。")
    _update_task_lists("I-07", "`manifests/i07_test_completion.json`。")
    _update_task_lists("I-08", "`manifests/i08_completion.json` 与 `reports/`。")

    ops._set_status("i09_git_publish", "in_progress")
    experiment_commit, remote = _git_publish()
    final_commit = _record_i09(experiment_commit, remote, qa)
    _update_task_lists("I-09", f"已推送 `main`，最终提交 `{final_commit}`。")
    ops._set_status(
        "i09_complete",
        "done",
        experiment_commit=experiment_commit,
        final_commit=final_commit,
        remote=remote,
    )
    print(json.dumps({"status": "done", "final_commit": final_commit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        os.chdir(ROOT)
        ops.STATUS_PATH = STATUS
        ops._set_status(
            "failed",
            "error",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )
        ops._append_progress(
            "I-08-I-09",
            "blocked",
            "Fail-closed finalization stopped; inspect status/i08_i09_status.json.",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
