from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from cwr_eg.progress import append_progress


GIB = 1024**3


class ResourceLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    maximum_gpu_temperature_c: float = 85.0
    temperature_grace_seconds: float = 120.0
    maximum_ram_gib: float = 26.0
    minimum_free_disk_gib: float = 100.0
    status_interval_percent: int = 1
    status_interval_minutes: float = 5.0


class ExperimentMonitor:
    def __init__(
        self,
        *,
        task_id: str,
        limits: ResourceLimits,
        disk_path: str | Path,
        progress_path: str | Path,
        device: str,
    ) -> None:
        self.task_id = task_id
        self.limits = limits
        resolved_disk_path = Path(disk_path).resolve()
        while not resolved_disk_path.exists() and resolved_disk_path.parent != resolved_disk_path:
            resolved_disk_path = resolved_disk_path.parent
        self.disk_path = resolved_disk_path
        self.progress_path = Path(progress_path)
        self.device = device
        self._temperature_high_since: float | None = None
        self._last_report_time = 0.0
        self._last_report_percent = -limits.status_interval_percent

    @classmethod
    def from_scope(
        cls, scope: dict[str, Any], *, task_id: str, disk_path: str | Path
    ) -> "ExperimentMonitor | None":
        payload = scope.get("monitoring")
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("monitoring scope must be an object")
        return cls(
            task_id=str(scope.get("task_id", task_id)),
            limits=ResourceLimits(
                maximum_gpu_temperature_c=float(payload["maximum_gpu_temperature_c"]),
                temperature_grace_seconds=float(payload["temperature_grace_seconds"]),
                maximum_ram_gib=float(payload["maximum_ram_gib"]),
                minimum_free_disk_gib=float(payload["minimum_free_disk_gib"]),
                status_interval_percent=int(payload.get("status_interval_percent", 1)),
                status_interval_minutes=float(payload.get("status_interval_minutes", 5)),
            ),
            disk_path=disk_path,
            progress_path=str(payload.get("progress_path", "status/progress.jsonl")),
            device=str(scope.get("device", "cpu")),
        )

    def _gpu_temperature(self) -> float | None:
        if not self.device.startswith("cuda"):
            return None
        device_index = self.device.partition(":")[2] or "0"
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(completed.stdout.strip().splitlines()[0])

    def check_resources(self) -> dict[str, float | None]:
        import psutil

        now = time.monotonic()
        temperature = self._gpu_temperature()
        if temperature is not None and temperature >= self.limits.maximum_gpu_temperature_c:
            if self._temperature_high_since is None:
                self._temperature_high_since = now
            elif now - self._temperature_high_since >= self.limits.temperature_grace_seconds:
                raise ResourceLimitExceeded(
                    f"GPU temperature remained at least {temperature:.1f}C for two minutes"
                )
        else:
            self._temperature_high_since = None
        used_ram_gib = float(psutil.virtual_memory().used / GIB)
        if used_ram_gib > self.limits.maximum_ram_gib:
            raise ResourceLimitExceeded(
                f"RAM usage {used_ram_gib:.2f} GiB exceeds the approved limit"
            )
        free_disk_gib = float(shutil.disk_usage(self.disk_path).free / GIB)
        if free_disk_gib < self.limits.minimum_free_disk_gib:
            raise ResourceLimitExceeded(
                f"Free disk {free_disk_gib:.2f} GiB is below the approved limit"
            )
        return {
            "gpu_temperature_c": temperature,
            "used_ram_gib": used_ram_gib,
            "free_disk_gib": free_disk_gib,
        }

    def update(self, *, phase: str, completed: int, total: int) -> None:
        if total < 1 or not 0 <= completed <= total:
            raise ValueError("Progress counters are invalid")
        resources = self.check_resources()
        percent = math.floor(100 * completed / total)
        now = time.monotonic()
        due_percent = percent >= (
            self._last_report_percent + self.limits.status_interval_percent
        )
        due_time = now - self._last_report_time >= self.limits.status_interval_minutes * 60
        if due_percent or due_time or completed == total:
            append_progress(
                self.progress_path,
                task_id=self.task_id,
                status="done" if completed == total else "in_progress",
                evidence=f"{phase}: {completed}/{total} ({percent}%)",
                details={"phase": phase, "completed": completed, "total": total, **resources},
            )
            self._last_report_percent = percent
            self._last_report_time = now
