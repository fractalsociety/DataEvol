"""Measured harness evaluation through a JSONL worker subprocess."""
from __future__ import annotations

import json
import queue
import resource
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Any

from .execution_contract import ExecutionEvent, PinnedExecutionRequest, REAL_EXECUTOR_KINDS, canonical_json, replay_hash
from .executor import _benchmark_cases
from .genome import HarnessGenome
from .scoring import HarnessEvaluation, ScoreWeights, composite_score


class WorkerTimeout(TimeoutError):
    pass


class WorkerCrash(RuntimeError):
    pass


class WorkerProtocolError(ValueError):
    pass


class WorkerCancelled(RuntimeError):
    pass


class WorkerProcess:
    """One JSONL worker process with bounded diagnostics and hard timeouts."""

    _EOF = object()

    def __init__(self, argv: list[str], *, stderr_limit: int = 65_536, terminate_grace: float = 0.25) -> None:
        if not argv or not all(isinstance(value, str) and value for value in argv):
            raise ValueError("worker argv must be a non-empty list of strings")
        if stderr_limit < 1:
            raise ValueError("stderr_limit must be >= 1")
        if terminate_grace < 0:
            raise ValueError("terminate_grace must be non-negative")
        self.argv = list(argv)
        self.stderr_limit = stderr_limit
        self.terminate_grace = terminate_grace
        self.exit_status: int | None = None
        self.peak_memory_mb: float | None = None
        self._cancelled = threading.Event()
        self._state_lock = threading.Lock()
        self._stderr_lock = threading.Lock()
        self._stderr = bytearray()
        self._stderr_total = 0
        self._stdout_lines: queue.Queue[object] = queue.Queue()
        self._process = subprocess.Popen(
            self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=False, bufsize=0,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    @property
    def stderr(self) -> str:
        with self._stderr_lock:
            return bytes(self._stderr).decode("utf-8", errors="replace")

    @property
    def stderr_range(self) -> tuple[int, int]:
        with self._stderr_lock:
            return self._stderr_total - len(self._stderr), self._stderr_total

    def call(self, case: Mapping[str, Any], *, seed: int, sequence: int, timeout: float) -> dict[str, Any]:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if self._cancelled.is_set():
            raise WorkerCancelled("worker was cancelled")
        line = canonical_json({"case": dict(case), "seed": seed, "sequence": sequence}).encode() + b"\n"
        with self._state_lock:
            process = self._process
            if process.poll() is not None:
                self._record_exit()
                raise WorkerCrash(f"worker exited with status {self.exit_status}")
            try:
                assert process.stdin is not None
                process.stdin.write(line)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._record_exit()
                raise WorkerCrash(f"worker input failed with status {self.exit_status}") from exc
        try:
            item = self._stdout_lines.get(timeout=timeout)
        except queue.Empty as exc:
            self._terminate()
            raise WorkerTimeout(f"worker exceeded {timeout:.3f}s wall-clock timeout") from exc
        if self._cancelled.is_set():
            raise WorkerCancelled("worker was cancelled")
        if item is self._EOF:
            self._record_exit()
            raise WorkerCrash(f"worker exited with status {self.exit_status}")
        assert isinstance(item, bytes)
        try:
            value = json.loads(item.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkerProtocolError("worker response must be one UTF-8 JSON object line") from exc
        if not isinstance(value, dict):
            raise WorkerProtocolError("worker response must be a JSON object")
        output, tokens_in, tokens_out = value.get("output"), value.get("tokens_in"), value.get("tokens_out")
        if not isinstance(output, str):
            raise WorkerProtocolError("worker response output must be a string")
        for key, count in (("tokens_in", tokens_in), ("tokens_out", tokens_out)):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise WorkerProtocolError(f"worker response {key} must be a non-negative integer")
        return {"output": output, "tokens_in": tokens_in, "tokens_out": tokens_out}

    def close(self) -> int | None:
        with self._state_lock:
            process = self._process
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        try:
            process.wait(timeout=max(1.0, self.terminate_grace))
        except subprocess.TimeoutExpired:
            self._terminate()
        self._record_exit()
        self._join_readers()
        return self.exit_status

    def cancel(self) -> None:
        self._cancelled.set()
        self._terminate()

    def _terminate(self) -> None:
        with self._state_lock:
            process = self._process
            if process.poll() is None:
                process.terminate()
        try:
            process.wait(timeout=self.terminate_grace)
        except subprocess.TimeoutExpired:
            with self._state_lock:
                if process.poll() is None:
                    process.kill()
            process.wait()
        self._record_exit()
        self._join_readers()

    def _record_exit(self) -> None:
        status = self._process.poll()
        if status is None:
            return
        self.exit_status = status
        try:
            maximum = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        except (AttributeError, ValueError, OSError):
            return
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        self.peak_memory_mb = maximum / divisor

    def _join_readers(self) -> None:
        self._stdout_thread.join(timeout=0.2)
        self._stderr_thread.join(timeout=0.2)

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                self._stdout_lines.put(self._EOF)
                return
            self._stdout_lines.put(line)

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        while True:
            chunk = self._process.stderr.read(4096)
            if not chunk:
                return
            with self._stderr_lock:
                self._stderr_total += len(chunk)
                self._stderr.extend(chunk)
                if len(self._stderr) > self.stderr_limit:
                    del self._stderr[:len(self._stderr) - self.stderr_limit]


class SubprocessHarnessExecutor:
    """Score observed outcomes, never structural capabilities.

    Robustness is the pass fraction among case-sessions requiring more than one
    protocol attempt, or quality when none retried. Verifier agreement is the
    fraction of case-sessions for which the public verifier returned normally,
    independent of whether its verdict passed.
    """

    def __init__(
        self,
        request: PinnedExecutionRequest,
        worker_argv: list[str],
        case_verifier: Callable[[Mapping[str, Any], str], bool],
        weights: ScoreWeights | None = None,
    ) -> None:
        if not request.verify_hash():
            raise ValueError("request content_hash does not match its canonical payload")
        if not callable(case_verifier):
            raise ValueError("case_verifier must be callable")
        if not worker_argv:
            raise ValueError("worker_argv must not be empty")
        scripted, mlx = "--scripted" in worker_argv, "--mlx" in worker_argv
        if scripted and mlx:
            raise ValueError("worker argv must select exactly one worker mode")
        if scripted and request.executor_kind != "fixture":
            raise ValueError("scripted workers require executor_kind 'fixture'")
        if request.executor_kind in REAL_EXECUTOR_KINDS and not mlx:
            raise ValueError("real executor kinds require a --mlx worker")
        if mlx and "--model" not in worker_argv:
            raise ValueError("--mlx workers require --model")
        self.request = request
        self.worker_argv = list(worker_argv)
        self.case_verifier = case_verifier
        self.weights = weights or ScoreWeights()
        self.last_event_streams: dict[str, tuple[ExecutionEvent, ...]] = {}
        self.all_event_streams: dict[str, tuple[ExecutionEvent, ...]] = {}
        self._active_lock = threading.Lock()
        self._active_worker: WorkerProcess | None = None
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._active_lock:
            worker = self._active_worker
        if worker is not None:
            worker.cancel()

    def evaluate(
        self, genome: HarnessGenome, benchmark: Any, *, seed: int = 17,
        repeated_runs: int = 1, weights: ScoreWeights | None = None,
    ) -> HarnessEvaluation:
        if seed != self.request.seed:
            raise ValueError("evaluation seed must match the pinned request seed")
        cases = _benchmark_cases(benchmark)
        if not cases:
            raise ValueError("benchmark contains no cases")
        run_count = max(1, repeated_runs)
        active_weights = weights or self.weights
        self._cancelled.clear()
        self.last_event_streams = {}
        self.all_event_streams = {}
        replay_hashes: dict[str, str] = {}
        run_metrics: list[dict[str, float]] = []
        failure_categories: list[str] = []
        category_totals: dict[str, dict[str, float]] = {}
        retry_passes = retry_cases = total_tokens_in = total_tokens_out = 0

        for run_index in range(run_count):
            run_passes = run_verifier_calls = 0
            run_latency_ms = 0.0
            current_streams: dict[str, tuple[ExecutionEvent, ...]] = {}
            for case_index, case in enumerate(cases):
                if self._cancelled.is_set():
                    raise WorkerCancelled("evaluation was cancelled")
                result = self._evaluate_case(genome, case, seed, run_index, case_index)
                session_id, events = result["session_id"], result["events"]
                current_streams[self._case_key(case, case_index)] = events
                self.all_event_streams[session_id] = events
                replay_hashes[session_id] = replay_hash(events)
                passed = bool(result["passed"])
                run_passes += int(passed)
                run_verifier_calls += int(result["verifier_succeeded"])
                run_latency_ms += float(result["latency_ms"])
                total_tokens_in += int(result["tokens_in"])
                total_tokens_out += int(result["tokens_out"])
                if result["attempts"] > 1:
                    retry_cases += 1
                    retry_passes += int(passed)
                failure = result["failure_classification"]
                if failure and failure not in failure_categories:
                    failure_categories.append(failure)
                category = str(case.get("category") or case.get("benchmark_type") or "normal").lower()
                totals = category_totals.setdefault(category, {"quality": 0.0, "failed": 0.0, "count": 0.0})
                totals["quality"] += float(passed)
                totals["failed"] += float(not passed)
                totals["count"] += 1.0
            self.last_event_streams = current_streams
            quality = run_passes / len(cases)
            run_metrics.append({
                "quality": quality, "robustness": 0.0,
                "verifier_agreement": run_verifier_calls / len(cases), "cost": 0.0,
                "latency": run_latency_ms / len(cases), "failure_rate": 1.0 - quality,
            })

        mean_quality = _mean(run_metrics, "quality")
        robustness = retry_passes / retry_cases if retry_cases else mean_quality
        for metrics in run_metrics:
            metrics["robustness"] = robustness
        mean_metrics = {key: _mean(run_metrics, key) for key in run_metrics[0]}
        per_run_scores = tuple(composite_score(metrics, active_weights) for metrics in run_metrics)
        per_category = {
            category: {
                "quality": totals["quality"] / totals["count"],
                "failure_rate": totals["failed"] / totals["count"],
                "count": int(totals["count"] / run_count),
            }
            for category, totals in category_totals.items()
        }
        return HarnessEvaluation(
            genome_id=genome.genome_id, quality=mean_metrics["quality"],
            robustness=mean_metrics["robustness"], verifier_agreement=mean_metrics["verifier_agreement"],
            cost=mean_metrics["cost"], latency=mean_metrics["latency"],
            failure_rate=mean_metrics["failure_rate"], score=composite_score(mean_metrics, active_weights),
            per_category=per_category, failure_categories=tuple(failure_categories),
            run_count=run_count, per_run_scores=per_run_scores,
            raw={
                "benchmark_cases": len(cases), "executor_kind": self.request.executor_kind,
                "request_content_hash": self.request.content_hash, "replay_hashes": replay_hashes,
                "tokens_in": total_tokens_in, "tokens_out": total_tokens_out,
            },
        )

    def _evaluate_case(
        self, genome: HarnessGenome, case: Mapping[str, Any], seed: int,
        run_index: int, case_index: int,
    ) -> dict[str, Any]:
        session_id = self._session_id(seed, run_index, case_index, case)
        worker = WorkerProcess(self.worker_argv)
        with self._active_lock:
            self._active_worker = worker
        started = time.monotonic()
        attempts = tokens_in = tokens_out = 0
        output, passed, verifier_succeeded, failure = "", False, False, None
        max_attempts = max(1, min(1 + genome.recovery.max_retries, self.request.max_actions))
        try:
            while attempts < max_attempts:
                attempts += 1
                try:
                    response = worker.call(case, seed=seed, sequence=attempts - 1,
                                           timeout=float(self.request.max_wall_seconds))
                    output = response["output"]
                    tokens_in += response["tokens_in"]
                    tokens_out += response["tokens_out"]
                except WorkerTimeout:
                    failure = "TIMEOUT"
                    break
                except (WorkerCancelled, WorkerCrash):
                    failure = "WORKER_CRASH"
                    break
                except WorkerProtocolError:
                    failure = "PROTOCOL_ERROR"
                    if attempts < max_attempts:
                        continue
                    break
                try:
                    passed = bool(self.case_verifier(case, output))
                    verifier_succeeded = True
                except Exception:
                    passed = verifier_succeeded = False
                if passed:
                    failure = None
                    break
                failure = "VERIFIER_FAILED"
            exit_status = worker.close()
            if exit_status not in (0, None) and failure is None:
                passed, failure = False, "WORKER_CRASH"
        finally:
            with self._active_lock:
                if self._active_worker is worker:
                    self._active_worker = None
        latency_ms = (time.monotonic() - started) * 1000.0
        values = [
            ("ROUTE", {"case_id": self._case_key(case, case_index), "run_index": run_index, "seed": seed}),
            ("PROPOSAL", {"output": output, "attempts": attempts}),
            ("OBSERVATION", {"tokens_in": tokens_in, "tokens_out": tokens_out}),
            ("VERIFIER", {"passed": passed, "succeeded_without_error": verifier_succeeded}),
            ("REWARD", {"value": 1.0 if passed else 0.0}),
            ("SUBPROCESS_EXIT", {
                "exit_status": worker.exit_status, "stderr_range": list(worker.stderr_range),
                "failure_classification": failure, "checkpoint_identity": self.request.model_revision,
                **({"peak_memory_mb": worker.peak_memory_mb} if worker.peak_memory_mb is not None else {}),
            }),
        ]
        events = tuple(ExecutionEvent.create({
            "session_id": session_id, "request_hash": self.request.content_hash, "sequence": sequence,
            "event_type": event_type, "payload": payload, "model_identity": self.request.model_revision,
            "tokens_in": tokens_in if event_type == "PROPOSAL" else 0,
            "tokens_out": tokens_out if event_type == "PROPOSAL" else 0,
            "latency_ms": latency_ms if event_type == "SUBPROCESS_EXIT" else 0.0, "cost_usd": 0.0,
        }) for sequence, (event_type, payload) in enumerate(values))
        return {
            "session_id": session_id, "events": events, "passed": passed,
            "verifier_succeeded": verifier_succeeded, "failure_classification": failure,
            "attempts": attempts, "latency_ms": latency_ms,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
        }

    def _session_id(self, seed: int, run_index: int, case_index: int, case: Mapping[str, Any]) -> str:
        material = canonical_json({
            "request_hash": self.request.content_hash, "seed": seed, "run_index": run_index,
            "case_index": case_index, "case": dict(case),
        })
        return f"hexec_{sha256(material.encode()).hexdigest()[:24]}"

    @staticmethod
    def _case_key(case: Mapping[str, Any], case_index: int) -> str:
        return str(case["id"]) if case.get("id") is not None else f"case_{case_index}"


def _mean(metrics: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in metrics) / len(metrics)


__all__ = [
    "SubprocessHarnessExecutor", "WorkerCancelled", "WorkerCrash", "WorkerProcess",
    "WorkerProtocolError", "WorkerTimeout",
]
