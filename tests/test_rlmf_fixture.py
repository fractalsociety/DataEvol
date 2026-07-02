from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataevol.api.app import create_app
from dataevol.config import DataEvolConfig
from dataevol.rlmf import (
    build_benchmark_fixture,
    build_judge_dashboard_panel,
    build_judge_disagreement_report,
    build_judge_request,
    build_calibration_benchmark_report,
    build_mds_selection_certificate,
    build_model_comparison_report,
    build_metacognitive_sft_rows,
    build_mlx_lora_manifest,
    build_promotion_fixture,
    build_qwen_benchmark_receipt,
    build_qlora_manifest,
    build_rlmf_fixture_dataset,
    build_rlmf_job_state,
    build_rlmf_reward_signal,
    build_training_command_record,
    call_qwen_judge_openai_compatible,
    emit_model_promotion_record,
    enforce_g16_promotion_gate,
    enforce_rlmf_promotion_gate,
    export_failure_assets,
    export_metacognitive_sft_dataset,
    extract_acceptance_criteria,
    generate_failure_classifier_benchmark,
    generate_intrinsic_confidence_dataset,
    load_qwen_judge_golden_output,
    parse_qwen_judge_fixture,
    parse_structured_qwen_judge_output,
    persist_judge_log,
    prepare_rlmf_fixture,
    qwen_judge_model_profile,
    qwen_judge_prompt_template,
    run_rlmf_training_fixture,
    select_judge_server_with_fallback,
    trainer_profile,
    validate_public_rlmf_rows,
    validate_rlmf_dataset_for_training,
    vllm_judge_server_profile,
    build_vllm_judge_health,
)


def config(tmp_path: Path) -> DataEvolConfig:
    return DataEvolConfig(
        path=tmp_path / "dataevol.toml",
        db_path=tmp_path / ".dataevol/dataevol.sqlite3",
        raw_path=tmp_path / ".dataevol/raw",
        artifacts_path=tmp_path / ".dataevol/artifacts",
        api_token="secret",
    )


def test_rlmf_fixture_dataset_and_training_manifests_are_deterministic(tmp_path: Path) -> None:
    first = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    second = build_rlmf_fixture_dataset(tmp_path / "rlmf_again", count=8)

    assert first.manifest["dataset_type"] == "rlmf.metacog.fixture"
    assert first.manifest["row_count"] == 8
    assert first.manifest["privacy"]["contains_private_user_data"] is False
    assert first.manifest["dataset_manifest_hash"] == second.manifest["dataset_manifest_hash"]
    assert first.train_path.exists()
    assert json.loads(first.train_path.read_text(encoding="utf-8").splitlines()[0])["acceptance_criteria"]

    mlx = build_mlx_lora_manifest(tmp_path / "rlmf", first.manifest, python_bin="python", experts=("failure_classifier",), iters=1)
    assert mlx["trainer"] == "mlx-lora"
    assert mlx["dry_run"] is True
    assert mlx["jobs"][0]["command"][:4] == ["python", "-m", "mlx_lm", "lora"]
    assert mlx["dataset_manifest_hash"] == first.manifest["dataset_manifest_hash"]
    assert Path(mlx["manifest_path"]).exists()

    qlora = build_qlora_manifest(tmp_path / "rlmf", first.manifest)
    assert qlora["trainer"] == "hf-trl-qlora"
    assert "--load_in_4bit" in qlora["command"]
    assert qlora["memory_profile"]["quantization"] == "4bit"
    assert qlora["memory_profile"]["batch_size"] == 4


def test_rlmf_dataset_export_writes_deterministic_splits_and_hashes(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=10)

    train = _read_jsonl(dataset.train_path)
    valid = _read_jsonl(dataset.valid_path)
    test = _read_jsonl(dataset.test_path)

    assert [row["task_id"] for row in train] == [f"rlmf-fixture-{index:03d}" for index in range(6)]
    assert [row["task_id"] for row in valid] == ["rlmf-fixture-006", "rlmf-fixture-007"]
    assert [row["task_id"] for row in test] == ["rlmf-fixture-008", "rlmf-fixture-009"]
    assert dataset.manifest["splits"]["train"]["rows"] == 6
    assert dataset.manifest["splits"]["valid"]["rows"] == 2
    assert dataset.manifest["splits"]["test"]["rows"] == 2
    assert dataset.manifest["splits"]["train"]["sha256"] == _file_sha256(dataset.train_path)
    assert dataset.manifest["splits"]["valid"]["sha256"] == _file_sha256(dataset.valid_path)
    assert dataset.manifest["splits"]["test"]["sha256"] == _file_sha256(dataset.test_path)
    assert dataset.manifest["splits"]["train"]["source_trace_hashes"]
    assert dataset.manifest["splits"]["train"]["redaction_statuses"] == ["not_required"]
    assert dataset.manifest["splits"]["train"]["consent"] == ["not_required"]
    assert dataset.manifest["splits"]["train"]["provenance"]["contains_private_user_data"] is False


def test_metacognitive_sft_exporter_maps_trace_fields_and_privacy_metadata(tmp_path: Path) -> None:
    traces = [
        {
            "id": f"trace-{index}",
            "task_id": f"task-{index}",
            "prompt": f"Answer task {index}. Acceptance Criteria:\n- cite evidence\n- mark confidence",
            "response": f"answer {index}",
            "reasoning_summary": "checked the evidence",
            "confidence": 0.7,
            "self_judged_success_probability": 0.75,
            "verification_result": "accepted",
            "privacy_status": "public_benchmark",
            "redaction_status": "redacted",
            "consent": "granted",
            "run_id": "run-a",
        }
        for index in range(6)
    ]

    dataset = export_metacognitive_sft_dataset(traces, tmp_path / "sft")
    rows = _read_jsonl(dataset.train_path) + _read_jsonl(dataset.valid_path) + _read_jsonl(dataset.test_path)

    assert dataset.manifest["dataset_type"] == "rlmf.metacog.sft"
    assert dataset.manifest["privacy"]["statuses"] == ["public_benchmark"]
    assert dataset.manifest["splits"]["train"]["redaction_statuses"] == ["redacted"]
    assert dataset.manifest["splits"]["train"]["consent"] == ["granted"]
    assert rows[0]["task"] == "Answer task 0. Acceptance Criteria:\n- cite evidence\n- mark confidence"
    assert rows[0]["answer"] == "answer 0"
    assert rows[0]["reasoning_summary"] == "checked the evidence"
    assert rows[0]["acceptance_criteria"] == ["cite evidence", "mark confidence"]
    assert rows[0]["source_trace_hash"]


def test_acceptance_criteria_extraction_supports_lists_text_blocks_and_fallback() -> None:
    assert extract_acceptance_criteria({"acceptance_criteria": ["pass tests", "update docs"]}) == ["pass tests", "update docs"]
    assert extract_acceptance_criteria({"prompt": "Do work. Acceptance Criteria:\n1. deterministic\n2. verified"}) == ["deterministic", "verified"]
    assert extract_acceptance_criteria({"prompt": "Do work."}) == ["Satisfy the task request and pass verification."]


def test_failure_assets_cover_overconfidence_hedges_traps_and_missed_escalation(tmp_path: Path) -> None:
    rows = [
        {"task_id": "a", "confidence": 0.95, "verification_result": "failed", "source_trace_hash": "a" * 64},
        {"task_id": "b", "confidence": 0.4, "verification_result": "accepted", "failure_type": "unfaithful_hedge", "source_trace_hash": "b" * 64},
        {"task_id": "c", "confidence": 0.6, "verification_result": "failed", "hidden_trap_failure": True, "source_trace_hash": "c" * 64},
        {"task_id": "d", "confidence": 0.6, "verification_result": "accepted", "should_escalate": True, "answer": "proceed", "source_trace_hash": "d" * 64},
    ]

    report = export_failure_assets(rows, tmp_path / "assets")

    assert report["asset_count"] == 4
    assert report["asset_types"] == [
        "hidden_trap_failure",
        "missed_escalation",
        "overconfident_failure",
        "unfaithful_hedge",
    ]
    assert Path(report["path"]).exists()
    assert all(len(asset["evidence_hash"]) == 64 for asset in report["assets"])


def test_k_sample_intrinsic_confidence_dataset_for_mds_scoring(tmp_path: Path) -> None:
    samples = [
        {"task_id": "cell-a", "confidence": 0.9, "verification_result": "accepted", "source_trace_hash": "a" * 64},
        {"task_id": "cell-a", "confidence": 0.7, "verification_result": "failed", "source_trace_hash": "b" * 64},
        {"task_id": "cell-b", "confidence": 0.5, "verification_result": "accepted", "source_trace_hash": "c" * 64},
    ]

    dataset = generate_intrinsic_confidence_dataset(samples, tmp_path / "mds")

    assert dataset["dataset_type"] == "rlmf.intrinsic_confidence.mds"
    assert dataset["row_count"] == 2
    assert dataset["rows"][0]["task_id"] == "cell-a"
    assert dataset["rows"][0]["sample_count"] == 2
    assert dataset["rows"][0]["mean_confidence"] == 0.8
    assert dataset["rows"][0]["acceptance_rate"] == 0.5
    assert len(dataset["rows"][0]["mds_row_hash"]) == 64
    assert Path(dataset["path"]).exists()


def test_rlmf_public_privacy_gate_rejects_private_rows(tmp_path: Path) -> None:
    public_row = {
        "task_id": "public",
        "privacy_status": "public_benchmark",
        "contains_private_user_data": False,
    }
    validate_public_rlmf_rows([public_row])

    with pytest.raises(PermissionError):
        validate_public_rlmf_rows([{**public_row, "task_id": "private", "privacy_status": "local_only"}])

    with pytest.raises(PermissionError):
        export_metacognitive_sft_dataset(
            [
                {
                    "id": "private-trace",
                    "prompt": "secret",
                    "response": "secret",
                    "privacy_status": "local_only",
                    "contains_private_user_data": True,
                }
            ],
            tmp_path / "private",
        )


def test_rlmf_dataset_privacy_gate_keeps_fixture_public_and_non_private(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    rows = _read_jsonl(dataset.train_path) + _read_jsonl(dataset.valid_path) + _read_jsonl(dataset.test_path)

    assert dataset.manifest["privacy"] == {
        "mode": "fixture-public",
        "contains_private_user_data": False,
        "consent_required": False,
    }
    assert {row["privacy_status"] for row in rows} == {"fixture-public"}
    assert all("private_user_content" not in json.dumps(row, sort_keys=True) for row in rows)


def test_rlmf_dataset_manifest_hash_is_path_independent_and_split_sensitive(tmp_path: Path) -> None:
    first = build_rlmf_fixture_dataset(tmp_path / "a" / "rlmf", count=8)
    second = build_rlmf_fixture_dataset(tmp_path / "b" / "rlmf", count=8)
    larger = build_rlmf_fixture_dataset(tmp_path / "c" / "rlmf", count=9)

    assert first.manifest["dataset_manifest_hash"] == second.manifest["dataset_manifest_hash"]
    assert first.manifest["dataset_manifest_hash"] != larger.manifest["dataset_manifest_hash"]
    assert first.manifest["splits"]["train"]["path"] != second.manifest["splits"]["train"]["path"]


def test_qlora_manifest_generation_selects_4bit_memory_profile_and_training_flags(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    qlora = build_qlora_manifest(
        tmp_path / "rlmf",
        dataset.manifest,
        base_model="Qwen/Qwen2.5-14B-Instruct",
        output_adapter="fixture-adapter",
    )

    assert qlora["trainer"] == "hf-trl-qlora"
    assert qlora["dry_run"] is True
    assert qlora["base_model"] == "Qwen/Qwen2.5-14B-Instruct"
    assert qlora["framework_stack"] == ["accelerate", "bitsandbytes", "peft", "transformers", "trl"]
    assert qlora["dataset_manifest_hash"] == dataset.manifest["dataset_manifest_hash"]
    assert qlora["memory_profile"]["quantization"] == "4bit"
    assert qlora["memory_profile"]["estimated_model_memory_gb"] == 8.52
    assert qlora["memory_profile"]["estimated_step_memory_gb"] == 10.72
    assert qlora["memory_profile"]["batch_size"] == 2
    assert qlora["memory_profile"]["gradient_accumulation_steps"] == 2
    assert qlora["memory_profile"]["sequence_length"] == 1024
    assert qlora["memory_profile"]["target"] == "single_cuda_or_remote_worker"
    assert qlora["command"][:3] == ["accelerate", "launch", "-m"]
    assert _flag_value(qlora["command"], "--model_name") == "Qwen/Qwen2.5-14B-Instruct"
    assert _flag_value(qlora["command"], "--dataset_text_field") == "prompt"
    assert _flag_value(qlora["command"], "--load_in_4bit") == "true"
    assert _flag_value(qlora["command"], "--use_peft") == "true"
    assert _flag_value(qlora["command"], "--lora_r") == "16"
    assert _flag_value(qlora["command"], "--per_device_train_batch_size") == "2"
    assert _flag_value(qlora["command"], "--gradient_accumulation_steps") == "2"
    assert _flag_value(qlora["command"], "--max_seq_length") == "1024"
    assert _flag_value(qlora["command"], "--output_dir").endswith("qlora/fixture-adapter")
    assert len(qlora["manifest_hash"]) == 64
    assert json.loads(Path(qlora["manifest_path"]).read_text(encoding="utf-8"))["manifest_hash"] == qlora["manifest_hash"]


def test_qlora_manifest_scales_down_to_batch_size_one_for_larger_models(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    qlora = build_qlora_manifest(
        tmp_path / "rlmf",
        dataset.manifest,
        base_model="Qwen/Qwen2.5-32B-Instruct",
    )

    assert qlora["memory_profile"]["batch_size"] == 1
    assert qlora["memory_profile"]["gradient_accumulation_steps"] == 4
    assert _flag_value(qlora["command"], "--per_device_train_batch_size") == "1"
    assert _flag_value(qlora["command"], "--gradient_accumulation_steps") == "4"


def test_rlmf_trainer_profiles_and_mlx_execute_manifest_cover_failure_classifier_and_calibration(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    manifest = build_mlx_lora_manifest(
        tmp_path / "rlmf",
        dataset.manifest,
        experts=("failure_classifier", "calibration_expert"),
        profile="mac_mlx",
        execute=True,
    )

    assert trainer_profile("mac_mlx")["trainer"] == "mlx-lora"
    assert trainer_profile("single_cuda")["trainer"] == "hf-trl-qlora"
    assert trainer_profile("multi_gpu_accelerate")["device"] == "multi_cuda"
    assert trainer_profile("remote_worker")["supports_execute"] is False
    assert manifest["status"] == "ready_to_execute"
    assert manifest["dry_run"] is False
    assert manifest["execute"] is True
    assert [job["expert"] for job in manifest["jobs"]] == ["failure_classifier", "calibration_expert"]
    assert manifest["jobs"][0]["job_type"] == "rlmf.failure_classifier.mlx_lora"
    assert manifest["jobs"][0]["checkpoint_path"].endswith("adapters/failure_classifier/adapters.safetensors")


def test_qlora_multi_gpu_profile_and_command_record_support_execute_mode(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    qlora = build_qlora_manifest(
        tmp_path / "rlmf",
        dataset.manifest,
        profile="multi_gpu_accelerate",
        execute=True,
        available_memory_gb=48,
    )
    command_record = build_training_command_record(qlora, execute=True, job_id="job-1")

    assert qlora["status"] == "ready_to_execute"
    assert qlora["profile"]["profile"] == "multi_gpu_accelerate"
    assert "--multi_gpu" in qlora["command"]
    assert qlora["checkpoint_path"].endswith("adapter_model.safetensors")
    assert command_record["job_id"] == "job-1"
    assert command_record["mode"] == "execute"
    assert command_record["would_execute"] is False
    assert len(command_record["command_hash"]) == 64


def test_resumable_job_state_records_checkpoint_metrics_logs_and_artifact_hashes() -> None:
    state = build_rlmf_job_state(
        "job-a",
        status="running",
        process_id=123,
        checkpoint_path=".dataevol/rlmf/checkpoint",
        last_metric={"loss": 0.12},
        logs=["started", "checkpoint saved"],
        artifact_hashes={"dataset_manifest_hash": "d" * 64},
    )

    assert state["job_id"] == "job-a"
    assert state["process_id"] == 123
    assert state["checkpoint_path"].endswith("checkpoint")
    assert state["last_metric"]["loss"] == 0.12
    assert state["logs"] == ["started", "checkpoint saved"]
    assert state["artifact_hashes"]["dataset_manifest_hash"] == "d" * 64
    assert len(state["job_state_hash"]) == 64


def test_deterministic_training_runner_and_promotion_gate(tmp_path: Path) -> None:
    run = run_rlmf_training_fixture(tmp_path / "run", execute=False)

    assert run["fixture_mode"] is True
    assert run["status"] == "planned"
    assert run["commands"]["mlx_lora"]["mode"] == "dry_run"
    assert run["job_state"]["status"] == "planned"
    assert run["promotion_gate"]["promoted"] is True
    assert len(run["training_run_hash"]) == 64

    bad_benchmark = {
        **run["prepared"]["benchmark"],
        "passed": True,
        "regression_passed": False,
        "metrics": {
            **run["prepared"]["benchmark"]["metrics"],
            "adaptive_ece": 0.4,
            "confidence_faithfulness": 0.5,
            "high_confidence_failure_rate": 0.5,
        },
    }
    blocked = enforce_rlmf_promotion_gate(bad_benchmark)
    assert blocked["promoted"] is False
    assert blocked["status"] == "blocked"
    assert blocked["rollback_metadata"]["required"] is True
    assert set(blocked["reasons"]) == {
        "adaptive_ece_regression",
        "confidence_faithfulness_regression",
        "high_confidence_failure_rate_regression",
        "regression_suite_failed",
    }


def test_training_manifests_require_public_privacy_gate(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    gate = validate_rlmf_dataset_for_training(dataset.manifest, dataset.rows)

    assert gate["status"] == "passed"
    assert gate["dataset_manifest_hash"] == dataset.manifest["dataset_manifest_hash"]

    mlx = build_mlx_lora_manifest(tmp_path / "rlmf", dataset.manifest)
    qlora = build_qlora_manifest(tmp_path / "rlmf", dataset.manifest)
    assert mlx["privacy_gate"]["privacy_gate_hash"] == gate["privacy_gate_hash"]
    assert qlora["privacy_gate"]["privacy_gate_hash"] == gate["privacy_gate_hash"]

    private_manifest = {
        **dataset.manifest,
        "privacy": {
            "mode": "private",
            "statuses": ["local_only"],
            "contains_private_user_data": False,
            "consent_required": False,
        },
    }
    with pytest.raises(PermissionError):
        build_qlora_manifest(tmp_path / "blocked", private_manifest)


def test_vllm_judge_profiles_and_health_checks_cover_readiness_model_context_batch_and_generation() -> None:
    local = vllm_judge_server_profile("local_cuda")
    remote = vllm_judge_server_profile("remote_host")
    compatible = vllm_judge_server_profile("openai_compatible")

    assert local["mode"] == "vllm-local"
    assert remote["mode"] == "vllm-remote"
    assert compatible["mode"] == "openai-compatible"

    health = build_vllm_judge_health("local_cuda")
    assert health["status"] == "ready"
    assert health["ready"] is True
    assert health["model_loaded"] is True
    assert health["max_context"] >= 4096
    assert health["batch_capacity"] >= 1
    assert {"temperature", "top_p", "max_tokens"} <= set(health["generation_parameters"])
    assert len(health["server_config_hash"]) == 64

    unavailable = build_vllm_judge_health("local_cuda", observed={"model_loaded": False})
    assert unavailable["status"] == "unavailable"


def test_judge_schema_log_persistence_retry_fallback_and_dashboard_panel(tmp_path: Path) -> None:
    request = build_judge_request(
        "Solve task",
        "Answer with 0.8 confidence",
        acceptance_criteria=["cite evidence", "mark confidence"],
        rubric="confidence_faithfulness",
        model_profile="qwen_high_quality",
    )
    response = parse_structured_qwen_judge_output(
        {
            "verdict": "pass",
            "confidence_faithfulness": 0.9,
            "acceptance_criteria_score": 0.8,
            "escalation_correctness": 0.75,
            "failure_classification": 0.7,
            "rationale": "confidence matches evidence",
        }
    )
    health = build_vllm_judge_health("remote_host")
    log = persist_judge_log(
        tmp_path,
        request,
        response,
        health,
        latency_ms=12.3456,
        token_counts={"prompt": 10, "completion": 8, "total": 18},
    )

    assert request["request_type"] == "rlmf.judge"
    assert request["required_scores"] == [
        "confidence_faithfulness",
        "acceptance_criteria_score",
        "escalation_correctness",
        "failure_classification",
    ]
    assert len(request["prompt_hash"]) == 64
    assert response["structured_output_valid"] is True
    assert set(response["scores"]) == {
        "confidence_faithfulness",
        "acceptance_criteria_score",
        "escalation_correctness",
        "failure_classification",
    }
    assert log["model_id"] == response["judge_model"]
    assert log["latency_ms"] == 12.346
    assert log["token_counts"]["total"] == 18
    assert Path(log["path"]).exists()

    fallback = select_judge_server_with_fallback(
        ["local_cuda", "remote_host"],
        observed_health={"local_cuda": {"ready": False}, "remote_host": {"ready": True}},
    )
    assert fallback["selected_profile"] == "remote_host"
    assert fallback["fallback_used"] is True

    panel = build_judge_dashboard_panel([health], [log])
    assert panel["panel"] == "rlmf_judge_servers"
    assert panel["ready_servers"] == 1
    assert panel["recent_decisions"][0]["decision_hash"] == log["decision_hash"]


def test_qwen_profiles_prompts_and_structured_parser_validation() -> None:
    high_quality = qwen_judge_model_profile("qwen_high_quality")
    dev = qwen_judge_model_profile("qwen_dev")
    prompt_kinds = [
        "calibration",
        "confidence_faithfulness",
        "hidden_trap_detection",
        "acceptance_criteria_grading",
    ]

    assert high_quality["quality_tier"] == "high"
    assert dev["quality_tier"] == "dev"
    assert all(qwen_judge_prompt_template(kind)["prompt_hash"] for kind in prompt_kinds)

    parsed = parse_structured_qwen_judge_output(
        {
            "verdict": "review",
            "confidence_faithfulness": "1.0",
            "acceptance_criteria_score": "0.5",
            "escalation_correctness": "0.25",
            "failure_classification": "0.0",
        }
    )
    assert parsed["verdict"] == "review"
    assert parsed["scores"]["confidence_faithfulness"] == 1.0
    assert len(parsed["response_hash"]) == 64

    with pytest.raises(ValueError):
        parse_structured_qwen_judge_output({"verdict": "unknown"})


def test_qwen_golden_fixture_output_parses_from_saved_json() -> None:
    fixture = Path(__file__).parent / "fixtures" / "qwen_judge_outputs" / "confidence_faithfulness_pass.json"

    parsed = load_qwen_judge_golden_output(fixture)

    assert parsed["verdict"] == "pass"
    assert parsed["judge_model"] == "Qwen/Qwen2.5-72B-Instruct"
    assert parsed["scores"]["confidence_faithfulness"] == 0.9
    assert parsed["scores"]["acceptance_criteria_score"] == 0.8
    assert parsed["golden_fixture_path"].endswith("confidence_faithfulness_pass.json")
    assert len(parsed["golden_fixture_hash"]) == 64


def test_qwen_openai_compatible_adapter_builds_chat_completion_request_and_parses_response() -> None:
    request = build_judge_request("Task", "Answer", acceptance_criteria=["pass"], model_profile="qwen_dev")
    captured: dict[str, object] = {}

    def transport(endpoint: str, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
        captured["endpoint"] = endpoint
        captured["headers"] = headers
        captured["body"] = body
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "verdict": "pass",
                                "confidence_faithfulness": 0.85,
                                "acceptance_criteria_score": 0.75,
                                "escalation_correctness": 1.0,
                                "failure_classification": 0.5,
                            }
                        )
                    }
                }
            ]
        }

    response = call_qwen_judge_openai_compatible(
        request,
        "openai_compatible",
        api_key="test-key",
        transport=transport,
    )

    assert captured["endpoint"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert response["adapter"] == "openai-compatible-chat-completions"
    assert response["request_hash"] == request["request_hash"]
    assert response["scores"]["confidence_faithfulness"] == 0.85
    assert len(response["raw_response_hash"]) == 64


def test_qwen_judge_connects_to_benchmark_and_fractalwork_mcl_receipt(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    judge = parse_qwen_judge_fixture({"verdict": "pass", "calibration": 0.9, "confidence_faithfulness": 0.9})

    receipt = build_qwen_benchmark_receipt(dataset.manifest, judge)

    assert receipt["receipt_type"] == "fractalwork.mcl.rlmf_judge"
    assert receipt["dataset_manifest_hash"] == dataset.manifest["dataset_manifest_hash"]
    assert receipt["judge_report_hash"] == judge["judge_report_hash"]
    assert receipt["benchmark"]["passed"] is True
    assert len(receipt["benchmark_report_hash"]) == 64
    assert len(receipt["receipt_hash"]) == 64


def test_judge_disagreement_report_compares_qwen_verifier_adjudication_and_trap() -> None:
    qwen = {"verdict": "pass", "response_hash": "q" * 64}
    report = build_judge_disagreement_report(
        qwen,
        [
            {"source": "verifier", "verification_result": "accepted", "source_trace_hash": "a" * 64},
            {"source": "adjudication", "verification_result": "accepted", "source_trace_hash": "b" * 64},
            {"source": "trap", "verification_result": "failed", "source_trace_hash": "c" * 64},
        ],
    )

    assert report["requires_adjudication"] is True
    assert report["source_summaries"]["verifier"]["pass_rate"] == 1.0
    assert report["source_summaries"]["adjudication"]["pass_rate"] == 1.0
    assert report["source_summaries"]["trap"]["pass_rate"] == 0.0
    assert report["disagreements"][0]["source"] == "trap"
    assert len(report["disagreement_hash"]) == 64


def test_rlmf_reward_signal_shapes_correctness_calibration_faithfulness_penalty_and_escalation() -> None:
    good = build_rlmf_reward_signal(
        {"verification_result": "accepted", "confidence": 0.8, "should_escalate": False, "source_trace_hash": "a" * 64},
        {"scores": {"confidence_faithfulness": 0.9}, "judge_report_hash": "j" * 64},
    )
    bad = build_rlmf_reward_signal(
        {"verification_result": "failed", "confidence": 0.95, "should_escalate": True, "answer": "proceed", "source_trace_hash": "b" * 64},
        {"scores": {"confidence_faithfulness": 0.2}, "judge_report_hash": "k" * 64},
    )

    assert good["components"]["correctness"] == 1.0
    assert good["components"]["calibration"] == 0.8
    assert good["components"]["confidence_faithfulness"] == 0.9
    assert good["components"]["overconfidence_penalty"] == 0.0
    assert good["components"]["escalation_recall"] == 1.0
    assert bad["components"]["correctness"] == 0.0
    assert bad["components"]["overconfidence_penalty"] == pytest.approx(0.25)
    assert bad["components"]["escalation_recall"] == 0.0
    assert good["reward"] > bad["reward"]
    assert len(good["reward_hash"]) == 64


def test_failure_classifier_benchmark_and_calibration_report_include_required_metrics(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    samples = [
        {"task_id": "a", "source": "trap", "cell_id": "cell-a", "verification_result": "failed", "confidence": 0.9, "should_escalate": True, "answer": "proceed", "failure_type": "overconfident_failure", "source_trace_hash": "a" * 64},
        {"task_id": "b", "source": "trap", "cell_id": "cell-a", "verification_result": "accepted", "confidence": 0.7, "should_escalate": False, "answer": "proceed", "source_trace_hash": "b" * 64},
        {"task_id": "c", "source": "verifier", "cell_id": "cell-b", "verification_result": "accepted", "confidence": 0.8, "should_escalate": True, "answer": "Escalate to verifier", "source_trace_hash": "c" * 64},
        {"task_id": "d", "source": "adjudication", "cell_id": "cell-c", "verification_result": "failed", "confidence": 0.4, "should_escalate": False, "answer": "proceed", "source_trace_hash": "d" * 64},
    ]

    benchmark = generate_failure_classifier_benchmark(dataset.manifest, samples)
    report = build_calibration_benchmark_report(benchmark, samples)

    assert benchmark["benchmark_id"] == "bench.calib.failure_classifier.v1"
    assert benchmark["row_count"] == 4
    assert benchmark["tasks"][0]["expected_failure_type"] == "overconfident_failure"
    assert len(benchmark["benchmark_hash"]) == 64
    assert set(report["metrics"]) == {
        "ece",
        "adaptive_ece",
        "high_confidence_failure_rate",
        "trap_verifier_gap",
        "escalation_precision",
        "escalation_recall",
        "confidence_faithfulness",
    }
    assert report["metrics"]["high_confidence_failure_rate"] == 0.5
    assert report["metrics"]["escalation_recall"] == 0.5
    assert report["blind_spot_leads"][0]["cell_id"] == "cell-a"
    assert len(report["benchmark_report_hash"]) == 64


def test_mds_selection_certificate_and_model_comparison_report_are_deterministic() -> None:
    pool = [
        {"task_id": "a", "intrinsic_confidence_score": 0.9, "mds_row_hash": "a" * 64},
        {"task_id": "b", "intrinsic_confidence_score": 0.4, "mds_row_hash": "b" * 64},
    ]
    certificate = build_mds_selection_certificate(pool, min_intrinsic_confidence_score=0.5)
    comparison = build_model_comparison_report(
        {"metrics": {"adaptive_ece": 0.4, "confidence_faithfulness": 0.5}},
        {"metrics": {"adaptive_ece": 0.3, "confidence_faithfulness": 0.7}},
        {"metrics": {"adaptive_ece": 0.2, "confidence_faithfulness": 0.8}},
    )

    assert certificate["pool_size"] == 2
    assert certificate["selected_hashes"] == ["a" * 64]
    assert certificate["rejected_hashes"] == ["b" * 64]
    assert len(certificate["selection_certificate_hash"]) == 64
    assert comparison["winner"] == "rlmf"
    assert comparison["deltas"]["adaptive_ece_improvement"] == 0.2
    assert comparison["deltas"]["confidence_faithfulness_improvement"] == 0.3
    assert len(comparison["model_comparison_hash"]) == 64


def test_g16_real_benchmark_gate_and_promotion_record_emit_rollback_metadata() -> None:
    report = {
        "passed": True,
        "benchmark_report_hash": "r" * 64,
        "metrics": {
            "adaptive_ece": 0.2,
            "high_confidence_failure_rate": 0.1,
            "confidence_faithfulness": 0.8,
            "escalation_recall": 0.75,
        },
    }
    gate = enforce_g16_promotion_gate(report)
    promoted = emit_model_promotion_record({"model_artifact_hash": "m" * 64}, gate, previous_model_hash="p" * 64)

    assert gate["source"] == "real_benchmark_report"
    assert gate["promoted"] is True
    assert promoted["status"] == "promoted"
    assert promoted["rollback_metadata"]["previous_model_hash"] == "p" * 64
    assert promoted["rollback_metadata"]["rollback_required"] is False

    blocked_gate = enforce_g16_promotion_gate({**report, "metrics": {**report["metrics"], "adaptive_ece": 0.5}})
    rejected = emit_model_promotion_record({"model_artifact_hash": "m" * 64}, blocked_gate)

    assert blocked_gate["promoted"] is False
    assert "adaptive_ece" in blocked_gate["reasons"]
    assert rejected["status"] == "rejected"
    assert rejected["rollback_metadata"]["rollback_required"] is True
    assert len(rejected["promotion_record_hash"]) == 64


def test_qwen_judge_benchmark_and_promotion_fixtures(tmp_path: Path) -> None:
    dataset = build_rlmf_fixture_dataset(tmp_path / "rlmf", count=8)
    judge = parse_qwen_judge_fixture({"calibration": 0.82, "confidence_faithfulness": 0.83})
    assert judge["provider"] == "qwen-vllm"
    assert judge["structured_output_valid"] is True
    assert len(judge["judge_report_hash"]) == 64

    benchmark = build_benchmark_fixture(dataset.manifest, judge)
    assert benchmark["benchmark_id"] == "bench.calib.failure_classifier.v1.fixture"
    assert benchmark["passed"] is True
    assert len(benchmark["benchmark_report_hash"]) == 64

    promotion = build_promotion_fixture(benchmark)
    assert promotion["promotion_gate"] == "G16"
    assert promotion["promoted"] is True


def test_qwen_judge_structured_output_accepts_required_score_fields() -> None:
    judge = parse_qwen_judge_fixture(
        {
            "judge_model": "Qwen/Qwen2.5-72B-Instruct",
            "provider": "vllm-openai-compatible",
            "verdict": "review",
            "correctness": "1.0",
            "calibration": "0.0",
            "confidence_faithfulness": "0.5",
            "escalation_recall": "0.25",
        }
    )

    assert judge["ok"] is True
    assert judge["status"] == "completed"
    assert judge["schema_version"] == 1
    assert judge["judge_model"] == "Qwen/Qwen2.5-72B-Instruct"
    assert judge["provider"] == "vllm-openai-compatible"
    assert judge["verdict"] == "review"
    assert judge["structured_output_valid"] is True
    assert set(judge["scores"]) == {
        "correctness",
        "calibration",
        "confidence_faithfulness",
        "escalation_recall",
    }
    assert judge["scores"] == {
        "correctness": 1.0,
        "calibration": 0.0,
        "confidence_faithfulness": 0.5,
        "escalation_recall": 0.25,
    }
    assert len(judge["judge_report_hash"]) == 64


def test_qwen_judge_structured_output_normalizes_missing_scores_to_defaults() -> None:
    judge = parse_qwen_judge_fixture({"verdict": "review"})
    hash_payload = {key: value for key, value in judge.items() if key != "judge_report_hash"}

    assert judge["judge_model"] == "Qwen/Qwen2.5-32B-Instruct"
    assert judge["provider"] == "qwen-vllm"
    assert judge["verdict"] == "review"
    assert judge["scores"] == {
        "correctness": 0.82,
        "calibration": 0.76,
        "confidence_faithfulness": 0.8,
        "escalation_recall": 0.75,
    }
    assert judge["judge_report_hash"] == sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correctness", -0.01),
        ("calibration", 1.01),
        ("confidence_faithfulness", "not-a-number"),
        ("escalation_recall", object()),
    ],
)
def test_qwen_judge_structured_output_rejects_malformed_or_out_of_bounds_scores(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_qwen_judge_fixture({field: value})


def test_prepare_rlmf_fixture_writes_all_artifacts(tmp_path: Path) -> None:
    prepared = prepare_rlmf_fixture(tmp_path / "rlmf", count=8)
    assert prepared["ok"] is True
    assert prepared["dataset"]["manifest"]["dataset_manifest_hash"]
    assert Path(prepared["mlx_lora"]["manifest_path"]).exists()
    assert Path(prepared["qlora"]["manifest_path"]).exists()
    assert prepared["benchmark"]["passed"] is True
    assert prepared["promotion"]["promoted"] is True


def test_rlmf_api_endpoints_are_protected_and_return_fixture_contracts(tmp_path: Path) -> None:
    client = TestClient(create_app(config(tmp_path)))
    unauthorized = client.post("/rlmf/prepare", json={"payload": {"output": str(tmp_path / "rlmf")}})
    assert unauthorized.status_code == 401

    headers = {"Authorization": "Bearer secret"}
    prepare = client.post("/rlmf/prepare", headers=headers, json={"payload": {"output": str(tmp_path / "rlmf"), "count": 8}})
    assert prepare.status_code == 200
    prepared = prepare.json()
    dataset_hash = prepared["dataset"]["manifest"]["dataset_manifest_hash"]
    assert dataset_hash
    assert Path(prepared["mlx_lora"]["manifest_path"]).exists()
    assert Path(prepared["qlora"]["manifest_path"]).exists()

    dry_run = client.post("/rlmf/train_dry_run", headers=headers, json={"payload": {"output": str(tmp_path / "rlmf"), "experts": ["failure_classifier"], "iters": 1}})
    assert dry_run.status_code == 200
    dry_run_body = dry_run.json()
    assert dry_run_body["ok"] is True
    assert dry_run_body["status"] == "planned"
    assert dry_run_body["dry_run"] is True
    assert dry_run_body["dataset_manifest_hash"] == dataset_hash
    assert dry_run_body["mlx_lora"]["status"] == "planned"
    assert dry_run_body["mlx_lora"]["jobs"][0]["expert"] == "failure_classifier"
    assert Path(dry_run_body["mlx_lora"]["manifest_path"]).exists()
    assert dry_run_body["qlora"]["status"] == "planned"
    assert dry_run_body["qlora"]["trainer"] == "hf-trl-qlora"
    assert Path(dry_run_body["qlora"]["manifest_path"]).exists()

    judge = client.post("/rlmf/judge_fixture", headers=headers, json={"payload": {"calibration": 0.8}})
    assert judge.status_code == 200
    assert len(judge.json()["judge_report_hash"]) == 64

    benchmark = client.post("/rlmf/benchmark_fixture", headers=headers, json={"payload": {"output": str(tmp_path / "rlmf")}})
    assert benchmark.status_code == 200
    assert benchmark.json()["benchmark_id"] == "bench.calib.failure_classifier.v1.fixture"

    promotion = client.post("/rlmf/promotion_fixture", headers=headers, json={"payload": {"output": str(tmp_path / "rlmf")}})
    assert promotion.status_code == 200
    assert promotion.json()["promotion_gate"] == "G16"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]
