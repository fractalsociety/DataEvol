from __future__ import annotations


def coordinate_run_trace() -> dict:
    return {
        "id": "coord_trace_001",
        "run_id": "coordinate_run_001",
        "source_system": "coordinate",
        "trace_type": "router_trace",
        "task": "Assign worker to low-risk documentation task",
        "task_type": "coding_agent_workflow",
        "decision": {"worker": "worker-docs", "provider": "openrouter_free", "model": "mock-small", "reason": "Low risk and verifier-backed."},
        "label": "accepted",
        "score": 0.94,
        "metrics": {"cost_usd": 0.0, "latency_ms": 900, "quality": 0.94},
        "privacy_status": "local_only",
        "why_good": "Correctly used a cheap worker and passed verification.",
    }


def router_biolatent_trace() -> dict:
    return {
        "id": "router_biolatent_001",
        "run_id": "biolatent_run_001",
        "source_system": "biolatent",
        "trace_type": "scientific_trace",
        "task": "Route protocol critique to BioLatent verifier",
        "task_type": "scientific_workflow",
        "decision": {"worker": "biolatent-verifier", "provider": "local", "model": "mock-biolatent-verifier", "reason": "Scientific claims require verifier."},
        "label": "accepted",
        "score": 0.91,
        "metrics": {"cost_usd": 0.01, "latency_ms": 1300, "quality": 0.91},
        "privacy_status": "local_only",
        "why_good": "Routed scientific workflow to verifier before claims could be accepted.",
    }
