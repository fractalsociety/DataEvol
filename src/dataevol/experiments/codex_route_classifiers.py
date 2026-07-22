from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPORT_SCHEMA = "dataevol.codex_route_classifier_report.v1"
EMBEDDING_SCHEMA = "dataevol.codex_frozen_embedding_cache.v1"
FAMILY_BY_MODEL = {
    "gpt-5.3-codex-spark": "mini",
    "gpt-5.4-mini": "mini",
    "gpt-5.4": "standard",
    "gpt-5.5": "standard",
    "gpt-5.6-sol": "frontier",
}
EFFORT_LEVEL = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
SELECTIVE_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97)
SELECTIVE_DECISION_SCHEMA = "dataevol.codex_selective_classifier_decision.v1"


def run_classifier_benchmark(
    train_rows: Iterable[Mapping[str, Any]],
    eval_rows: Iterable[Mapping[str, Any]],
    *,
    embedding_train: np.ndarray | None = None,
    embedding_eval: np.ndarray | None = None,
    seed: int = 1701,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = [dict(row) for row in train_rows]
    evaluate = [dict(row) for row in eval_rows]
    _validate_rows(train, "train")
    _validate_rows(evaluate, "eval")
    train_labels = [_exact_label(row) for row in train]
    eval_labels = [_exact_label(row) for row in evaluate]
    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform([_tabular_features(row) for row in train])
    x_eval = vectorizer.transform([_tabular_features(row) for row in evaluate])
    estimators: dict[str, Any] = {
        "tabular_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=seed)),
        "tabular_random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "tabular_hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=0.1, random_state=seed),
    }
    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {"tabular_vectorizer": vectorizer}
    rule_predictions = [_rule_prediction(row) for row in evaluate]
    results["deterministic_rule"] = _evaluate_predictions(evaluate, rule_predictions)
    for name, estimator in estimators.items():
        estimator.fit(x_train, train_labels)
        predictions = _classifier_predictions(estimator, x_eval, evaluate)
        results[name] = _evaluate_predictions(evaluate, predictions)
        fitted[name] = estimator

    tabular_hierarchy = _fit_hierarchy(x_train, train, seed)
    results["tabular_hierarchical"] = _evaluate_predictions(evaluate, _hierarchy_predictions(tabular_hierarchy, x_eval, evaluate))
    fitted["tabular_hierarchical"] = tabular_hierarchy

    if (embedding_train is None) != (embedding_eval is None):
        raise ValueError("embedding train and eval arrays must be provided together")
    if embedding_train is not None and embedding_eval is not None:
        if embedding_train.shape[0] != len(train) or embedding_eval.shape[0] != len(evaluate):
            raise ValueError("embedding row count does not match classifier rows")
        embedding_hierarchy = _fit_hierarchy(embedding_train, train, seed, scale=True)
        results["frozen_embedding_hierarchical"] = _evaluate_predictions(
            evaluate,
            _hierarchy_predictions(embedding_hierarchy, embedding_eval, evaluate),
        )
        fitted["frozen_embedding_hierarchical"] = embedding_hierarchy

    report_body = {
        "schema": REPORT_SCHEMA,
        "seed": seed,
        "train_rows": len(train),
        "eval_rows": len(evaluate),
        "eval_task_groups": len({str(row["task_group"]) for row in evaluate}),
        "models": results,
    }
    report = {**report_body, "report_hash": _hash_object(report_body)}
    return report, fitted


def extract_frozen_embeddings(
    rows: Iterable[Mapping[str, Any]],
    *,
    model_path: str,
    output_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    import mlx.core as mx
    from mlx_lm import load

    from dataevol.local_models.layer_specialist import model_fingerprint

    values = [dict(row) for row in rows]
    _validate_rows(values, "embedding")
    destination = Path(output_path)
    sidecar = destination.with_suffix(destination.suffix + ".manifest.json")
    fingerprint = model_fingerprint(model_path)
    identities = [
        {"id": row["id"], "text_sha256": _sha256_text(_embedding_text(row))}
        for row in values
    ]
    config = {
        "schema": EMBEDDING_SCHEMA,
        "model_fingerprint": fingerprint["sha256"],
        "pooling": "last_token_plus_mean",
        "rows": identities,
    }
    config_hash = _hash_object(config)
    if destination.exists() or sidecar.exists():
        if not destination.is_file() or not sidecar.is_file():
            raise ValueError("embedding cache is incomplete")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config_hash or manifest.get("output_sha256") != _sha256_path(destination):
            raise ValueError("embedding cache identity or integrity mismatch")
        body = np.load(destination, allow_pickle=False)
        return np.asarray(body["embeddings"], dtype=np.float32), manifest
    model, tokenizer = load(model_path)
    model.freeze()
    embeddings = []
    started = time.time()
    for index, row in enumerate(values, start=1):
        token_ids = list(tokenizer.encode(_embedding_text(row)))
        if not token_ids:
            raise ValueError(f"embedding row has no tokens: {row['id']}")
        hidden = model.model(mx.array(token_ids)[None, :])
        pooled = mx.concatenate([hidden[0, -1], hidden[0].mean(axis=0)], axis=0)
        mx.eval(pooled)
        embeddings.append(np.asarray(pooled, dtype=np.float32))
        if index == 1 or index % 100 == 0:
            print(json.dumps({"embedded": index, "total": len(values)}), flush=True)
    matrix = np.stack(embeddings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, embeddings=matrix)
    manifest = {
        "schema": EMBEDDING_SCHEMA,
        "config": config,
        "config_hash": config_hash,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "elapsed_seconds": time.time() - started,
        "output_sha256": _sha256_path(destination),
    }
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return matrix, manifest


def selective_tabular_decision(
    row: Mapping[str, Any],
    fitted: Mapping[str, Any],
    *,
    model_name: str = "tabular_logistic",
    confidence_threshold: float = 0.97,
    require_trusted_boundary: bool = True,
) -> dict[str, Any]:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    item = dict(row)
    _validate_rows([item], "selective")
    vectorizer = fitted.get("tabular_vectorizer")
    estimator = fitted.get(model_name)
    if vectorizer is None or estimator is None:
        raise ValueError(f"classifier artifact does not contain {model_name}")
    matrix = vectorizer.transform([_tabular_features(item)])
    if model_name == "tabular_hierarchical":
        prediction = _hierarchy_predictions(estimator, matrix, [item])[0]
    else:
        prediction = _classifier_predictions(estimator, matrix, [item])[0]
    features = item["features"]
    low_medium = features["risk"] in {"low", "medium"}
    boundary_trusted = set(features["boundary_states"]) == {"trusted"}
    reasons = []
    if not low_medium:
        reasons.append("risk-requires-teacher")
    if not prediction["valid"]:
        reasons.append("invalid-model-effort-combination")
    if prediction["confidence"] < confidence_threshold:
        reasons.append("confidence-below-threshold")
    if require_trusted_boundary and not boundary_trusted:
        reasons.append("capability-boundary-not-trusted")
    accepted = not reasons
    selected = prediction.get("option_id")
    remaining = [str(option["option_id"]) for option in item["eligible_options"] if option["option_id"] != selected]
    body = {
        "schema": SELECTIVE_DECISION_SCHEMA,
        "row_id": item["id"],
        "task_group": item["task_group"],
        "subtask_id": item["subtask_id"],
        "classifier": model_name,
        "confidence_threshold": confidence_threshold,
        "confidence": prediction["confidence"],
        "predicted_option_id": selected,
        "ranked_option_ids": ([selected] if selected else []) + remaining,
        "accepted": accepted,
        "authority": "classifier" if accepted else "teacher",
        "reasons": reasons,
        "catalog_hash": item["catalog_hash"],
        "policy_hash": item["policy_hash"],
        "candidate_set_hash": item["candidate_set_hash"],
    }
    return {**body, "decision_hash": _hash_object(body)}


def _fit_hierarchy(x_train: np.ndarray, rows: list[dict[str, Any]], seed: int, *, scale: bool = False) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    family_labels = np.asarray([_family_label(row) for row in rows])
    family_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=seed)) if scale else LogisticRegression(max_iter=2_000, random_state=seed)
    family_model.fit(x_train, family_labels)
    effort_models = {}
    for family in sorted(set(family_labels)):
        indices = np.where(family_labels == family)[0]
        labels = np.asarray([_chosen_option(rows[index])["reasoning_effort"] for index in indices])
        unique = sorted(set(labels))
        if len(unique) == 1:
            effort_models[family] = {"constant": unique[0]}
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=seed)) if scale else LogisticRegression(max_iter=2_000, random_state=seed)
            model.fit(x_train[indices], labels)
            effort_models[family] = {"model": model}
    return {"family_model": family_model, "effort_models": effort_models}


def _hierarchy_predictions(hierarchy: Mapping[str, Any], matrix: np.ndarray, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_model = hierarchy["family_model"]
    family_probabilities = family_model.predict_proba(matrix)
    family_classes = list(family_model.classes_)
    predictions = []
    for index, row in enumerate(rows):
        family_position = int(np.argmax(family_probabilities[index]))
        family = str(family_classes[family_position])
        family_confidence = float(family_probabilities[index, family_position])
        effort_head = hierarchy["effort_models"].get(family)
        if not effort_head:
            effort, effort_confidence = "medium", 0.0
        elif "constant" in effort_head:
            effort, effort_confidence = str(effort_head["constant"]), 1.0
        else:
            model = effort_head["model"]
            probabilities = model.predict_proba(matrix[index:index + 1])[0]
            position = int(np.argmax(probabilities))
            effort, effort_confidence = str(model.classes_[position]), float(probabilities[position])
        option = _resolve_family_effort(row, family, effort)
        predictions.append(
            {
                "option_id": option.get("option_id") if option else None,
                "model_id": option.get("model_id") if option else None,
                "reasoning_effort": option.get("reasoning_effort") if option else effort,
                "family": family,
                "confidence": family_confidence * effort_confidence,
                "valid": option is not None,
            }
        )
    return predictions


def _classifier_predictions(estimator: Any, matrix: np.ndarray, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probabilities = estimator.predict_proba(matrix)
    classes = list(estimator.classes_)
    predictions = []
    for index, row in enumerate(rows):
        position = int(np.argmax(probabilities[index]))
        model_id, effort = str(classes[position]).split("|", 1)
        option = _resolve_model_effort(row, model_id, effort)
        predictions.append(
            {
                "option_id": option.get("option_id") if option else None,
                "model_id": model_id,
                "reasoning_effort": effort,
                "family": FAMILY_BY_MODEL.get(model_id, "unknown"),
                "confidence": float(probabilities[index, position]),
                "valid": option is not None,
            }
        )
    return predictions


def _rule_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    features = row["features"]
    risk = str(features["risk"])
    capabilities = set(features["required_capabilities"])
    if risk in {"high", "critical"}:
        model_id, effort = "gpt-5.6-sol", "xhigh" if risk == "critical" else "high"
    elif risk == "medium" or capabilities & {"architecture", "integration", "migration", "research", "security"}:
        model_id, effort = "gpt-5.4", "medium"
    else:
        model_id = "gpt-5.4-mini"
        effort = "medium" if capabilities & {"code", "tests", "verification"} else "low"
    option = _resolve_model_effort(row, model_id, effort)
    return {
        "option_id": option.get("option_id") if option else None,
        "model_id": model_id,
        "reasoning_effort": effort,
        "family": FAMILY_BY_MODEL[model_id],
        "confidence": 1.0,
        "valid": option is not None,
    }


def _evaluate_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("prediction count mismatch")
    outcomes = []
    for row, prediction in zip(rows, predictions):
        chosen = _chosen_option(row)
        correct = prediction.get("option_id") == chosen["option_id"]
        outcomes.append(
            {
                "id": row["id"],
                "task_group": row["task_group"],
                "risk": row["features"]["risk"],
                "correct": correct,
                "valid": bool(prediction.get("valid")),
                "confidence": float(prediction["confidence"]),
                "expected_option_id": chosen["option_id"],
                "predicted_option_id": prediction.get("option_id"),
                "model_correct": prediction.get("model_id") == chosen["model_id"],
                "effort_correct": prediction.get("reasoning_effort") == chosen["reasoning_effort"],
            }
        )
    selective = {}
    low_medium = [item for item in outcomes if item["risk"] in {"low", "medium"}]
    for threshold in SELECTIVE_THRESHOLDS:
        accepted = [item for item in low_medium if item["valid"] and item["confidence"] >= threshold]
        selective[f"{threshold:.2f}"] = {
            "accepted": len(accepted),
            "eligible_low_medium": len(low_medium),
            "coverage": len(accepted) / len(low_medium) if low_medium else 0.0,
            "precision": statistics.fmean(float(item["correct"]) for item in accepted) if accepted else None,
            "model_precision": statistics.fmean(float(item["model_correct"]) for item in accepted) if accepted else None,
        }
    count = len(outcomes)
    return {
        "metrics": {
            "option_accuracy": statistics.fmean(float(item["correct"]) for item in outcomes),
            "model_accuracy": statistics.fmean(float(item["model_correct"]) for item in outcomes),
            "reasoning_effort_accuracy": statistics.fmean(float(item["effort_correct"]) for item in outcomes),
            "valid_combination_rate": statistics.fmean(float(item["valid"]) for item in outcomes),
            "brier_score": statistics.fmean((item["confidence"] - float(item["correct"])) ** 2 for item in outcomes),
            "rows": count,
        },
        "selective_low_medium": selective,
        "predictions": outcomes,
    }


def _tabular_features(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row["features"]
    result: dict[str, Any] = {
        "risk": str(raw["risk"]),
        "verification_floor": str(raw["verification_floor"]),
        "estimated_input_tokens_log": math.log1p(float(raw["estimated_input_tokens"])),
        "dependency_count": float(raw["dependency_count"]),
        "subtask_count": float(raw["subtask_count"]),
        "eligible_option_count": float(raw["eligible_option_count"]),
    }
    for capability in raw["required_capabilities"]:
        result[f"capability={capability}"] = 1.0
    for state in raw["boundary_states"]:
        result[f"boundary={state}"] = 1.0
    for signal, present in raw["signals"].items():
        result[f"signal={signal}"] = float(bool(present))
    return result


def _embedding_text(row: Mapping[str, Any]) -> str:
    features = row["features"]
    return (
        f"Task: {features['task_text']}\nSubtask: {features['objective']}\n"
        f"Risk: {features['risk']}\nCapabilities: {', '.join(features['required_capabilities']) or 'general'}\n"
        f"Context tokens: {features['estimated_input_tokens']}\nVerification: {features['verification_floor']}"
    )


def _exact_label(row: Mapping[str, Any]) -> str:
    option = _chosen_option(row)
    return f"{option['model_id']}|{option['reasoning_effort']}"


def _family_label(row: Mapping[str, Any]) -> str:
    return FAMILY_BY_MODEL[_chosen_option(row)["model_id"]]


def _chosen_option(row: Mapping[str, Any]) -> dict[str, Any]:
    chosen_id = str(row["chosen_option_id"])
    option = next((dict(value) for value in row["eligible_options"] if value["option_id"] == chosen_id), None)
    if option is None:
        raise ValueError(f"chosen option is absent from eligible options: {row.get('id')}")
    return option


def _resolve_model_effort(row: Mapping[str, Any], model_id: str, effort: str) -> dict[str, Any] | None:
    return next((dict(option) for option in row["eligible_options"] if option["model_id"] == model_id and option["reasoning_effort"] == effort), None)


def _resolve_family_effort(row: Mapping[str, Any], family: str, effort: str) -> dict[str, Any] | None:
    return next((dict(option) for option in row["eligible_options"] if FAMILY_BY_MODEL.get(option["model_id"]) == family and option["reasoning_effort"] == effort), None)


def _validate_rows(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} classifier rows are empty")
    seen = set()
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in seen:
            raise ValueError(f"{label} classifier rows require unique IDs")
        seen.add(row_id)
        if not isinstance(row.get("features"), Mapping) or not isinstance(row.get("eligible_options"), list):
            raise ValueError(f"classifier row lacks structured features or options: {row_id}")
        _chosen_option(row)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_object(value: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False))


def main() -> None:
    import joblib
    import sklearn

    parser = argparse.ArgumentParser(description="Train frozen-feature Codex route classifiers")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    train = _read_jsonl(args.train)
    evaluate = _read_jsonl(args.eval)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"classifier output already exists: {output}")
    output.mkdir(parents=True)
    embedding_train = embedding_eval = None
    embedding_manifests = None
    if args.model:
        embedding_train, train_manifest = extract_frozen_embeddings(train, model_path=args.model, output_path=output / "train_embeddings.npz")
        embedding_eval, eval_manifest = extract_frozen_embeddings(evaluate, model_path=args.model, output_path=output / "eval_embeddings.npz")
        embedding_manifests = {"train": train_manifest, "eval": eval_manifest}
    report, fitted = run_classifier_benchmark(train, evaluate, embedding_train=embedding_train, embedding_eval=embedding_eval, seed=args.seed)
    model_path = output / "classifiers.joblib"
    joblib.dump(fitted, model_path)
    report = {
        **report,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "inputs": {"train_sha256": _sha256_path(args.train), "eval_sha256": _sha256_path(args.eval)},
        "embedding_manifests": embedding_manifests,
        "model_artifact": {"path": str(model_path), "sha256": _sha256_path(model_path)},
    }
    report["report_hash"] = _hash_object({key: value for key, value in report.items() if key != "report_hash"})
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / 'report.json'), "models": {name: body["metrics"] for name, body in report["models"].items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
