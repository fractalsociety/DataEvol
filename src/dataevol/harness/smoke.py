"""S1.10 (+S1.5) harness smoke suite: 20 deterministic cases spanning
arithmetic, Python repair, and JSON extraction, with public training checks
kept separate from hidden verifiers.

Each case in ``tests/fixtures/harness_smoke/cases.jsonl`` is a case dict with
an ``id`` and ``category`` (consistent with how :mod:`executor` reads
benchmarks) plus a ``family`` and a ``public_checks`` spec. The matching
hidden verifier data lives in ``tests/fixtures/harness_smoke/hidden/<id>.json``
and must never be handed to a worker: :func:`public_check` only ever reads
``public_checks``, while :func:`hidden_verify` reads the hidden file. This
module is stdlib-only, like the rest of the S1 execution contract.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .execution_contract import canonical_json

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
DEFAULT_CASES_PATH = _REPO_ROOT / "tests" / "fixtures" / "harness_smoke" / "cases.jsonl"
DEFAULT_HIDDEN_DIR = _REPO_ROOT / "tests" / "fixtures" / "harness_smoke" / "hidden"

FAMILIES = frozenset({"arithmetic", "python_repair", "json_extraction"})
EXPECTED_FAMILY_COUNTS = {"arithmetic": 7, "python_repair": 7, "json_extraction": 6}
TOTAL_CASES = 20

_REQUIRED_FIELDS = ("id", "family", "category", "prompt", "public_checks", "hidden_ref")
_PY_TIMEOUT_SECONDS = 5.0
_TYPE_CHECKERS: dict[str, Any] = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
}


class SmokeSuiteError(ValueError):
    """Raised when the smoke fixture set fails validation."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SmokeSuiteError(f"smoke cases file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SmokeSuiteError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise SmokeSuiteError(f"{path}:{line_number} must be a JSON object")
            rows.append(parsed)
    return rows


def _secret_markers(family: str, hidden_payload: Mapping[str, Any]) -> list[str]:
    """Extract the hidden answer content that must never leak into a case."""
    if family == "arithmetic":
        if "expected_answer" not in hidden_payload:
            raise SmokeSuiteError("arithmetic hidden payload requires expected_answer")
        return [str(hidden_payload["expected_answer"])]
    if family == "python_repair":
        tests = hidden_payload.get("hidden_tests")
        if not isinstance(tests, list) or not tests:
            raise SmokeSuiteError("python_repair hidden payload requires a non-empty hidden_tests list")
        return [str(t) for t in tests]
    if family == "json_extraction":
        if "expected_records" not in hidden_payload:
            raise SmokeSuiteError("json_extraction hidden payload requires expected_records")
        return [canonical_json(hidden_payload["expected_records"])]
    raise SmokeSuiteError(f"unknown family: {family}")


def load_smoke_cases(
    path: Path | str = DEFAULT_CASES_PATH,
    *,
    hidden_dir: Path | str = DEFAULT_HIDDEN_DIR,
) -> list[dict[str, Any]]:
    """Load and validate the smoke suite.

    Validates: exactly 20 cases; the required family distribution (7
    arithmetic, 7 python_repair, 6 json_extraction); unique ids; all
    required fields present; and that no hidden verifier content leaks into
    the public case (prompt + public_checks).
    """
    cases_path = Path(path)
    hidden_path = Path(hidden_dir)
    rows = _read_jsonl(cases_path)

    if len(rows) != TOTAL_CASES:
        raise SmokeSuiteError(f"expected exactly {TOTAL_CASES} cases, found {len(rows)}")

    seen_ids: set[str] = set()
    family_counts: dict[str, int] = {}
    for row in rows:
        missing = [field for field in _REQUIRED_FIELDS if field not in row]
        if missing:
            raise SmokeSuiteError(f"case {row.get('id')!r} is missing required field(s): {missing}")

        case_id = row["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise SmokeSuiteError("case id must be a non-empty string")
        if case_id in seen_ids:
            raise SmokeSuiteError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)

        family = row["family"]
        if family not in FAMILIES:
            raise SmokeSuiteError(f"case {case_id!r} has unknown family {family!r}")
        family_counts[family] = family_counts.get(family, 0) + 1

        category = row["category"]
        if not isinstance(category, str) or not category.strip():
            raise SmokeSuiteError(f"case {case_id!r} has an invalid category")

        prompt = row["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise SmokeSuiteError(f"case {case_id!r} has an empty prompt")

        public_checks = row["public_checks"]
        if not isinstance(public_checks, Mapping):
            raise SmokeSuiteError(f"case {case_id!r} public_checks must be an object")

        hidden_ref = row["hidden_ref"]
        if not isinstance(hidden_ref, str) or not hidden_ref.strip():
            raise SmokeSuiteError(f"case {case_id!r} has an invalid hidden_ref")

        hidden_file = hidden_path / hidden_ref
        if not hidden_file.exists():
            raise SmokeSuiteError(f"case {case_id!r} hidden file not found: {hidden_file}")
        try:
            hidden_payload = json.loads(hidden_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SmokeSuiteError(f"case {case_id!r} hidden file is not valid JSON") from exc
        if not isinstance(hidden_payload, Mapping):
            raise SmokeSuiteError(f"case {case_id!r} hidden file must contain a JSON object")

        haystack = prompt + canonical_json(public_checks)
        for marker in _secret_markers(family, hidden_payload):
            if marker and marker in haystack:
                raise SmokeSuiteError(
                    f"case {case_id!r} leaks hidden verifier content into its prompt/public_checks"
                )

    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise SmokeSuiteError(
            f"family distribution must be {EXPECTED_FAMILY_COUNTS}, found {family_counts}"
        )

    return rows


def _load_hidden(case: Mapping[str, Any], hidden_dir: Path | str) -> dict[str, Any]:
    hidden_ref = case.get("hidden_ref")
    if not isinstance(hidden_ref, str) or not hidden_ref.strip():
        raise SmokeSuiteError("case has no hidden_ref")
    hidden_file = Path(hidden_dir) / hidden_ref
    if not hidden_file.exists():
        raise SmokeSuiteError(f"hidden file not found: {hidden_file}")
    payload = json.loads(hidden_file.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SmokeSuiteError(f"hidden file must contain a JSON object: {hidden_file}")
    return dict(payload)


def _run_python(code: str, *, timeout: float = _PY_TIMEOUT_SECONDS) -> bool:
    """Execute ``code`` in a fresh subprocess interpreter and report success.

    Using ``sys.executable -c`` keeps the candidate/tests out of our own
    process namespace and gives a hard wall-clock timeout; a non-zero exit
    (assertion failure, exception, or timeout) is treated as a failing run.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _public_check_regex(case: Mapping[str, Any], output: str) -> bool:
    import re

    pattern = case["public_checks"].get("pattern")
    if not isinstance(pattern, str):
        raise SmokeSuiteError("regex public_checks requires a string pattern")
    return re.fullmatch(pattern, output.strip()) is not None


def _public_check_python_test(case: Mapping[str, Any], output: str) -> bool:
    test_code = case["public_checks"].get("test_code")
    if not isinstance(test_code, str) or not test_code.strip():
        raise SmokeSuiteError("python_public_test public_checks requires test_code")
    return _run_python(output + "\n" + test_code)


def _parse_json_records(output: str) -> list[Any] | None:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _public_check_json_schema(case: Mapping[str, Any], output: str) -> bool:
    spec = case["public_checks"]
    required_keys = spec.get("required_keys")
    sort_key = spec.get("sort_key")
    if not isinstance(required_keys, Mapping) or not required_keys:
        raise SmokeSuiteError("json_records_schema public_checks requires required_keys")
    if not isinstance(sort_key, str) or sort_key not in required_keys:
        raise SmokeSuiteError("json_records_schema public_checks requires sort_key in required_keys")

    records = _parse_json_records(output)
    if records is None:
        return False

    for record in records:
        if not isinstance(record, Mapping):
            return False
        if set(record.keys()) != set(required_keys.keys()):
            return False
        for key, type_name in required_keys.items():
            checker = _TYPE_CHECKERS.get(type_name)
            if checker is None:
                raise SmokeSuiteError(f"unknown required_keys type: {type_name}")
            if not checker(record[key]):
                return False

    sort_values = [record[sort_key] for record in records]
    return sort_values == sorted(sort_values)


_PUBLIC_CHECKERS = {
    "regex": _public_check_regex,
    "python_public_test": _public_check_python_test,
    "json_records_schema": _public_check_json_schema,
}


def public_check(case: Mapping[str, Any], output: str) -> bool:
    """Evaluate only ``case["public_checks"]`` against a worker's output.

    Never touches the hidden verifier file, so a worker process can be
    handed the case (including public_checks) without exposing the answer.
    """
    public_checks = case.get("public_checks")
    if not isinstance(public_checks, Mapping):
        raise SmokeSuiteError("case public_checks must be an object")
    check_type = public_checks.get("type")
    checker = _PUBLIC_CHECKERS.get(check_type)
    if checker is None:
        raise SmokeSuiteError(f"unknown public_checks type: {check_type}")
    return checker(case, output)


def _hidden_verify_arithmetic(case: Mapping[str, Any], output: str, hidden: Mapping[str, Any]) -> bool:
    expected = hidden.get("expected_answer")
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        raise SmokeSuiteError("arithmetic hidden payload requires a numeric expected_answer")
    try:
        actual = int(output.strip())
    except ValueError:
        return False
    return actual == expected


def _hidden_verify_python_repair(case: Mapping[str, Any], output: str, hidden: Mapping[str, Any]) -> bool:
    hidden_tests = hidden.get("hidden_tests")
    if not isinstance(hidden_tests, list) or not hidden_tests:
        raise SmokeSuiteError("python_repair hidden payload requires a non-empty hidden_tests list")
    public_test = case["public_checks"].get("test_code", "")
    script = output + "\n" + public_test + "\n" + "\n".join(str(t) for t in hidden_tests)
    return _run_python(script)


def _hidden_verify_json_extraction(case: Mapping[str, Any], output: str, hidden: Mapping[str, Any]) -> bool:
    expected_records = hidden.get("expected_records")
    if not isinstance(expected_records, list):
        raise SmokeSuiteError("json_extraction hidden payload requires expected_records")
    records = _parse_json_records(output)
    if records is None:
        return False
    return records == expected_records


_HIDDEN_VERIFIERS = {
    "arithmetic": _hidden_verify_arithmetic,
    "python_repair": _hidden_verify_python_repair,
    "json_extraction": _hidden_verify_json_extraction,
}


def hidden_verify(
    case: Mapping[str, Any],
    output: str,
    hidden_dir: Path | str = DEFAULT_HIDDEN_DIR,
) -> bool:
    """Fully verify ``output`` against the case's hidden verifier data.

    Loads ``hidden_dir / case["hidden_ref"]`` (never present in the case
    itself) and performs the family-specific check: exact numeric match for
    arithmetic, execution of public + hidden tests for python_repair, and an
    exact complete-records match for json_extraction.
    """
    family = case.get("family")
    verifier = _HIDDEN_VERIFIERS.get(family)
    if verifier is None:
        raise SmokeSuiteError(f"unknown family: {family}")
    hidden = _load_hidden(case, hidden_dir)
    return verifier(case, output, hidden)


__all__ = [
    "DEFAULT_CASES_PATH",
    "DEFAULT_HIDDEN_DIR",
    "EXPECTED_FAMILY_COUNTS",
    "FAMILIES",
    "TOTAL_CASES",
    "SmokeSuiteError",
    "hidden_verify",
    "load_smoke_cases",
    "public_check",
]
