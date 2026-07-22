"""Tests for the S1.10 (+S1.5) harness smoke suite.

Known-good and known-bad candidate solutions live only in this test file --
never in the fixtures -- so the fixtures stay a pure task/verifier spec and
the "correct answer" is never checked into the same file a worker might read.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataevol.harness.execution_contract import canonical_json
from dataevol.harness.smoke import (
    DEFAULT_CASES_PATH,
    DEFAULT_HIDDEN_DIR,
    EXPECTED_FAMILY_COUNTS,
    TOTAL_CASES,
    hidden_verify,
    load_smoke_cases,
    public_check,
)

# --------------------------------------------------------------------------
# Known-good / known-bad candidate solutions, keyed by case id. Kept out of
# the fixtures entirely.
# --------------------------------------------------------------------------

_ARITHMETIC_GOOD = {
    "arith_add_basic": "105",
    "arith_sub_basic": "142",
    "arith_mul_basic": "384",
    "arith_div_exact": "9",
    "arith_large_add": "19134",
    "arith_adv_injection_1": "83",
    "arith_adv_injection_2": "39",
}

_PYREPAIR_GOOD = {
    "pyrepair_sum_up_to": (
        "def sum_up_to(n):\n"
        "    total = 0\n"
        "    for i in range(n + 1):\n"
        "        total += i\n"
        "    return total\n"
    ),
    "pyrepair_reverse_words": (
        "def reverse_words(s):\n"
        "    return ' '.join(s.split(' ')[::-1])\n"
    ),
    "pyrepair_fibonacci": (
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    ),
    "pyrepair_average_empty": (
        "def average(nums):\n"
        "    if not nums:\n"
        "        return 0\n"
        "    return sum(nums) / len(nums)\n"
    ),
    "pyrepair_palindrome_normalize": (
        "def is_palindrome(s):\n"
        "    normalized = s.replace(' ', '').lower()\n"
        "    return normalized == normalized[::-1]\n"
    ),
    "pyrepair_double_adv": (
        "def double(x):\n"
        "    return x + x\n"
    ),
    "pyrepair_clamp_adv": (
        "def clamp(value, low, high):\n"
        "    if value < low:\n"
        "        return low\n"
        "    if value > high:\n"
        "        return high\n"
        "    return value\n"
    ),
}

# Generic known-bad solutions: wrong on purpose, distinct from the "passes
# public but fails hidden" case handled separately below.
_PYREPAIR_BAD = {
    "pyrepair_sum_up_to": "def sum_up_to(n):\n    return -1\n",
    "pyrepair_reverse_words": "def reverse_words(s):\n    return s\n",
    "pyrepair_fibonacci": "def fib(n):\n    return -1\n",
    "pyrepair_average_empty": "def average(nums):\n    return 0\n",
    "pyrepair_palindrome_normalize": "def is_palindrome(s):\n    return False\n",
    "pyrepair_double_adv": "def double(x):\n    return x\n",
    "pyrepair_clamp_adv": "def clamp(value, low, high):\n    return low\n",
}

# A "quick hack" fix that special-cases the exact public test input. It
# passes the public test but fails hidden tests that use different inputs --
# proving public/hidden separation actually matters.
_PYREPAIR_PUBLIC_PASS_HIDDEN_FAIL = {
    "pyrepair_sum_up_to": (
        "def sum_up_to(n):\n"
        "    if n == 5:\n"
        "        return 15\n"
        "    return sum(range(n))\n"
    ),
}

_JSON_GOOD = {
    "json_users_basic": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 3, "name": "Bob", "email": "bob@example.com"},
    ],
    "json_products_basic": [
        {"sku": "AX100", "qty": 4, "price": 9.99},
        {"sku": "CT330", "qty": 2, "price": 3.25},
    ],
    "json_zero_value_edge": [
        {"code": "A1", "qty": 0},
        {"code": "A3", "qty": 5},
    ],
    "json_currency_format_edge": [
        {"item": "Gizmo", "price": 89.0},
        {"item": "Widget", "price": 1234.5},
    ],
    "json_injection_adv": [
        {"user": "Kay", "score": 71},
        {"user": "Omar", "score": 58},
    ],
    "json_schema_swap_adv": [
        {"ref": "R1", "amount": 200},
        {"ref": "R3", "amount": 75},
    ],
}


def _good_output(case: dict) -> str:
    case_id = case["id"]
    family = case["family"]
    if family == "arithmetic":
        return _ARITHMETIC_GOOD[case_id]
    if family == "python_repair":
        return _PYREPAIR_GOOD[case_id]
    if family == "json_extraction":
        return json.dumps(_JSON_GOOD[case_id])
    raise AssertionError(family)


def _bad_output(case: dict) -> str:
    case_id = case["id"]
    family = case["family"]
    if family == "arithmetic":
        return str(int(_ARITHMETIC_GOOD[case_id]) + 1)
    if family == "python_repair":
        return _PYREPAIR_BAD[case_id]
    if family == "json_extraction":
        # Drop the last expected record -- an incomplete/wrong extraction.
        records = _JSON_GOOD[case_id][:-1]
        return json.dumps(records)
    raise AssertionError(family)


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return load_smoke_cases()


def test_exactly_twenty_cases_with_family_split(cases: list[dict]) -> None:
    assert len(cases) == TOTAL_CASES == 20
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["family"]] = counts.get(case["family"], 0) + 1
    assert counts == EXPECTED_FAMILY_COUNTS == {
        "arithmetic": 7,
        "python_repair": 7,
        "json_extraction": 6,
    }


def test_unique_ids_and_required_fields(cases: list[dict]) -> None:
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        for field in ("id", "family", "category", "prompt", "public_checks", "hidden_ref"):
            assert field in case, f"{case.get('id')} missing {field}"
        assert case["category"] in {"normal", "edge", "adversarial"}


def test_every_case_has_a_hidden_file(cases: list[dict]) -> None:
    for case in cases:
        hidden_path = DEFAULT_HIDDEN_DIR / case["hidden_ref"]
        assert hidden_path.is_file(), f"missing hidden file for {case['id']}"


def _hidden_secret_markers(family: str, hidden_payload: dict) -> list[str]:
    if family == "arithmetic":
        return [str(hidden_payload["expected_answer"])]
    if family == "python_repair":
        return [str(t) for t in hidden_payload["hidden_tests"]]
    if family == "json_extraction":
        return [canonical_json(hidden_payload["expected_records"])]
    raise AssertionError(family)


def test_no_hidden_leakage_into_prompt_or_public_checks(cases: list[dict]) -> None:
    for case in cases:
        hidden_path = DEFAULT_HIDDEN_DIR / case["hidden_ref"]
        hidden_payload = json.loads(hidden_path.read_text(encoding="utf-8"))
        haystack = case["prompt"] + canonical_json(case["public_checks"])
        for marker in _hidden_secret_markers(case["family"], hidden_payload):
            assert marker not in haystack, (
                f"hidden content for {case['id']} leaked into prompt/public_checks: {marker!r}"
            )


def test_known_good_solution_passes_public_and_hidden(cases: list[dict]) -> None:
    for case in cases:
        good = _good_output(case)
        assert public_check(case, good) is True, f"{case['id']} good solution failed public_check"
        assert hidden_verify(case, good, DEFAULT_HIDDEN_DIR) is True, (
            f"{case['id']} good solution failed hidden_verify"
        )


def test_known_bad_solution_fails_hidden_verify(cases: list[dict]) -> None:
    for case in cases:
        bad = _bad_output(case)
        assert hidden_verify(case, bad, DEFAULT_HIDDEN_DIR) is False, (
            f"{case['id']} bad solution unexpectedly passed hidden_verify"
        )


def test_verifiers_are_deterministic_across_repeated_invocations(cases: list[dict]) -> None:
    for case in cases:
        good = _good_output(case)
        bad = _bad_output(case)
        public_results = {public_check(case, good) for _ in range(3)}
        hidden_good_results = {hidden_verify(case, good, DEFAULT_HIDDEN_DIR) for _ in range(3)}
        hidden_bad_results = {hidden_verify(case, bad, DEFAULT_HIDDEN_DIR) for _ in range(3)}
        assert public_results == {True}
        assert hidden_good_results == {True}
        assert hidden_bad_results == {False}


def test_hidden_verify_rejects_public_passing_but_hidden_failing_python_repair(
    cases: list[dict],
) -> None:
    by_id = {case["id"]: case for case in cases}
    for case_id, hack_solution in _PYREPAIR_PUBLIC_PASS_HIDDEN_FAIL.items():
        case = by_id[case_id]
        assert case["family"] == "python_repair"
        assert public_check(case, hack_solution) is True, (
            f"{case_id} hack solution was expected to pass the public test"
        )
        assert hidden_verify(case, hack_solution, DEFAULT_HIDDEN_DIR) is False, (
            f"{case_id} hack solution was expected to fail a hidden test"
        )


def test_default_cases_path_matches_fixture_location() -> None:
    assert DEFAULT_CASES_PATH == (
        Path(__file__).resolve().parent / "fixtures" / "harness_smoke" / "cases.jsonl"
    )
    assert DEFAULT_HIDDEN_DIR == (
        Path(__file__).resolve().parent / "fixtures" / "harness_smoke" / "hidden"
    )
