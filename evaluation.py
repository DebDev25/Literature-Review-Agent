import json
import re
from collections import Counter

# 2507.02995 or 2507.02995v2 (not part of a longer number)
ARXIV_ID_PATTERN = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")


# ============================================================
# READING THE HISTORY
# ============================================================

def tool_call_args(result, name):
    """Arguments of every call the model made to the named tool, in order."""
    return [item.get("arguments") or {} for item in result["history"]
            if item.get("type") == "function_call" and item.get("name") == name]


def tool_payloads(result, name=None):
    """Parsed tool results (optionally for one tool), in order."""
    payloads = []
    for item in result["history"]:
        if item.get("type") != "function_result" or (name and item.get("name") != name):
            continue
        try:
            payloads.append(json.loads(item["result"][0]["text"]))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return payloads


def _collect_ids(value, found):
    """Every 'arxiv_id' value anywhere inside a nested payload."""
    if isinstance(value, dict):
        for key, inner in value.items():
            if key == "arxiv_id" and isinstance(inner, str):
                found.add(inner)
            else:
                _collect_ids(inner, found)
    elif isinstance(value, list):
        for inner in value:
            _collect_ids(inner, found)
    return found


def ids_from_tools(result):
    return _collect_ids(tool_payloads(result), set())


def ids_in_answer(result):
    return set(ARXIV_ID_PATTERN.findall(result.get("answer") or ""))


def notes_sent(result):
    return [note for args in tool_call_args(result, "save_notes") for note in args.get("notes", [])]


def notes_saved(result):
    return [i for payload in tool_payloads(result, "save_notes") for i in payload.get("saved", [])]


# ============================================================
# SCHEMA VALIDATION (uses your own tool schema as the spec)
# ============================================================

def validate(value, schema, path="note"):
    """Minimal JSON Schema check: type, required, enum, minItems/maxItems."""
    errors = []
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected an object"]
        errors += [f"{path}.{field}: missing" for field in schema.get("required", []) if field not in value]
        for field, sub_schema in schema.get("properties", {}).items():
            if field in value:
                errors += validate(value[field], sub_schema, f"{path}.{field}")
    elif kind == "array":
        if not isinstance(value, list):
            return [f"{path}: expected a list"]
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: {len(value)} items, max {schema['maxItems']}")
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: {len(value)} items, min {schema['minItems']}")
        for i, item in enumerate(value):
            errors += validate(item, schema.get("items", {}), f"{path}[{i}]")
    elif kind == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected a string")
        elif "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: {value!r} not in {schema['enum']}")
    elif kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        errors.append(f"{path}: expected an integer")
    return errors


# ============================================================
# CHECKS: each returns (passed: bool, detail: str)
# ============================================================

def check_completed(result):
    error = result.get("error")
    return error is None, "no model-call failure" if error is None else f"failed: {error['kind']}"


def check_answer_present(result):
    """An empty answer must fail, otherwise other checks can pass vacuously."""
    answer = (result.get("answer") or "").strip()
    return bool(answer), f"{len(answer)} characters" if answer else "EMPTY answer"


def check_grounding(result):
    """Every arXiv ID in the answer or in a note must have come from a tool result."""
    seen = ids_from_tools(result)
    claimed = ids_in_answer(result) | {note.get("arxiv_id", "") for note in notes_sent(result)}
    ungrounded = sorted(claimed - seen)
    return not ungrounded, (f"{len(claimed)} IDs cited, all from tools" if not ungrounded
                            else f"NOT from any tool result: {ungrounded}")


def check_notes_schema(result, note_schema):
    notes = notes_sent(result)
    errors = [error for note in notes for error in validate(note, note_schema)]
    return not errors, (f"{len(notes)} notes valid" if not errors else "; ".join(errors[:5]))


def check_rating_spread(result, field="usefulness_score", min_notes=4):
    """With enough notes, a rating that's always the same carries no information."""
    counts = Counter(note.get(field) for note in notes_sent(result))
    total = sum(counts.values())
    ok = total < min_notes or len(counts) > 1
    return ok, f"{dict(counts)}" + ("" if ok else " (no spread: the rating isn't discriminating)")


def check_efficiency(result, max_model_calls=6):
    saved = len(notes_saved(result))
    per_note = f", {result['total_tokens'] // saved} tokens/note" if saved else ""
    ok = not result["hit_limit"] and result["model_calls"] <= max_model_calls
    return ok, (f"{result['model_calls']} model calls, {result['total_tokens']} tokens{per_note}"
                + (", HIT THE CAP" if result["hit_limit"] else ""))


def evaluate_run(result, note_schema, expectations=()):
    """Run the standard checks plus case-specific expectations: [(name, fn(result) -> bool)]."""
    checks = [
        ("completed", *check_completed(result)),
        ("answer present", *check_answer_present(result)),
        ("grounding", *check_grounding(result)),
        ("notes match schema", *check_notes_schema(result, note_schema)),
        ("rating spread", *check_rating_spread(result)),
        ("efficiency", *check_efficiency(result)),
    ]
    checks += [(name, bool(fn(result)), "") for name, fn in expectations]
    return checks


def print_report(case_name, checks):
    passed = sum(ok for _, ok, _ in checks)
    print(f"\n=== {case_name}: {passed}/{len(checks)} checks passed ===")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")


# ============================================================
# SELF-TEST (hand-built results, offline)
# ============================================================

def _self_test():
    note_schema = {
        "type": "object",
        "properties": {
            "arxiv_id": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "usefulness_score": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["arxiv_id", "key_findings", "usefulness_score"],
    }

    def call(name, arguments, call_id):
        return {"type": "function_call", "name": name, "arguments": arguments, "id": call_id}

    def res(name, payload, call_id):
        return {"type": "function_result", "name": name, "call_id": call_id,
                "result": [{"type": "text", "text": json.dumps(payload)}]}

    good_note = {"arxiv_id": "2601.99999", "key_findings": ["x"], "usefulness_score": "high"}
    good = {
        "answer": "Read Frequency Echoes (2601.99999v1).",
        "hit_limit": False, "model_calls": 4, "total_tokens": 4000, "error": None,
        "history": [
            call("fetch_abstracts", {"arxiv_ids": ["2601.99999"]}, "c1"),
            res("fetch_abstracts", {"papers": [{"arxiv_id": "2601.99999", "title": "t"}]}, "c1"),
            call("save_notes", {"notes": [good_note]}, "c2"),
            res("save_notes", {"saved": ["2601.99999"]}, "c2"),
        ],
    }
    bad_note = {"arxiv_id": "2601.99999", "key_findings": ["a", "b", "c", "d", "e"], "usefulness_score": "great"}
    bad = {**good,
           "answer": "See also 2403.12345, a classic.",  # never returned by a tool
           "hit_limit": True, "model_calls": 6,
           "history": good["history"][:2] + [call("save_notes", {"notes": [bad_note]}, "c2")]}

    good_checks = {name: ok for name, ok, _ in evaluate_run(good, note_schema)}
    bad_checks = {name: ok for name, ok, _ in evaluate_run(bad, note_schema)}
    print(f"{'PASS' if all(good_checks.values()) else 'FAIL'}: a good run passes every check")
    empty = {**good, "answer": "   "}
    print(f"{'PASS' if not check_answer_present(empty)[0] else 'FAIL'}: an empty answer fails 'answer present'")
    for name in ("grounding", "notes match schema", "efficiency"):
        print(f"{'PASS' if not bad_checks[name] else 'FAIL'}: a bad run fails '{name}'")

    same = {**good, "history": [call("save_notes", {"notes": [dict(good_note)] * 5}, "c9")]}
    print(f"{'PASS' if not check_rating_spread(same)[0] else 'FAIL'}: 5 identical ratings fail the spread check")


if __name__ == "__main__":
    _self_test()
