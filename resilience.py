import json
import random
import re
import time

MAX_ATTEMPTS = 3  # 1 try + 2 retries
BASE_DELAY_SECONDS = 2.0  # exponential backoff: 2s, 4s, 8s...
MAX_DELAY_SECONDS = 60.0


class ModelCallFailed(Exception):
    """The model call failed for good (not retryable, or retries used up)."""

    def __init__(self, kind, message, attempts):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.attempts = attempts


def _status_code(error):
    """HTTP status from an SDK error: an int attribute, else 'Error code: 429' in the text."""
    for attr in ("status_code", "code", "status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    match = re.search(r"error code:\s*(\d{3})", str(error).lower())
    return int(match.group(1)) if match else None


def classify_error(error):
    """Return (kind, retryable, retry_after_seconds or None)."""
    text = str(error).lower()
    status = _status_code(error)
    match = re.search(r"retry in\s*([\d.]+)\s*s", text)
    retry_after = float(match.group(1)) if match else None

    if status == 429:
        # Daily quota: waiting seconds won't help, even if the message says "retry in 29s".
        if "per day" in text or "perday" in text:
            return "daily_quota_exhausted", False, None
        return "rate_limited", True, retry_after
    if status is not None and 500 <= status < 600:
        return "server_error", True, retry_after  # overloaded / transient: retry
    if status is not None and 400 <= status < 500:
        return "client_error", False, None  # our request is wrong: a code bug
    if isinstance(error, (TimeoutError, ConnectionError)) or any(
            word in type(error).__name__ for word in ("Timeout", "Connect")
    ):
        return "network_error", True, None
    return "unknown_error", False, None


def call_with_retries(call_model, history, last_iter=False, sleep=time.sleep, log=print):
    """Call the model, retrying transient failures with exponential backoff + jitter.

    Raises ModelCallFailed when the error isn't retryable or attempts run out.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call_model(history, last_iter=last_iter)
        except AssertionError:
            raise  # test-harness checks (e.g. check_history) must never be swallowed
        except Exception as error:
            kind, retryable, retry_after = classify_error(error)
            if not retryable or attempt == MAX_ATTEMPTS:
                raise ModelCallFailed(kind, str(error)[:300], attempt) from error
            delay = retry_after if retry_after is not None else BASE_DELAY_SECONDS * 2 ** (attempt - 1)
            delay = min(delay, MAX_DELAY_SECONDS) + random.uniform(0, 1)  # jitter
            log(f"    ! model call failed ({kind}); retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.1f}s")
            sleep(delay)


def _partial_work(history):
    """Papers read and notes saved so far, recovered from tool results in the history."""
    papers_read, notes_saved = {}, []
    for item in history:
        if item.get("type") != "function_result":
            continue
        try:
            payload = json.loads(item["result"][0]["text"])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if item.get("name") == "fetch_abstracts":
            for paper in payload.get("papers", []):
                papers_read[paper["arxiv_id"]] = paper.get("title", "")
        elif item.get("name") == "save_notes":
            notes_saved.extend(payload.get("saved", []))
    return papers_read, notes_saved


def partial_summary(history, headline):
    """A code-written answer from the work recorded in the history.

    Used whenever the model can't give the answer itself: the model call
    failed for good, or the wrap-up call came back empty.
    """
    papers_read, notes_saved = _partial_work(history)
    lines = [headline]
    if not papers_read and not notes_saved:
        lines.append("No papers were read, so there is nothing to report.")
    else:
        if papers_read:
            lines.append("Papers read: " + "; ".join(f"{t} ({i})" for i, t in papers_read.items()))
        lines.append(f"Notes saved in notes/: {', '.join(notes_saved)}" if notes_saved
                     else "No notes were saved.")
    return "\n".join(lines)


def fallback_answer(error, history):
    """Written by CODE, because the model is the thing that just failed."""
    return partial_summary(
        history,
        f"The run stopped early: the model API failed ({error.kind}) after "
        f"{error.attempts} attempt(s). This is a partial result.",
    )


def failed_result(error, history, model_calls, total_tokens):
    """The run_agent result for a run that ended on a model-call failure."""
    return {
        "answer": fallback_answer(error, history),
        "hit_limit": False,
        "model_calls": model_calls,
        "total_tokens": total_tokens,
        "history": history,
        "error": {"kind": error.kind, "message": error.message, "attempts": error.attempts},
    }


# ============================================================
# SELF-TEST (offline, no sleeping)
# ============================================================

class _FakeAPIError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def _self_test():
    def report(name, ok):
        print(f"{'PASS' if ok else 'FAIL'}: {name}")

    def scripted(outcomes):
        """A call_model that raises/returns the scripted outcomes in order."""
        outcomes = list(outcomes)
        calls = {"n": 0}

        def call_model(history, last_iter=False):
            calls["n"] += 1
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return call_model, calls

    slept = []
    quiet = lambda *a: None

    call, calls = scripted([_FakeAPIError(503, "model overloaded"), "OK"])
    result = call_with_retries(call, [], sleep=slept.append, log=quiet)
    report("503 is retried, then succeeds", result == "OK" and calls["n"] == 2 and len(slept) == 1)

    slept.clear()
    call, calls = scripted([_FakeAPIError(429, "limit: 15 requests per minute. Please retry in 7s."), "OK"])
    call_with_retries(call, [], sleep=slept.append, log=quiet)
    report("per-minute 429 waits the server's suggested delay", 7 <= slept[0] < 8)

    for status, message, kind in [
        (429, "Rate limit exceeded (limit: 20 requests per day). Please retry in 29s.", "daily_quota_exhausted"),
        (400, "Invalid request: bad history", "client_error"),
    ]:
        call, calls = scripted([_FakeAPIError(status, message), "OK"])
        try:
            call_with_retries(call, [], sleep=slept.append, log=quiet)
            report(f"{kind} is not retried", False)
        except ModelCallFailed as e:
            report(f"{kind} is not retried", e.kind == kind and calls["n"] == 1)

    slept.clear()
    call, calls = scripted([_FakeAPIError(500, "boom")] * 3)
    try:
        call_with_retries(call, [], sleep=slept.append, log=quiet)
        report("gives up after MAX_ATTEMPTS", False)
    except ModelCallFailed as e:
        report("gives up after MAX_ATTEMPTS, delays grow",
               calls["n"] == MAX_ATTEMPTS and e.attempts == MAX_ATTEMPTS and slept[1] > slept[0])

    call, _ = scripted([AssertionError("history broken")])
    try:
        call_with_retries(call, [], sleep=slept.append, log=quiet)
        report("test assertions are never swallowed", False)
    except AssertionError:
        report("test assertions are never swallowed", True)

    history = [
        {"type": "function_result", "name": "fetch_abstracts", "call_id": "c1",
         "result": [{"type": "text",
                     "text": json.dumps({"papers": [{"arxiv_id": "2601.99999", "title": "Frequency Echoes"}]})}]},
        {"type": "function_result", "name": "save_notes", "call_id": "c2",
         "result": [{"type": "text", "text": json.dumps({"saved": ["2601.99999"]})}]},
    ]
    error = ModelCallFailed("server_error", "boom", 3)
    answer = failed_result(error, history, 3, 1000)["answer"]
    report("partial result lists papers read and notes saved",
           "Frequency Echoes" in answer and "2601.99999" in answer and "partial" in answer)
    report("empty run says there is nothing to report",
           "nothing to report" in fallback_answer(error, []))
    report("partial_summary works for an empty wrap-up answer",
           "2601.99999" in partial_summary(history, "The model returned an empty answer."))


if __name__ == "__main__":
    _self_test()
