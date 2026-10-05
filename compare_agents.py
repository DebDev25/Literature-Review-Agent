import json
from pathlib import Path
import tempfile

import agent
import arxiv_client as ac
from agent import (BROKEN_TOOLS, FAKE_TOOLS, FakeModel, fake_answer, fake_tool_call,
                   fake_fetch_by_ids_raw, no_network)
from evaluation import tool_payloads
from graph_agent import run_graph_agent

REQUEST = "Find recent papers on detecting diffusion-generated images"


def tool_results(result):
    return tool_payloads(result)


SCENARIOS = [
    ("(a) answers immediately",
     lambda: [fake_answer("Direct answer.")], FAKE_TOOLS, 1, False, None),
    ("(b) two searches, then answers",
     lambda: [fake_tool_call("call_1"), fake_tool_call("call_2"), fake_answer("Found papers.")],
     FAKE_TOOLS, 3, False, None),
    ("(c) never stops calling tools",
     lambda: [fake_tool_call(f"call_{i}") for i in range(1, 6)], FAKE_TOOLS, 6, True, None),
    ("(d) search -> batch fetch -> save notes -> answer",
     lambda: [fake_tool_call("call_1"),
              fake_tool_call("call_2", "fetch_abstracts", {"arxiv_ids": ["2601.99999"]}),
              fake_tool_call("call_3", "save_notes",
                             {"notes": [{"arxiv_id": "2601.99999", "key_findings": ["x"]}]}),
              fake_answer("Saved 1 note.")],
     FAKE_TOOLS, 4, False, lambda r: tool_results(r)[2]["saved"] == ["2601.99999"]),
    ("(e) arXiv down -> error result, loop survives",
     lambda: [fake_tool_call("call_1"), fake_answer("Sorry, arXiv is down.")],
     BROKEN_TOOLS, 2, False, lambda r: "unavailable" in tool_results(r)[0]["error"]),
    ("(f) unknown tool -> error lists available tools",
     lambda: [fake_tool_call("call_1", "summarize_pdf", {"url": "x"}), fake_answer("OK.")],
     FAKE_TOOLS, 2, False, lambda r: "available_tools" in tool_results(r)[0]),
    ("(g) bad arguments -> fixable error",
     lambda: [fake_tool_call("call_1", "arxiv_search", {"topic": "diffusion"}), fake_answer("OK.")],
     FAKE_TOOLS, 2, False, lambda r: tool_results(r)[0]["error"].startswith("Invalid request")),
]

IMPLEMENTATIONS = [("loop ", agent.run_agent), ("graph", run_graph_agent)]


def main():
    original_get, original_fetch, original_notes = ac._get, ac.fetch_by_ids_raw, agent.NOTES_DIR
    ac._get, ac.fetch_by_ids_raw = no_network, fake_fetch_by_ids_raw
    agent.NOTES_DIR = Path(tempfile.mkdtemp(prefix="notes_compare_"))
    failures = 0
    try:
        for name, script, tools, expected_calls, expected_limit, check in SCENARIOS:
            print(f"\n=== {name} ===")
            for label, run in IMPLEMENTATIONS:
                agent.PAPER_CACHE.clear()
                model = FakeModel(script())
                result = run(REQUEST, model, tools=tools, max_iterations=5)
                ok = (model.call_count == expected_calls
                      and result["model_calls"] == expected_calls
                      and result["hit_limit"] == expected_limit
                      and result["error"] is None
                      and bool(result["answer"])
                      and (check is None or check(result)))
                failures += not ok
                print(f"  {'PASS' if ok else 'FAIL'} [{label}] calls={model.call_count} "
                      f"(expected {expected_calls}), hit_limit={result['hit_limit']}")

        # Both implementations must produce the SAME history for the same script.
        print("\n=== both produce an identical history ===")
        for name, script, tools, *_ in SCENARIOS:
            results = []
            for _, run in IMPLEMENTATIONS:
                agent.PAPER_CACHE.clear()
                results.append(run(REQUEST, FakeModel(script()), tools=tools, max_iterations=5))
            same = json.dumps(results[0]["history"], sort_keys=True, default=str) == \
                   json.dumps(results[1]["history"], sort_keys=True, default=str)
            failures += not same
            print(f"  {'PASS' if same else 'FAIL'} {name}")
    finally:
        ac._get, ac.fetch_by_ids_raw, agent.NOTES_DIR = original_get, original_fetch, original_notes
        agent.PAPER_CACHE.clear()
    print(f"\n{'ALL PASSED' if not failures else str(failures) + ' FAILURES'}")


if __name__ == "__main__":
    main()
