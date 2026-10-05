import sys

from agent import PAPER_CACHE, real_call_model, run_agent, save_notes_function
from evaluation import ids_in_answer, notes_saved, print_report, tool_payloads, evaluate_run

NOTE_SCHEMA = save_notes_function["parameters"]["properties"]["notes"]["items"]


def retried_if_first_search_empty(result):
    """If the first search found nothing, the model must have searched again."""
    searches = tool_payloads(result, "arxiv_search")
    if searches and not searches[0].get("papers"):
        return len(searches) >= 2
    return True


CASES = [
    {
        "name": "normal",
        "request": ("Find recent papers on detecting diffusion-generated images, focusing on "
                    "lightweight or explainable detectors. Read the most relevant ones and save notes on them."),
        "expect": [("saved at least one note", lambda r: len(notes_saved(r)) > 0)],
    },
    {
        "name": "narrow",
        "request": ("Find papers on detecting breast cancer in male patients over 50 years, "
                    "and save notes on the relevant ones."),
        "expect": [("retried the search if the first one was empty", retried_if_first_search_empty)],
    },
    {
        "name": "off-topic",
        "request": "Tell me the top 10 strongest Marvel characters.",
        "expect": [
            ("saved no notes", lambda r: not notes_saved(r)),
            ("cited no arXiv IDs", lambda r: not ids_in_answer(r)),
        ],
    },
]


def main():
    selected = [c for c in CASES if len(sys.argv) < 2 or c["name"] == sys.argv[1]]
    summary = []
    for case in selected:
        print(f"\n##### {case['name']}: {case['request']}")
        PAPER_CACHE.clear()  # each case starts with no papers read
        result = run_agent(case["request"], real_call_model, max_iterations=6)
        print("\nAnswer (first 400 chars):\n" + (result["answer"] or "")[:400])
        checks = evaluate_run(result, NOTE_SCHEMA, case["expect"])
        print_report(case["name"], checks)
        summary.append((case["name"], sum(ok for _, ok, _ in checks), len(checks)))

    print("\n##### Summary")
    for name, passed, total in summary:
        print(f"  {name}: {passed}/{total}")


if __name__ == "__main__":
    main()
