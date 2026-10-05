import datetime
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from google import genai

import arxiv_client as ac
from resilience import ModelCallFailed, call_with_retries, failed_result, partial_summary

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"
SYSTEM_PROMPT = (
    f"Today's date is {datetime.date.today()}. "
    "Take this into consideration when choosing recent papers/articles. "
    "Only cite papers that the arxiv_search tool returned. "
    "Do not fill gaps from training knowledge. Explicitly say that you couldn't find anything instead."
)
FINAL_ANSWER_INSTRUCTION = (
    "\n\nYou have reached the maximum number of tool-call iterations. "
    "Do not call any tools. Produce the final answer using ONLY the "
    "information already present in the conversation and tool results."
)

# Created lazily, so the offline tests run without an API key.
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=api_key)
    return _client


# ============================================================
# TOOLS
# ============================================================

VALID_SORTS = ("relevance", "lastUpdatedDate", "submittedDate")
VALID_CATEGORIES = ("cs.CV", "cs.LG", "cs.AI", "cs.CR", "eess.IV")
BROAD_SEARCH_THRESHOLD = 500


def arxiv_search(query, max_results=5, sort_by="relevance", category=None):
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string of title keywords")
    if category is not None and category not in VALID_CATEGORIES:
        raise ValueError(f"category must be one of {list(VALID_CATEGORIES)} or omitted")

    requested_sort = sort_by
    if sort_by not in VALID_SORTS:
        sort_by = "relevance"

    # int(): JSON numbers can arrive as floats (8.0); arXiv wants an integer.
    max_results = max(1, min(int(max_results), 10))

    raw = ac.search_raw(query, max_results, sort_by, category)

    # Lightweight results only: the model picks papers from titles,
    # then reads abstracts on demand with fetch_abstracts.
    papers = [
        {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "published": paper["published"],
        }
        for paper in raw["papers"]
    ]

    result = {
        "papers": papers,
        "total_matches": raw["total"],
        "sort_by_used": sort_by,
        "sort_by_requested": requested_sort,
    }
    if not papers:
        # Every keyword must be in the title, so zero results usually means
        # the words are too specific: synonyms or fewer keywords help.
        result["note"] = "No papers matched. Try fewer keywords or synonyms (e.g. 'forensics', 'synthetic')."
    elif raw["total"] > BROAD_SEARCH_THRESHOLD:
        result["note"] = "Very broad search. Add a keyword or a category to narrow it."
    return result


# Papers read in this process, keyed by arxiv_id. Filled by fetch_abstracts,
# read by save_notes. This is agent state that lives in CODE, not in the
# history: the model never has to re-type titles or authors.
PAPER_CACHE = {}
MAX_IDS_PER_FETCH = 10
NOTES_DIR = Path("notes")
METADATA_FIELDS = ("arxiv_id", "title", "authors", "published", "url")


def normalize_id(arxiv_id):
    """'2301.12345v2 ' -> '2301.12345'"""
    return re.sub(r"v\d+$", "", str(arxiv_id).strip())


def fetch_abstracts(arxiv_ids):
    if not isinstance(arxiv_ids, list) or not arxiv_ids:
        raise ValueError("arxiv_ids must be a non-empty list of arXiv IDs")
    ids = list(dict.fromkeys(normalize_id(i) for i in arxiv_ids))  # dedupe, keep order
    if len(ids) > MAX_IDS_PER_FETCH:
        raise ValueError(f"at most {MAX_IDS_PER_FETCH} IDs per call; split them into several calls")

    found = {p["arxiv_id"]: p for p in ac.fetch_by_ids_raw(ids) if p["abstract"]}
    PAPER_CACHE.update(found)

    result = {"papers": list(found.values())}
    not_found = [i for i in ids if i not in found]
    if not_found:
        result["not_found"] = not_found
        result["note"] = "Some IDs were not found. Use arxiv_ids exactly as returned by arxiv_search."
    return result


def save_notes(notes):
    if not isinstance(notes, list) or not notes:
        raise ValueError("notes must be a non-empty list of note objects")

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    saved, errors = [], []
    for note in notes:
        arxiv_id = normalize_id(note.get("arxiv_id", "")) if isinstance(note, dict) else ""
        paper = PAPER_CACHE.get(arxiv_id)
        if paper is None:
            # Guardrail: no note for a paper the agent never actually read.
            # Also means the filename below always comes from a real arXiv ID.
            errors.append(f"{arxiv_id or '<missing arxiv_id>'}: read it with fetch_abstracts "
                          "before saving a note")
            continue

        model_fields = {k: v for k, v in note.items() if k not in METADATA_FIELDS}
        record = {
            **{k: paper[k] for k in METADATA_FIELDS},  # metadata from arXiv, never from the model
            **model_fields,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        path = NOTES_DIR / f"{arxiv_id.replace('/', '_')}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        saved.append(arxiv_id)

    result = {"saved": saved}
    if errors:
        result["errors"] = errors
    return result


function_dispatch = {
    "arxiv_search": arxiv_search,
    "fetch_abstracts": fetch_abstracts,
    "save_notes": save_notes,
}

arxiv_search_function = {
    "type": "function",
    "name": "arxiv_search",
    "description": (
        "Searches arXiv paper TITLES only. Every keyword must appear in the title. "
        "Returns {'papers': [{'arxiv_id', 'title', 'authors', 'published'}], 'total_matches'} and "
        "sometimes a note on what to do when results are empty or too broad; follow the note's advice. "
        "It does NOT return abstracts or full text; use fetch_abstracts for that."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "2-3 short keywords that must all appear in the paper title. "
                               "Choose the most important concepts defining the user's research topic. "
                               "Avoid overly broad or generic keywords. "
                               "The query holds only the core topic.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "How many papers to return. 5-10 is usually enough.",
            },
            "sort_by": {
                "type": "string",
                "enum": list(VALID_SORTS),
                "description": (
                    "How to sort the results. "
                    "Use 'relevance' when the user wants the most relevant papers "
                    "or gives no preference. "
                    "Use 'submittedDate' when the user wants recent or new papers. "
                    "Use 'lastUpdatedDate' when the user wants recently updated papers."
                ),
            },
            "category": {
                "type": "string",
                "enum": list(VALID_CATEGORIES),
                "description": "Optional arXiv category to restrict the search to. "
                               "Use cs.CV for computer vision, cs.LG for machine learning, cs.AI for "
                               "artificial intelligence, cs.CR for cryptography, security, and some "
                               "digital forensics research, and eess.IV for image and video processing. "
                               "Choose the most specific applicable category rather than adding "
                               "multiple categories.",
            },
        },
        "required": ["query"],
    },
}

fetch_abstracts_function = {
    "type": "function",
    "name": "fetch_abstracts",
    "description": (
        "Fetches up to 10 arXiv papers by ID in ONE call and returns each paper's "
        "{'arxiv_id', 'title', 'authors', 'published', 'url', 'abstract'}. "
        "Pass every ID you want to read in a single call instead of calling this once per paper. "
        "Only use IDs returned by arxiv_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "arxiv_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_IDS_PER_FETCH,
                "description": "arxiv_ids exactly as returned by arxiv_search, e.g. ['2301.12345', '2302.00001']",
            },
        },
        "required": ["arxiv_ids"],
    },
}

save_notes_function = {
    "type": "function",
    "name": "save_notes",
    "description": "Save literature notes for papers that have already been read using "
                   "fetch_abstracts. ONE call to this function saves notes for ALL papers; "
                   "do not make separate calls for individual papers. "
                   "Only use information explicitly stated in each paper's abstract. "
                   "Do not infer, guess, or add information from the paper title, your "
                   "general knowledge, or assumptions. If the abstract does not provide "
                   "enough information for a field, write 'Not stated in abstract'. "
                   "The paper title, authors, publication date, and URL are filled in "
                   "automatically from the fetched paper metadata, so do not provide them.",
    "parameters": {
        "type": "object",
        "properties": {
            "notes": {
                "type": "array",
                "description": "One literature note per paper.",
                "items": {
                    "type": "object",
                    "properties": {
                        "arxiv_id": {
                            "type": "string",
                            "description": (
                                "The arXiv ID of a paper that was already "
                                "read using fetch_abstracts."
                            ),
                        },
                        "research_problem": {
                            "type": "string",
                            "description": (
                                "The research problem or question addressed by the paper, based only on the abstract. "
                                "If not stated, write 'Not stated in abstract'."
                            ),
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 4,
                            "description": (
                                "2-4 short findings, results, or conclusions explicitly stated in the abstract. "
                                "Use an empty list [] if the abstract does not state any findings."
                            ),
                        },
                        "datasets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Datasets or benchmarks explicitly mentioned in the abstract. "
                                "Use an empty list [] if none are mentioned."
                            ),
                        },
                        "usefulness_reason": {
                            "type": "string",
                            "description": (
                                "Briefly explain how the information explicitly stated in the abstract relates to the user query's research area. "
                                "Do not infer relevance from information not present in the abstract."
                                "The focus is judged when reading abstracts, not searched for."
                            )
                        },
                        "usefulness_score": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": (
                                "How useful this paper is for the topic chosen by the user, based only on the usefulness_reason and abstract. "
                                "'high' means the paper directly addresses the users research reason, "
                                "'medium' means the paper addresses a closely related technique, "
                                "dataset, evaluation method, or problem that can support the research reason and "
                                "'low' means the paper provides only general background or has limited direct connection to the user's research reason."
                            )
                        },
                    },
                    "required": [
                        "arxiv_id",
                        "research_problem",
                        "key_findings",
                        "datasets",
                        "usefulness_reason",
                        "usefulness_score",
                    ],
                },
            },
        },
        "required": ["notes"],
    },
}

TOOL_DECLARATIONS = [arxiv_search_function, fetch_abstracts_function, save_notes_function]

# ============================================================
# TOOL EXECUTION + ERROR HANDLING
# ============================================================

# Errors the model can fix by changing its arguments.
MODEL_FIXABLE_ERRORS = (ValueError, TypeError)
# Errors the model can't fix: the service is down or returned garbage.
SERVICE_ERRORS = (ac.ArxivUnavailableError, ET.ParseError)


def execute_tool(name, arguments, tools):
    """Run one tool call. Never raises: every failure becomes an error result."""
    if name not in tools:
        return {"error": f"Unknown tool {name!r}.", "available_tools": list(tools)}
    try:
        return tools[name](**(arguments or {}))
    except MODEL_FIXABLE_ERRORS as e:
        return {"error": f"Invalid request: {e}. Fix the arguments and try again."}
    except SERVICE_ERRORS as e:
        return {
            "error": f"arXiv is unavailable right now ({e}). Do not retry; "
                     "continue with the information you already have gotten from previous tool uses. "
                     "If no data is gathered, don't use training knowledge; explicitly say that no data could be found."
        }
    except Exception as e:  # last resort: a bug in our tool code
        return {"error": f"Tool {name!r} failed unexpectedly: {type(e).__name__}: {e}"}


def summarize_tool_result(name, result):
    """One short trace line describing what a tool returned (what the model will see)."""
    if not isinstance(result, dict):
        return f"{name}: returned {type(result).__name__}"
    if "error" in result:
        return f"{name}: ERROR {str(result['error'])[:120]}"

    if name == "arxiv_search":
        line = f"{len(result.get('papers', []))} papers (total matches: {result.get('total_matches', '?')})"
    elif name == "fetch_abstracts":
        line = f"{len(result.get('papers', []))} abstracts"
        if result.get("not_found"):
            line += f", not found: {result['not_found']}"
    elif name == "save_notes":
        line = f"saved {result.get('saved', [])}"
        if result.get("errors"):
            line += f", {len(result['errors'])} refused"
    else:
        line = "keys: " + ", ".join(result)

    if result.get("note"):
        line += f" | note: {result['note'][:80]}"
    return f"{name}: {line}"


def append_function_result(history, fc_steps, tools):
    for fc_step in fc_steps:
        result = execute_tool(fc_step.name, fc_step.arguments, tools)

        print(f"    -> {summarize_tool_result(fc_step.name, result)}")

        history.append({
            "type": "function_result",
            "name": fc_step.name,
            "call_id": fc_step.id,
            "result": [{"type": "text", "text": json.dumps(result)}],
        })


# ============================================================
# HISTORY HELPERS
# ============================================================

def add_user_message(input_text, history):
    history.append({
        "type": "user_input",
        "content": [{"type": "text", "text": input_text}],
    })


# ============================================================
# REAL MODEL
# ============================================================

def real_call_model(history, last_iter=False):
    if last_iter:
        # No tools on the wrap-up call: the API enforces "don't call tools".
        return get_client().interactions.create(
            model=MODEL_NAME,
            input=history,
            store=False,
            system_instruction=SYSTEM_PROMPT + FINAL_ANSWER_INSTRUCTION,
        )
    return get_client().interactions.create(
        model=MODEL_NAME,
        input=history,
        tools=TOOL_DECLARATIONS,
        store=False,
        system_instruction=SYSTEM_PROMPT,
    )


# ============================================================
# AGENT LOOP
# ============================================================

def run_agent(user_request, call_model, tools=None, max_iterations=5):
    tools = tools if tools is not None else function_dispatch
    history = []
    add_user_message(user_request, history)
    total_tokens = 0
    model_calls = 0

    for iteration in range(1, max_iterations + 1):
        try:
            interaction = call_with_retries(call_model, history)
        except ModelCallFailed as error:
            print(f"[iter {iteration}] model call failed for good: {error.kind}")
            return failed_result(error, history, model_calls, total_tokens)
        model_calls += 1
        tokens = interaction.usage.total_tokens
        total_tokens += tokens

        for step in interaction.steps:
            history.append(step.model_dump())

        function_calls = [s for s in interaction.steps if s.type == "function_call"]

        if not function_calls:
            print(f"[iter {iteration}] final answer | tokens: {tokens}")
            return {
                "answer": interaction.output_text,
                "hit_limit": False,
                "model_calls": model_calls,
                "total_tokens": total_tokens,
                "history": history,
                "error": None
            }

        for fc in function_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in (fc.arguments or {}).items())
            print(f"[iter {iteration}] called: {fc.name}({args}) | tokens: {tokens}")

        append_function_result(history, function_calls, tools)

    print(f"[iter {max_iterations}] max iterations reached, forcing final answer")
    try:
        final_interaction = call_with_retries(call_model, history, last_iter=True)
    except ModelCallFailed as error:
        print(f"[iter {iteration}] model call failed for good: {error.kind}")
        return failed_result(error, history, model_calls, total_tokens)
    model_calls += 1
    total_tokens += final_interaction.usage.total_tokens

    if not final_interaction.output_text:
        answer = partial_summary(history, "The model returned no final answer.")
    else:
        answer = final_interaction.output_text

    return {
        "answer": answer,
        "hit_limit": True,
        "model_calls": model_calls,
        "total_tokens": total_tokens,
        "history": history,
        "error": None
    }


# ============================================================
# FAKE MODEL + FAKE TOOLS (offline testing)
# ============================================================

def make_fake_step(step_type, name=None, arguments=None, step_id=None):
    step = SimpleNamespace(type=step_type, name=name, arguments=arguments, id=step_id)
    step.model_dump = lambda: {
        "type": step.type,
        "name": step.name,
        "arguments": step.arguments,
        "id": step.id,
    }
    return step


def make_fake_interaction(steps, total_tokens, output_text=None):
    return SimpleNamespace(
        steps=steps,
        usage=SimpleNamespace(total_tokens=total_tokens),
        output_text=output_text,
    )


def fake_tool_call(call_id, name="arxiv_search", arguments=None, tokens=100):
    if arguments is None:
        arguments = {"query": "diffusion image detection", "max_results": 5}
    step = make_fake_step("function_call", name, arguments, call_id)
    return make_fake_interaction([step], tokens)


def fake_answer(text, tokens=100):
    return make_fake_interaction([make_fake_step("model_output")], tokens, output_text=text)


def check_history(history):
    """Every function_result must answer a function_call that came before it."""
    seen_call_ids = set()
    for i, item in enumerate(history):
        if item["type"] == "function_call":
            seen_call_ids.add(item["id"])
        elif item["type"] == "function_result":
            assert item["call_id"] in seen_call_ids, (
                f"history[{i}]: function_result for call_id {item['call_id']!r} "
                f"has no earlier matching function_call"
            )


class FakeModel:
    """Replays scripted responses, checks the history it receives, counts calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def __call__(self, history, last_iter=False):
        check_history(history)
        self.call_count += 1
        if last_iter:
            return fake_answer("Wrap-up answer from gathered results.", tokens=50)
        if not self.responses:
            raise RuntimeError("Fake model ran out of scripted responses.")
        return self.responses.pop(0)


FAKE_PAPER = {
    "arxiv_id": "2601.99999",
    "title": "Frequency Echoes: Spotting Diffusion Ghosts",
    "authors": ["Elena Voss", "Rohan Merriweather"],
    "published": "2026-01-15",
    "url": "http://arxiv.org/abs/2601.99999v1",
    "abstract": "We detect diffusion images via frequency artifacts.",
}


def fake_arxiv_search(query, max_results=5, sort_by="relevance", category=None):
    summary = {k: FAKE_PAPER[k] for k in ("arxiv_id", "title", "authors", "published")}
    return {"papers": [summary], "total_matches": 1,
            "sort_by_used": sort_by, "sort_by_requested": sort_by}


def fake_fetch_by_ids_raw(arxiv_ids):
    """Stands in for arXiv: only FAKE_PAPER exists."""
    return [dict(FAKE_PAPER)] if FAKE_PAPER["arxiv_id"] in arxiv_ids else []


def broken_arxiv_search(**kwargs):
    raise ac.ArxivUnavailableError("simulated network failure")


def no_network(params):
    raise AssertionError(f"Offline test tried to reach arXiv with {params}")


# Real fetch_abstracts and save_notes run in tests (with arXiv faked and
# notes written to a temp folder), because their logic is what we're testing.
FAKE_TOOLS = {"arxiv_search": fake_arxiv_search,
              "fetch_abstracts": fetch_abstracts,
              "save_notes": save_notes}
BROKEN_TOOLS = {**FAKE_TOOLS, "arxiv_search": broken_arxiv_search}


def tool_results(history):
    """Parsed payloads of every function_result in the history."""
    return [json.loads(item["result"][0]["text"])
            for item in history if item["type"] == "function_result"]


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}: {name}{' | ' + detail if detail else ''}")


def test_tool_logic():
    print("=== Tool logic ===")
    original_search = ac.search_raw
    try:
        captured = {}
        ac.search_raw = lambda q, n, s, c: captured.update(q=q, n=n, s=s, c=c) or {"total": 0, "papers": []}
        result = arxiv_search("diffusion detection", max_results=50.0, sort_by="newest", category="cs.CV")
        report("search: clamps max_results, falls back on invalid sort_by, passes category",
               captured == {"q": "diffusion detection", "n": 10, "s": "relevance", "c": "cs.CV"}
               and result["sort_by_requested"] == "newest")
        report("search: zero results come with a hint", "note" in result)

        for bad_call, label in [(lambda: arxiv_search("   "), "empty query"),
                                (lambda: arxiv_search("x", category="astro-ph"), "unknown category")]:
            try:
                bad_call()
                report(f"search: rejects {label}", False)
            except ValueError:
                report(f"search: rejects {label}", True)

        ac.search_raw = lambda q, n, s, c: {"total": 8970, "papers": [FAKE_PAPER]}
        result = arxiv_search("diffusion")
        report("search: broad search gets a hint, abstracts stripped",
               "note" in result and "abstract" not in result["papers"][0])
    finally:
        ac.search_raw = original_search

    PAPER_CACHE.clear()
    result = fetch_abstracts(["2601.99999v2", "2601.99999", "9999.00000"])
    report("fetch: normalizes versions, dedupes, reports not_found, fills cache",
           len(result["papers"]) == 1 and result["not_found"] == ["9999.00000"]
           and "2601.99999" in PAPER_CACHE)
    try:
        fetch_abstracts([f"2601.{i:05d}" for i in range(11)])
        report("fetch: rejects more than 10 IDs", False)
    except ValueError:
        report("fetch: rejects more than 10 IDs", True)

    result = save_notes([
        {"arxiv_id": "2601.99999", "title": "WRONG TITLE", "key_findings": ["finding"]},
        {"arxiv_id": "2602.11111", "key_findings": ["never read"]},
    ])
    saved = json.loads((NOTES_DIR / "2601.99999.json").read_text(encoding="utf-8"))
    report("save: writes note with metadata from arXiv, not the model",
           result["saved"] == ["2601.99999"] and saved["title"] == FAKE_PAPER["title"]
           and saved["key_findings"] == ["finding"])
    report("save: refuses a note for a paper never fetched",
           "errors" in result and not (NOTES_DIR / "2602.11111.json").exists())


def run_tests():
    global NOTES_DIR
    request = "Find recent papers on detecting diffusion-generated images"

    # Offline guarantees: any real arXiv request fails the test, fetches are
    # faked, and notes go to a throwaway folder.
    original_get, original_fetch, original_notes_dir = ac._get, ac.fetch_by_ids_raw, NOTES_DIR
    ac._get, ac.fetch_by_ids_raw = no_network, fake_fetch_by_ids_raw
    NOTES_DIR = Path(tempfile.mkdtemp(prefix="notes_test_"))
    try:
        print("=== History check ===")
        broken_history = [
            {"type": "user_input", "content": [{"type": "text", "text": request}]},
            {"type": "function_result", "name": "arxiv_search", "call_id": "call_1", "result": []},
        ]
        try:
            check_history(broken_history)
            report("history check catches a result with no matching call", False)
        except AssertionError:
            report("history check catches a result with no matching call", True)

        print()
        test_tool_logic()

        # (name, script, tools, expected model calls, expected hit_limit, extra check)
        scenarios = [
            ("(a) answers immediately",
             [fake_answer("Direct answer.")], FAKE_TOOLS, 1, False, None),
            ("(b) two searches, then answers",
             [fake_tool_call("call_1"), fake_tool_call("call_2"), fake_answer("Found papers.")],
             FAKE_TOOLS, 3, False, None),
            ("(c) never stops calling tools",
             [fake_tool_call(f"call_{i}") for i in range(1, 6)], FAKE_TOOLS, 6, True, None),
            ("(d) search -> batch fetch -> save notes -> answer",
             [fake_tool_call("call_1"),
              fake_tool_call("call_2", "fetch_abstracts", {"arxiv_ids": ["2601.99999"]}),
              fake_tool_call("call_3", "save_notes",
                             {"notes": [{"arxiv_id": "2601.99999", "key_findings": ["x"]}]}),
              fake_answer("Saved 1 note.")],
             FAKE_TOOLS, 4, False,
             lambda r: tool_results(r["history"])[2]["saved"] == ["2601.99999"]),
            ("(e) arXiv down -> error result, loop survives",
             [fake_tool_call("call_1"), fake_answer("Sorry, arXiv is down.")],
             BROKEN_TOOLS, 2, False,
             lambda r: "unavailable" in tool_results(r["history"])[0]["error"]),
            ("(f) unknown tool -> error lists available tools",
             [fake_tool_call("call_1", "summarize_pdf", {"url": "x"}), fake_answer("OK.")],
             FAKE_TOOLS, 2, False,
             lambda r: "available_tools" in tool_results(r["history"])[0]),
            ("(g) bad arguments -> fixable error",
             [fake_tool_call("call_1", "arxiv_search", {"topic": "diffusion"}), fake_answer("OK.")],
             FAKE_TOOLS, 2, False,
             lambda r: tool_results(r["history"])[0]["error"].startswith("Invalid request")),
        ]

        for name, script, tools, expected_calls, expected_limit, check in scenarios:
            PAPER_CACHE.clear()
            print(f"\n=== {name} ===")
            model = FakeModel(script)
            result = run_agent(request, model, tools=tools, max_iterations=5)
            ok = (
                    model.call_count == expected_calls
                    and result["hit_limit"] == expected_limit
                    and (check is None or check(result))
            )
            report(name, ok, f"calls predicted={expected_calls} actual={model.call_count}, "
                             f"hit_limit={result['hit_limit']}")
    finally:
        ac._get, ac.fetch_by_ids_raw, NOTES_DIR = original_get, original_fetch, original_notes_dir
        PAPER_CACHE.clear()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        result = run_agent(
            "Find recent papers on detecting diffusion-generated images, "
            "read the most relevant ones and save notes on them.",
            real_call_model,
            max_iterations=6,
        )
        print("\nFinal output:\n" + (result["answer"] or ""))
        print(
            f"\nModel calls: {result['model_calls']} | "
            f"Hit limit: {result['hit_limit']} | Total tokens: {result['total_tokens']}"
        )
