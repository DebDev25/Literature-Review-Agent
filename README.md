# Literature Scout

A ReAct-style research agent that searches arXiv, reads abstracts, and writes structured literature notes to disk. The agent loop is built **by hand** on raw API calls, with no framework, and then ported to LangGraph so the two implementations can be compared against the same test suite.

Built as a learning project: the goal was to understand tool calling, the agent loop, state, guardrails and evaluation from first principles, rather than to wire up a framework.

```
[iter 1] called: arxiv_search(query='diffusion lightweight detector', category='cs.CV', sort_by='submittedDate')
    -> arxiv_search: 0 papers (total matches: 0) | note: No papers matched. Try fewer keywords or synonyms...
[iter 2] called: arxiv_search(query='diffusion generated image detection', category='cs.CV', sort_by='submittedDate')
    -> arxiv_search: 10 papers (total matches: 30)
[iter 3] called: fetch_abstracts(arxiv_ids=['2608.12155', '2604.09106', ... ])
    -> fetch_abstracts: 8 abstracts
[iter 4] called: save_notes(notes=[ ... ])
    -> save_notes: saved ['2604.09106', '2511.00352', '2507.02995', '2603.14220', '2608.12155']
[iter 5] final answer | tokens: 6783
```

## What it does

Given a research topic (and optionally a focus, such as "lightweight or explainable detectors"), the agent:

1. searches arXiv by title keywords, narrowing or broadening when a search comes back empty or far too wide;
2. batch-fetches the abstracts it wants to read, in a single call;
3. writes one structured note per paper into `notes/<arxiv_id>.json`: research problem, key findings, datasets, a usefulness rating and the reason for it;
4. answers with a summary that cites only papers the tools actually returned.

A saved note:

```json
{
  "arxiv_id": "2507.02995",
  "title": "FreqCross: A Multi-Modal Frequency-Spatial Fusion Network ...",
  "authors": ["Guang Yang"],
  "published": "2025-07-01",
  "url": "http://arxiv.org/abs/2507.02995v2",
  "research_problem": "Robust detection of Stable Diffusion 3.5 generated images ...",
  "key_findings": ["FreqCross achieves 97.8% accuracy ...", "..."],
  "datasets": ["MS-COCO"],
  "usefulness_reason": "Presents a multi-modal network with a lightweight CNN ...",
  "usefulness_score": "medium",
  "saved_at": "2026-09-29T23:41:02"
}
```

Paper metadata (title, authors, date, URL) is filled in by the code from the arXiv response, never retyped by the model.

## Architecture

```
user request
     │
     ▼
run_agent(user_request, call_model, tools, max_iterations)
     │  history = [user_input]              ← the agent's entire state
     │
     ├─► loop, at most max_iterations times:
     │      interaction = call_model(history)        ← stateless LLM call (with retries)
     │      append EVERY returned step to history    (thoughts, tool calls, text)
     │      no function_call steps? → return the answer          (normal stop)
     │      else: execute each tool  → append function_result    (linked by call_id)
     │
     └─► cap reached → run pending tools, then one wrap-up call with NO tools
                       ("answer from what you already have")

Returns: {answer, hit_limit, model_calls, total_tokens, history, error}
```

The LangGraph version replaces the loop with a `StateGraph`:

```
START → call_model ──(no tool calls, or a failed call)──→ END
                   └─(tool calls)─────────────────────→ run_tools
        run_tools ──(under the cap)───────────────────→ call_model
                   └─(at the cap)────────────────────→ force_answer → END
```

`run_graph_agent` has the same signature and returns the same result dict as `run_agent`, so every test and the whole evaluation harness run unchanged against both.

## Tools

| Tool | Purpose |
|---|---|
| `arxiv_search(query, max_results, sort_by, category)` | Title-keyword search. Returns compact results **without** abstracts, plus `total_matches` and a note when the search is empty or too broad. |
| `fetch_abstracts(arxiv_ids)` | Up to 10 papers in one request, abstracts included. |
| `save_notes(notes)` | Writes all notes in one call. Refuses any paper that was not actually fetched. |

Search results are deliberately lightweight: ten abstracts would be about 2,500 tokens sitting in the history and re-sent on every later call, mostly for papers the model rejects by title. Abstracts are fetched on demand instead.

## Setup

```bash
git clone git@github.com:DebDev25/Literature-Review-Agent.git && cd Literature-Review-Agent
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install google-genai python-dotenv requests langgraph
echo "GEMINI_API_KEY=your_key_here" > .env             # https://aistudio.google.com/apikey
```

Add `.env` and `notes/` to `.gitignore`.

## Usage

```bash
python agent.py              # one real run (hand-written loop)
python graph_agent.py        # the same run through the LangGraph port

python agent.py test         # offline tests: scripted fake model + fake tools, no network
python resilience.py         # retry/backoff self-tests
python evaluation.py         # evaluation self-tests
python compare_agents.py     # every scenario on BOTH implementations + history diff

python eval_runs.py          # three real evaluation cases (~10-15 API requests)
python eval_runs.py normal   # just one case
```

Every offline command runs with **zero API calls and zero network access**: the fake model is scripted, the arXiv client is stubbed out, and any accidental HTTP request fails the test.

## Project structure

| File | Contents |
|---|---|
| `agent.py` | Tool schemas and functions, tool execution and error classes, the agent loop, the fake model and the offline test suite |
| `arxiv_client.py` | arXiv HTTP and Atom XML plumbing, rate limiting |
| `resilience.py` | Error classification, retries with exponential backoff and jitter, code-written fallback answers |
| `evaluation.py` | Automated checks on a run result: completed, answer present, grounding, schema validity, rating spread, efficiency |
| `eval_runs.py` | Real evaluation cases: a normal request, a very narrow one, an off-topic one |
| `graph_agent.py` | The LangGraph port |
| `compare_agents.py` | Runs the same scenarios on both implementations and diffs the histories |

## Design notes

- **The history is the state.** The Gemini Interactions API is used with `store=False`, so the conversation is built and resent by this code rather than held server-side.
- **Descriptions guide, code enforces.** Argument ranges, enum values and batch sizes are described in the schema *and* checked in code.
- **Tool errors are prompts.** Failures come back to the model as messages that say whether it caused the problem (fix the arguments and retry) or not (don't retry, work with what you have).
- **Model-call errors are the code's problem.** Rate limits, 5xx and network errors are retried with backoff; a daily quota or a 4xx is not. When a call fails for good, the code writes a partial answer from the history rather than asking the model to do it.
- **Grounding is enforced in three places:** a system-prompt rule, a guardrail that refuses notes for papers never fetched, and an automated check that every cited arXiv ID came from a tool result.
- **The loop is testable without a model.** `call_model` and `tools` are injected, so scenarios like "the model never stops" or "arXiv is down" can be tested in milliseconds, for free.

## Evaluation

`evaluation.py` checks any run result; `eval_runs.py` runs three real cases:

| Case | Expected behaviour | Result |
|---|---|---|
| Normal topic with a focus | Searches, fetches, saves notes, cites only real results | 7/7 checks |
| Very narrow topic | Broadens the search, and says plainly when nothing matches | 6/6 checks |
| Off-topic request | Declines, with no tool calls and no invented papers | 7/7 checks |

Worth saying explicitly: agent behaviour is a **distribution, not a value**. Two runs of this code on the same request produced 5 notes and 0 notes respectively, differing only in the model's choices. Judge an agent over several runs, not one.

## Known limitations

- Title-only search is precise but brittle across word forms: "detector" and "detection" return very different result sets.
- The model sometimes puts focus words ("lightweight", "explainable") into the search query, where they rarely appear in titles, and wastes iterations.
- Nothing caps the number of searches, so a thrashing run can exhaust the iteration limit.
- Notes are written from abstracts only, never from full texts.
