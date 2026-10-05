"""The same agent, orchestrated by LangGraph instead of a hand-written loop."""
import operator
import sys
from types import SimpleNamespace
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import agent
from resilience import ModelCallFailed, call_with_retries, fallback_answer, partial_summary


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    """Annotated[..., operator.add] = nodes CONTRIBUTE to it (append / accumulate).
    Plain fields = nodes DECIDE it (last write wins)."""
    history: Annotated[list, operator.add]
    total_tokens: Annotated[int, operator.add]
    model_calls: Annotated[int, operator.add]
    iterations: Annotated[int, operator.add]
    pending_calls: list          # function_call steps from the last response
    hit_limit: bool
    answer: Optional[str]
    error: Optional[dict]


def _as_step(call):
    """A history dict -> an object with .name/.arguments/.id, which agent.py expects."""
    return SimpleNamespace(type="function_call", name=call.get("name"),
                           arguments=call.get("arguments") or {}, id=call.get("id"))


# ============================================================
# GRAPH
# ============================================================

def build_graph(call_model, tools, max_iterations):

    def call_model_node(state: AgentState):
        iteration = state["iterations"] + 1
        try:
            interaction = call_with_retries(call_model, state["history"])
        except ModelCallFailed as error:
            print(f"[iter {iteration}] model call failed for good: {error.kind}")
            return {"answer": fallback_answer(error, state["history"]), "pending_calls": [],
                    "error": {"kind": error.kind, "message": error.message,
                              "attempts": error.attempts}}

        tokens = interaction.usage.total_tokens
        steps = [step.model_dump() for step in interaction.steps]
        calls = [step for step in steps if step.get("type") == "function_call"]

        if not calls:
            print(f"[iter {iteration}] final answer | tokens: {tokens}")
        for call in calls:
            args = ", ".join(f"{k}={v!r}" for k, v in (call.get("arguments") or {}).items())
            print(f"[iter {iteration}] called: {call['name']}({args}) | tokens: {tokens}")

        return {
            "history": steps,
            "total_tokens": tokens,
            "model_calls": 1,
            "iterations": 1,
            "pending_calls": calls,
            "answer": None if calls else interaction.output_text,
        }

    def run_tools_node(state: AgentState):
        # append_function_result appends to the list it's given, so hand it a
        # fresh one: in a graph a node returns its updates, it never mutates state.
        new_items = []
        agent.append_function_result(new_items, [_as_step(c) for c in state["pending_calls"]], tools)
        return {"history": new_items, "pending_calls": []}

    def force_answer_node(state: AgentState):
        print(f"[iter {state['iterations']}] max iterations reached, forcing final answer")
        try:
            interaction = call_with_retries(call_model, state["history"], last_iter=True)
        except ModelCallFailed as error:
            return {"answer": fallback_answer(error, state["history"]), "hit_limit": True,
                    "error": {"kind": error.kind, "message": error.message,
                              "attempts": error.attempts}}
        answer = interaction.output_text or partial_summary(
            state["history"], "The model returned no final answer.")
        return {"answer": answer, "hit_limit": True,
                "total_tokens": interaction.usage.total_tokens, "model_calls": 1}

    def after_model(state: AgentState):
        """The model answered (or failed) -> stop. It asked for tools -> run them."""
        if state.get("error") or not state["pending_calls"]:
            return END
        return "run_tools"

    def after_tools(state: AgentState):
        """Out of iterations -> one wrap-up call; otherwise back to the model.
        Tools run BEFORE the wrap-up, so the last results reach the model."""
        return "force_answer" if state["iterations"] >= max_iterations else "call_model"

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model_node)
    graph.add_node("run_tools", run_tools_node)
    graph.add_node("force_answer", force_answer_node)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", after_model, {"run_tools": "run_tools", END: END})
    graph.add_conditional_edges("run_tools", after_tools,
                                {"call_model": "call_model", "force_answer": "force_answer"})
    graph.add_edge("force_answer", END)
    return graph.compile()


def run_graph_agent(user_request, call_model, tools=None, max_iterations=5):
    """Same signature and same result dict as agent.run_agent."""
    tools = tools if tools is not None else agent.function_dispatch
    history = []
    agent.add_user_message(user_request, history)

    compiled = build_graph(call_model, tools, max_iterations)
    final_state = compiled.invoke(
        {"history": history, "total_tokens": 0, "model_calls": 0, "iterations": 0,
         "pending_calls": [], "hit_limit": False, "answer": None, "error": None},
        # Every node visit counts as one step, so leave room: our own counter is the real cap.
        config={"recursion_limit": max_iterations * 3 + 10},
    )
    return {
        "answer": final_state["answer"],
        "hit_limit": final_state["hit_limit"],
        "model_calls": final_state["model_calls"],
        "total_tokens": final_state["total_tokens"],
        "history": final_state["history"],
        "error": final_state["error"],
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import compare_agents
        compare_agents.main()
    else:
        result = run_graph_agent(
            "Find recent papers on detecting diffusion-generated images, focusing on "
            "lightweight or explainable detectors. Read the most relevant ones and save notes on them.",
            agent.real_call_model, max_iterations=6)
        print("\nFinal output:\n" + (result["answer"] or ""))
        print(f"\nModel calls: {result['model_calls']} | Hit limit: {result['hit_limit']} | "
              f"Total tokens: {result['total_tokens']}")
