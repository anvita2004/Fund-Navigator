from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agent.nodes import classify_intent, answer_lookup, answer_comparison, answer_calculation


class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    result: Optional[dict]


def route_node(state: AgentState) -> AgentState:
    intent = classify_intent(state["query"])
    return {**state, "intent": intent}


def lookup_node(state: AgentState) -> AgentState:
    result = answer_lookup(state["query"])
    return {**state, "result": result.model_dump()}


def comparison_node(state: AgentState) -> AgentState:
    result = answer_comparison(state["query"])
    return {**state, "result": result.model_dump()}


def calculation_node(state: AgentState) -> AgentState:
    result = answer_calculation(state["query"])
    return {**state, "result": result.model_dump()}


def decide_next(state: AgentState) -> str:
    intent = state["intent"]
    if "comparison" in intent:
        return "comparison"
    elif "calculation" in intent:
        return "calculation"
    else:
        return "lookup"


workflow = StateGraph(AgentState)

workflow.add_node("route", route_node)
workflow.add_node("lookup", lookup_node)
workflow.add_node("comparison", comparison_node)
workflow.add_node("calculation", calculation_node)

workflow.set_entry_point("route")

workflow.add_conditional_edges(
    "route",
    decide_next,
    {
        "lookup": "lookup",
        "comparison": "comparison",
        "calculation": "calculation",
    },
)

workflow.add_edge("lookup", END)
workflow.add_edge("comparison", END)
workflow.add_edge("calculation", END)

app_graph = workflow.compile()
