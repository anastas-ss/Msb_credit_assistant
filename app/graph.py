from langgraph.graph import StateGraph, START, END

from .state import AgentState
from . import nodes


def build_graph():
    """Собирает и компилирует граф агента."""
    g = StateGraph(AgentState)

    g.add_node("classify", nodes.classify_node)
    g.add_node("info", nodes.info_node)
    g.add_node("transactional", nodes.transactional_node)
    g.add_node("escalation", nodes.escalation_node)
    g.add_node("rejection", nodes.rejection_node)
    g.add_node("answer", nodes.answer_node)

    g.add_edge(START, "classify")

    g.add_conditional_edges(
        "classify",
        nodes.route_after_classify,
        {
            "info": "info",
            "transactional": "transactional",
            "escalation": "escalation",
            "rejection": "rejection",
        },
    )

    for node_name in ("info", "transactional", "escalation", "rejection"):
        g.add_edge(node_name, "answer")
    g.add_edge("answer", END)

    return g.compile()


GRAPH = build_graph()
