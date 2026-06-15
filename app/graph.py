# app/graph.py — сборка графа на LangGraph.
#
# Структура (как в плане, раздел 6):
#   START → classify → route → info / transactional / escalation / rejection → answer → END
#
# route — это conditional edge: по результату classify (intent) LangGraph выбирает,
# в какой из четырёх узлов пойти. Все четыре сходятся в answer, затем END.

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from . import nodes


def build_graph():
    """Собирает и компилирует граф агента."""
    g = StateGraph(AgentState)

    # регистрируем узлы
    g.add_node("classify", nodes.classify_node)
    g.add_node("info", nodes.info_node)
    g.add_node("transactional", nodes.transactional_node)
    g.add_node("escalation", nodes.escalation_node)
    g.add_node("rejection", nodes.rejection_node)
    g.add_node("answer", nodes.answer_node)

    # вход → классификация
    g.add_edge(START, "classify")

    # из classify — условный переход по intent в один из четырёх узлов
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

    # все рабочие узлы сходятся в финальную сборку, затем END
    for node_name in ("info", "transactional", "escalation", "rejection"):
        g.add_edge(node_name, "answer")
    g.add_edge("answer", END)

    return g.compile()


# Компилируем один раз при импорте — переиспользуем готовый граф.
GRAPH = build_graph()
