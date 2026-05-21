from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    plan: list[str]
    contexts: list[dict[str, Any]]
    answer: str
    evaluation: dict[str, Any]
    requires_human_review: bool


def planner(state: AgentState) -> AgentState:
    return {**state, "plan": ["retrieve", "answer", "evaluate"]}


def retriever_node(state: AgentState) -> AgentState:
    return {**state, "contexts": []}


def answer_generator(state: AgentState) -> AgentState:
    return {**state, "answer": ""}


def evaluator_node(state: AgentState) -> AgentState:
    return {**state, "evaluation": {"status": "not_implemented"}}


def human_review_router(state: AgentState) -> str:
    return "human_review" if state.get("requires_human_review") else "done"


def build_workflow() -> Any:
    """Return a LangGraph workflow once concrete node dependencies are wired."""
    return {
        "planner": planner,
        "retriever": retriever_node,
        "answer_generator": answer_generator,
        "evaluator": evaluator_node,
        "human_review_router": human_review_router,
    }
