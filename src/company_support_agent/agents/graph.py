"""LangGraph workflow for the customer support agent."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from company_support_agent.agents.nodes import (
    generate_answer_node,
    retrieve_node,
)
from company_support_agent.agents.state import SupportAgentState


def build_support_graph(
    vector_store: VectorStore,
    llm: BaseChatModel,
):
    """Build and compile the customer support workflow."""

    builder = StateGraph(SupportAgentState)

    def retrieval_step(state: SupportAgentState) -> dict:
        """Execute document retrieval."""

        return retrieve_node(
            state=state,
            vector_store=vector_store,
        )

    def generation_step(state: SupportAgentState) -> dict:
        """Execute answer generation."""

        return generate_answer_node(
            state=state,
            llm=llm,
        )

    builder.add_node(
        "retrieve",
        retrieval_step,
    )

    builder.add_node(
        "generate",
        generation_step,
    )

    builder.add_edge(START, "retrieve")

    builder.add_edge("retrieve", "generate")

    builder.add_edge("generate", END)

    return builder.compile()