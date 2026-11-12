"""State definition for the customer support agent."""

from typing import TypedDict

from langchain_core.documents import Document


class SupportAgentState(TypedDict):
    """Define the data shared between agent nodes."""

    question: str

    retrieved_documents: list[Document]

    context: str

    answer: str