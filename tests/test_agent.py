"""Tests for the LangGraph customer support agent."""

import hashlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from company_support_agent.agents.graph import build_support_graph

#from company_support_agent.agents.graph import build_support_graph
from company_support_agent.agents.nodes import generate_answer_node

from company_support_agent.rag.vector_store import create_vector_store


class FakeEmbeddings(Embeddings):
    """Provide deterministic embeddings for tests."""

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed multiple documents."""

        return [
            self.embed_query(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Embed a single query."""

        digest = hashlib.sha256(
            text.encode("utf-8")
        ).digest()

        return [
            value / 255
            for value in digest[:8]
        ]


class FakeChatModel(BaseChatModel):
    """Simulate Gemini responses."""

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ):
        """Return a predictable AI response."""

        from langchain_core.outputs import (
            ChatGeneration,
            ChatResult,
        )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=(
                            "TechNova allows customers "
                            "to return products within 7 days."
                        )
                    )
                )
            ]
        )


@pytest.fixture
def support_graph(tmp_path):
    """Create an isolated customer support graph."""

    documents = [
        Document(
            page_content=(
                "Customers can return products within 7 days."
            ),
            metadata={
                "source": "returns.txt",
                "company": "TechNova",
            },
        ),
        Document(
            page_content=(
                "Standard delivery takes 3-5 business days."
            ),
            metadata={
                "source": "delivery.txt",
                "company": "TechNova",
            },
        ),
    ]

    vector_store = create_vector_store(
        documents=documents,
        embeddings=FakeEmbeddings(),
        persist_directory=tmp_path / "agent_test",
    )

    return build_support_graph(
        vector_store=vector_store,
        llm=FakeChatModel(),
    )


def test_agent_returns_answer(support_graph):
    """Test the complete agent workflow."""

    result = support_graph.invoke(
        {
            "question": "Can I return a product after 5 days?",
            "retrieved_documents": [],
            "context": "",
            "answer": "",
        }
    )

    assert result["answer"]

    assert "7 days" in result["answer"]

    assert len(result["retrieved_documents"]) > 0


# def test_agent_returns_fallback_without_context(support_graph):
#     """Test the agent fallback when retrieval finds no documents."""

#     result = support_graph.invoke(
#         {
#             "question": "",
#             "retrieved_documents": [],
#             "context": "",
#             "answer": "",
#         }
#     )

#     assert "couldn't find relevant information" in result["answer"]
def test_agent_returns_fallback_without_context():
    """Test fallback generation when no context is available."""

    state = {
        "question": "Does TechNova offer international delivery?",
        "retrieved_documents": [],
        "context": "",
        "answer": "",
    }

    result = generate_answer_node(
        state=state,
        llm=FakeChatModel(),
    )

    assert "couldn't find relevant information" in result["answer"]
    assert "support@technova.com" in result["answer"]