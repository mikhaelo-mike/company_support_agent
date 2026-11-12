"""Tests for the RAG retrieval pipeline."""

import hashlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from company_support_agent.rag.retrieval import (
    format_context,
    retrieve_documents,
)
from company_support_agent.rag.vector_store import (
    create_vector_store,
)


class FakeEmbeddings(Embeddings):
    """Generate deterministic embeddings for testing."""

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents."""

        return [
            self.embed_query(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for a single query."""

        digest = hashlib.sha256(
            text.encode("utf-8")
        ).digest()

        return [
            value / 255
            for value in digest[:8]
        ]


@pytest.fixture
def vector_store(tmp_path):
    """Create a temporary ChromaDB instance for testing."""

    documents = [
        Document(
            page_content="Standard delivery takes 3-5 business days.",
            metadata={"source": "delivery.txt"},
        ),
        Document(
            page_content="Customers can return products within 7 days.",
            metadata={"source": "returns.txt"},
        ),
        Document(
            page_content="Refunds take 5-7 business days.",
            metadata={"source": "refunds.txt"},
        ),
    ]

    return create_vector_store(
        documents=documents,
        embeddings=FakeEmbeddings(),
        persist_directory=tmp_path / "retrieval_test",
    )


def test_retrieve_documents(vector_store):
    """Test retrieving relevant documents."""

    results = retrieve_documents(
        query="How long does delivery take?",
        vector_store=vector_store,
        k=2,
    )

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_retrieve_documents_rejects_empty_query(vector_store):
    """Test that an empty query raises an error."""

    with pytest.raises(ValueError):
        retrieve_documents(
            query="   ",
            vector_store=vector_store,
        )


def test_retrieve_documents_rejects_invalid_k(vector_store):
    """Test that an invalid retrieval count raises an error."""

    with pytest.raises(ValueError):
        retrieve_documents(
            query="Delivery information",
            vector_store=vector_store,
            k=0,
        )


def test_format_context():
    """Test formatting retrieved documents."""

    documents = [
        Document(
            page_content="Standard delivery takes 3-5 business days.",
            metadata={"source": "delivery.txt"},
        )
    ]

    context = format_context(documents)

    assert "Standard delivery takes 3-5 business days." in context
    assert "delivery.txt" in context


def test_format_empty_context():
    """Test formatting when no documents are retrieved."""

    context = format_context([])

    assert context == ""