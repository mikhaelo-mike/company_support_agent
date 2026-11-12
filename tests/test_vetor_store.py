"""Tests for ChromaDB vector store creation."""

import hashlib

import pytest
from langchain_core.embeddings import Embeddings

from company_support_agent.rag.vector_store import (
    create_vector_store,
    generate_document_ids,
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


def test_generate_document_ids():
    """Verify that document IDs are deterministic."""

    from langchain_core.documents import Document

    document = Document(
        page_content="Standard delivery takes 3-5 days.",
        metadata={"source": "company_faq.txt"},
    )

    first_ids = generate_document_ids([document])
    second_ids = generate_document_ids([document])

    assert first_ids == second_ids


def test_create_vector_store(tmp_path):
    """Verify that ChromaDB stores and retrieves documents."""

    from langchain_core.documents import Document

    documents = [
        Document(
            page_content="Standard delivery takes 3-5 business days.",
            metadata={
                "source": "company_faq.txt",
                "company": "TechNova",
            },
        )
    ]

    vector_store = create_vector_store(
        documents=documents,
        embeddings=FakeEmbeddings(),
        persist_directory=tmp_path / "chroma_test",
    )

    results = vector_store.similarity_search(
        "How long does delivery take?",
        k=1,
    )

    assert len(results) == 1

    assert "3-5 business days" in results[0].page_content

    assert results[0].metadata["company"] == "TechNova"


def test_create_vector_store_rejects_empty_documents(tmp_path):
    """Verify that an empty document list is rejected."""

    with pytest.raises(ValueError):
        create_vector_store(
            documents=[],
            embeddings=FakeEmbeddings(),
            persist_directory=tmp_path / "empty_test",
        )