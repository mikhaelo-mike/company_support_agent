"""ChromaDB vector store management."""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from company_support_agent.config import get_settings
from company_support_agent.rag.embeddings import get_embeddings

COLLECTION_NAME = "technova_knowledge"


def generate_document_ids(
    documents: list[Document],
) -> list[str]:
    """Generate stable IDs for documents."""

    document_ids = []

    for document in documents:
        content = (
            document.metadata.get("source", "")
            + document.page_content
        )

        document_id = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        document_ids.append(document_id)

    return document_ids


def create_vector_store(
    documents: list[Document],
    embeddings: Embeddings | None = None,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Create or update the persistent ChromaDB vector store."""

    if not documents:
        raise ValueError("Cannot create a vector store without documents.")

    config = get_settings()

    embedding_model = embeddings or get_embeddings()

    database_path = Path(
        persist_directory or config.chroma_db_path
    )

    database_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(database_path),
    )

    document_ids = generate_document_ids(documents)

    vector_store.add_documents(
        documents=documents,
        ids=document_ids,
    )

    return vector_store