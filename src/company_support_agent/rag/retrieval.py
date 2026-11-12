"""Document retrieval and context formatting."""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def retrieve_documents(
    query: str,
    vector_store: VectorStore,
    k: int = 3,
) -> list[Document]:
    """Retrieve the most relevant documents for a question."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k <= 0:
        raise ValueError("Number of retrieved documents must be positive.")

    documents = vector_store.similarity_search(
        query=query,
        k=k,
    )

    return documents


def format_context(
    documents: list[Document],
) -> str:
    """Format retrieved documents into context for an AI agent."""

    if not documents:
        return ""

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n---\n\n".join(formatted_documents)