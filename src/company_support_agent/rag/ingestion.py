"""Document loading and text chunking utilities."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path: Path) -> Document:
    """Load a text file into a LangChain Document."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Knowledge document not found: {file_path}"
        )

    content = file_path.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError("Knowledge document is empty.")

    return Document(
        page_content=content,
        metadata={
            "source": file_path.name,
            "company": "TechNova",
        },
    )


def split_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split a document into smaller overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents([document])

    return chunks