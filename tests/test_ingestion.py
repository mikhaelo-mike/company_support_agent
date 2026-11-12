"""Tests for document ingestion and chunking."""

from pathlib import Path

import pytest

from company_support_agent.rag.ingestion import (
    load_document,
    split_document,
)


def test_load_document(tmp_path):
    """Test loading a company knowledge document."""

    file_path = tmp_path / "company.txt"

    file_path.write_text(
        "Standard delivery takes 3-5 business days.",
        encoding="utf-8",
    )

    document = load_document(file_path)

    assert document.page_content == (
        "Standard delivery takes 3-5 business days."
    )

    assert document.metadata["company"] == "TechNova"


def test_load_missing_document():
    """Test that a missing document raises an error."""

    file_path = Path("missing_file.txt")

    with pytest.raises(FileNotFoundError):
        load_document(file_path)


def test_load_empty_document(tmp_path):
    """Test that an empty document raises an error."""

    file_path = tmp_path / "empty.txt"

    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(file_path)


def test_split_document():
    """Test splitting a document into smaller chunks."""

    document = load_document(Path("data/company_faq.txt"))

    chunks = split_document(
        document,
        chunk_size=500,
        chunk_overlap=50,
    )

    assert len(chunks) > 1

    assert all(chunk.page_content for chunk in chunks)

    assert all(
        chunk.metadata["company"] == "TechNova"
        for chunk in chunks
    )