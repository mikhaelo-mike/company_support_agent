"""Tests for the company knowledge base."""

from pathlib import Path

# Define the location of the company knowledge document.
KNOWLEDGE_BASE_PATH = Path("data/company_faq.txt")


def test_knowledge_base_exists():
    """Verify that the company knowledge document exists."""

    assert KNOWLEDGE_BASE_PATH.exists()


def test_knowledge_base_contains_delivery_policy():
    """Verify that delivery information is available."""

    content = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")

    assert "Standard delivery takes 3-5 business days" in content


def test_knowledge_base_contains_return_policy():
    """Verify that return information is available."""

    content = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")

    assert "within 7 days" in content


def test_knowledge_base_contains_customer_support():
    """Verify that customer support information is available."""

    content = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")

    assert "support@technova.com" in content