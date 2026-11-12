"""Tests for application configuration."""

from company_support_agent.config import Settings


def test_default_settings():
    """Test application default configuration."""

    settings = Settings(_env_file=None)

    assert settings.gemini_model == "gemini-2.5-flash"

    assert settings.chroma_db_path == "chroma_db"


def test_google_api_key_from_environment(monkeypatch):
    """Test loading the Google API key from environment variables."""

    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

    settings = Settings(_env_file=None)

    assert settings.google_api_key == "test-api-key"