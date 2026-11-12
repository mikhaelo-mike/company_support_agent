"""Gemini embedding model configuration."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from company_support_agent.config import Settings, get_settings


def get_embeddings(
    settings: Settings | None = None,
) -> GoogleGenerativeAIEmbeddings:
    """Create and return the configured Gemini embedding model."""

    config = settings or get_settings()

    if not config.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model=config.embedding_model,
        google_api_key=config.google_api_key,
    )