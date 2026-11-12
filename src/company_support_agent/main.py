"""Application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from company_support_agent.agents.graph import build_support_graph
from company_support_agent.agents.nodes import get_chat_model
from company_support_agent.api.routes import router
from company_support_agent.config import get_settings
from company_support_agent.rag.ingestion import (
    load_document,
    split_document,
)
from company_support_agent.rag.vector_store import (
    create_vector_store,
)


def initialize_agent():
    """Initialize the RAG components and LangGraph agent."""

    settings = get_settings()

    knowledge_base_path = Path("data/company_faq.txt")

    document = load_document(knowledge_base_path)

    chunks = split_document(document)

    vector_store = create_vector_store(
        documents=chunks,
    )

    llm = get_chat_model(settings)

    return build_support_graph(
        vector_store=vector_store,
        llm=llm,
    )


def create_app(support_graph=None) -> FastAPI:
    """Create the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize application resources during startup."""

        if app.state.support_graph is None:
            app.state.support_graph = initialize_agent()

        yield

    app = FastAPI(
        title="TechNova Customer Support Agent",
        description=(
            "An AI-powered customer support API "
            "using RAG, Gemini, and LangGraph."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.support_graph = support_graph

    app.include_router(router)

    return app


app = create_app()