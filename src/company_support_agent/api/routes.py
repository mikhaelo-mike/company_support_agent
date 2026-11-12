"""Customer support API routes."""

import logging

from fastapi import APIRouter, HTTPException, Request

from company_support_agent.api.schemas import (
    SupportRequest,
    SupportResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Customer Support"],
)


@router.post(
    "/support",
    response_model=SupportResponse,
)
def ask_support(
    payload: SupportRequest,
    request: Request,
) -> SupportResponse:
    """Process a customer question using the support agent."""

    support_graph = getattr(
        request.app.state,
        "support_graph",
        None,
    )

    if support_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Customer support service is unavailable.",
        )

    try:
        result = support_graph.invoke(
            {
                "question": payload.question,
                "retrieved_documents": [],
                "context": "",
                "answer": "",
            }
        )

        sources = list(
            dict.fromkeys(
                document.metadata.get(
                    "source",
                    "Unknown source",
                )
                for document in result.get(
                    "retrieved_documents",
                    []
                )
            )
        )

        return SupportResponse(
            answer=result["answer"],
            sources=sources,
        )

    except Exception:
        logger.exception(
            "Customer support agent failed to process a request."
        )

        raise HTTPException(
            status_code=503,
            detail="Customer support is temporarily unavailable.",
        ) from None


@router.get("/health")
def health_check(request: Request) -> dict:
    """Check whether the support agent is available."""

    support_graph = getattr(
        request.app.state,
        "support_graph",
        None,
    )

    return {
        "status": "ok" if support_graph is not None else "degraded",
        "agent_available": support_graph is not None,
    }