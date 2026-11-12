"""Tests for the customer support API."""

from fastapi.testclient import TestClient

from company_support_agent.main import create_app


class FakeSupportGraph:
    """Simulate the LangGraph customer support workflow."""

    def invoke(self, state: dict) -> dict:
        """Return a predictable customer support response."""

        return {
            "question": state["question"],
            "answer": (
                "Standard delivery takes 3-5 business days."
            ),
            "retrieved_documents": [
                type(
                    "FakeDocument",
                    (),
                    {
                        "metadata": {
                            "source": "company_faq.txt"
                        }
                    },
                )()
            ],
            "context": "Standard delivery takes 3-5 business days.",
        }


def test_support_endpoint():
    """Test a successful customer support request."""

    app = create_app(
        support_graph=FakeSupportGraph(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/support",
            json={
                "question": "How long does delivery take?"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert "3-5 business days" in data["answer"]

    assert "company_faq.txt" in data["sources"]


def test_support_endpoint_rejects_empty_question():
    """Test that an empty customer question is rejected."""

    app = create_app(
        support_graph=FakeSupportGraph(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/support",
            json={
                "question": "   "
            },
        )

    assert response.status_code == 422


def test_support_endpoint_when_agent_unavailable(monkeypatch):
    """Test the API response when the agent is unavailable."""

    # Prevent real Gemini and ChromaDB initialization.
    monkeypatch.setattr(
        "company_support_agent.main.initialize_agent",
        lambda: None,
    )

    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/support",
            json={
                "question": "How long does delivery take?"
            },
        )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Customer support service is unavailable."
    }



class FailingSupportGraph:
    """Simulate an unavailable AI agent."""

    def invoke(self, state: dict) -> dict:
        """Simulate an agent execution failure."""

        raise RuntimeError("Simulated Gemini failure.")



def test_support_endpoint_handles_agent_failure():
    """Test controlled handling of agent execution failures."""

    app = create_app(
        support_graph=FailingSupportGraph(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/support",
            json={
                "question": "How long does delivery take?"
            },
        )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Customer support is temporarily unavailable."
    }


def test_health_endpoint():
    """Test the application health endpoint."""

    app = create_app(
        support_graph=FakeSupportGraph(),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "agent_available": True,
    }    