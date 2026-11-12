"""Pydantic schemas for the customer support API."""

from pydantic import BaseModel, Field, field_validator


class SupportRequest(BaseModel):
    """Validate incoming customer support questions."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Customer support question.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject questions containing only whitespace."""

        cleaned_question = value.strip()

        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        return cleaned_question


class SupportResponse(BaseModel):
    """Define the customer support API response."""

    answer: str

    sources: list[str]