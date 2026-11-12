# Use Python 3.11 as the application runtime.
FROM python:3.11-slim

# Install the official UV binary.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the application working directory.
WORKDIR /app

# Configure Python and UV.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency definitions.
COPY pyproject.toml uv.lock ./

# Install production dependencies without installing the project yet.
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code and knowledge base.
COPY . .

# Install the application itself.
RUN uv sync --frozen --no-dev

# Make the virtual environment available.
ENV PATH="/app/.venv/bin:$PATH"

# Make application source imports available.
ENV PYTHONPATH="/app/src"

# Expose the FastAPI port.
EXPOSE 8000

# Start the FastAPI application.
CMD ["uvicorn", "company_support_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]