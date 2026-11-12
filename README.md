# TechNova AI Customer Support Agent

An AI-powered customer support application built using
Retrieval-Augmented Generation (RAG), LangGraph, Gemini,
ChromaDB, and FastAPI.

## Problem Statement

Customer support teams frequently handle repetitive questions
about deliveries, returns, refunds, and order cancellations.

TechNova AI Customer Support Agent automates responses to
these questions using the company's internal knowledge base.

## Architecture

Customer
   |
   v
FastAPI
   |
   v
LangGraph Agent
   |
   v
RAG Retrieval
   |
   v
ChromaDB
   |
   v
Gemini LLM
   |
   v
Grounded Response

## Technologies

- Python 3.11
- UV
- FastAPI
- LangChain
- LangGraph
- Google Gemini
- ChromaDB
- Docker
- Pytest
- Ruff

## Features

- Document ingestion and chunking
- Vector embeddings
- Semantic document retrieval
- AI-generated customer support responses
- LangGraph workflow orchestration
- REST API
- Request validation
- Error handling
- Health monitoring
- Docker containerization
- Automated tests

## Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate into the project:

cd company-support-agent

Create the environment:

uv sync

Configure environment variables using .env.example.

Add your Gemini API key to .env.

Start the application:

uv run uvicorn company_support_agent.main:app \
    --reload --app-dir src

## API Documentation

Visit:

http://localhost:8000/docs

## Run Tests

uv run pytest -v

## Code Quality

uv run ruff check .

## Docker

Build:

docker compose build

Start:

docker compose up -d

## Future Improvements

- Authentication and authorization
- Conversation history
- Customer feedback collection
- Monitoring and observability
- Rate limiting
- Multi-tenant document isolation

## Author

AI Engineering Portfolio Project