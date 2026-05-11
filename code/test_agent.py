
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

import agent

from agent import (
    AgentOrchestrator,
    LLMService,
    PythonCodeFormatter,
    SecurityComplianceManager,
    GenerateCodeRequest,
    GenerateCodeResponse,
    FALLBACK_RESPONSE,
    app
)

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError

@pytest.fixture
def orchestrator():
    return AgentOrchestrator()

@pytest.fixture
def formatter():
    return PythonCodeFormatter()

@pytest.fixture
def llm_service():
    return LLMService()

@pytest.fixture
def security_manager():
    return SecurityComplianceManager()

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_valid_requirement_generates_python_code(test_client):
    """
    Functional: Valid requirement returns code and no error.
    """
    # Patch LLMService.generate_code to return a valid Python function
    code_snippet = "def add(a, b):\n    return a + b"
    with patch.object(agent.LLMService, "generate_code", new=AsyncMock(return_value=code_snippet)):
        payload = {
            "user_requirement": "Write a function to add two numbers",
            "output_format": "markdown"
        }
        response = test_client.post("/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "def" in data["code"]
        assert data["error"] is None
        assert data["error_type"] is None

@pytest.mark.asyncio
async def test_ambiguous_requirement_triggers_clarification(orchestrator):
    """
    Unit: Ambiguous requirement triggers clarification.
    """
    result = await orchestrator.process_user_request(
        user_requirement="Do something",
        output_format="markdown"
    )
    assert result["success"] is False
    assert "clarify" in result["error"].lower()
    assert result["error_type"] == "INVALID_REQUIREMENT"

@pytest.mark.asyncio
async def test_unsafe_requirement_is_blocked(orchestrator):
    """
    Unit: Unsafe requirement is blocked with polite refusal.
    """
    result = await orchestrator.process_user_request(
        user_requirement="Delete all files on disk",
        output_format="text"
    )
    assert result["success"] is False
    assert "cannot generate code" in result["error"].lower()
    assert result["error_type"] == "INVALID_REQUIREMENT"

@pytest.mark.asyncio
async def test_llmservice_handles_api_failure_gracefully(llm_service):
    """
    Unit: LLMService.generate_code returns fallback on API failure.
    """
    with patch.object(llm_service, "get_llm_client", side_effect=Exception("API Down")):
        output = await llm_service.generate_code(prompt="Write a function", context={})
        assert output == FALLBACK_RESPONSE

def test_python_code_formatter_returns_original_on_failure(formatter):
    """
    Unit: PythonCodeFormatter returns original code if formatting fails.
    """
    # Patch both black and autopep8 to raise ImportError
    with patch("agent.black", side_effect=ImportError()), patch("agent.autopep8", side_effect=ImportError()):
        code = "def foo(:"
        output = formatter.format_code(code)
        assert output == code

def test_generate_code_request_model_validation():
    """
    Unit: GenerateCodeRequest enforces required fields and value constraints.
    """
    # Empty user_requirement
    with pytest.raises(ValidationError):
        agent.GenerateCodeRequest(user_requirement="", output_format="markdown")
    # Invalid output_format
    with pytest.raises(ValidationError):
        agent.GenerateCodeRequest(user_requirement="Valid", output_format="invalid")

def test_security_compliance_manager_log_event_handles_logging_errors(security_manager):
    """
    Unit: SecurityComplianceManager.log_event does not raise if logging fails.
    """
    with patch("agent.logging.info", side_effect=Exception("Logging failed")):
        try:
            security_manager.log_event({"event": object()})
        except Exception:
            pytest.fail("log_event should not propagate exceptions")

@pytest.mark.asyncio
async def test_health_endpoint_returns_ok():
    """
    Functional: /health endpoint returns status ok.
    """
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_output_formatting_as_markdown(orchestrator):
    """
    Unit: output_format='markdown' wraps code in markdown code block.
    """
    # Patch LLMService.generate_code to return a code snippet
    code_snippet = "def add(a, b):\n    return a + b"
    with patch.object(agent.LLMService, "generate_code", new=AsyncMock(return_value=code_snippet)):
        result = await orchestrator.process_user_request(
            user_requirement="Write a function to add two numbers",
            output_format="markdown"
        )
        assert result["code"].startswith("```python")

@pytest.mark.asyncio
async def test_output_formatting_as_plain_text(orchestrator):
    """
    Unit: output_format='text' returns code as plain text (no markdown block).
    """
    code_snippet = "def add(a, b):\n    return a + b"
    with patch.object(agent.LLMService, "generate_code", new=AsyncMock(return_value=code_snippet)):
        result = await orchestrator.process_user_request(
            user_requirement="Write a function to add two numbers",
            output_format="text"
        )
        assert "```" not in result["code"]