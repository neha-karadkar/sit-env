import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator
from pathlib import Path

import openai

from config import Config

# =========================
# CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are a professional Python code generation assistant. Your role is to interpret user requirements and generate high-quality, well-structured Python code that fulfills the specified task. Always follow Python best practices, ensure code safety, and provide clear, concise solutions. If the requirement is unclear, request clarification. If the request is unsafe or unethical, politely refuse. Present your code in a readable format and include brief explanations if necessary. If you cannot fulfill the request, respond with an appropriate fallback message."
)
OUTPUT_FORMAT = (
    "- Output only the Python code in a properly formatted code block.\n\n"
    "- If an explanation is needed, provide it before the code.\n\n"
    "- If the requirement is unclear, ask for clarification.\n\n"
    "- If the request is unsafe or cannot be fulfilled, provide a polite refusal message."
)
FALLBACK_RESPONSE = (
    "I'm sorry, I cannot generate the requested code. Please provide a clear and safe requirement."
)
FEW_SHOT_EXAMPLES = [
    "Write a Python function to calculate the factorial of a number.",
    "Generate code to read a CSV file and print each row."
]
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# LLM Output Sanitizer
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")


def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()


@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# Input/Output Models
# =========================

class GenerateCodeRequest(BaseModel):
    user_requirement: str = Field(..., description="Natural language Python code requirement")
    output_format: Optional[str] = Field("markdown", description="Output format: 'text' or 'markdown'")

    @field_validator("user_requirement")
    @classmethod
    def validate_requirement_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Requirement must not be empty.")
        if len(v) > 50000:
            raise ValueError("Requirement is too long (max 50,000 characters).")
        return v.strip()

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: Optional[str]) -> str:
        if v is None:
            return "markdown"
        v = v.strip().lower()
        if v not in ("text", "markdown"):
            raise ValueError("output_format must be 'text' or 'markdown'.")
        return v

class GenerateCodeResponse(BaseModel):
    success: bool
    code: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    fixing_tips: Optional[str] = None

# =========================
# SecurityComplianceManager (Stub)
# =========================

class SecurityComplianceManager:
    """
    Handles authentication and audit logging.
    NOTE: Authentication is handled externally; this is a stub for audit logging.
    """
    def authenticate(self, token: str) -> bool:
        # Authentication is handled by API gateway/infrastructure.
        return True

    def log_event(self, event: dict) -> None:
        try:
            logging.info(f"Audit Event: {json.dumps(event, default=str)}")
        except Exception as e:
            logging.warning(f"Failed to log audit event: {e}")

# =========================
# RequirementValidator
# =========================

class RequirementValidator:
    """
    Validates user requirements for clarity and safety.
    """
    def __init__(self):
        pass

    def validate(self, requirement: str) -> (bool, Optional[str]):
        """
        Returns (is_valid, message). If not valid, message is error or clarification.
        """
        # Simple heuristics for ambiguity/unsafe detection.
        ambiguous_phrases = [
            "do something", "etc", "and so on", "as needed", "as appropriate", "whatever", "anything", "something", "thing"
        ]
        unsafe_keywords = [
            "delete all files", "format disk", "shutdown", "os.remove", "os.system('rm", "subprocess", "eval(", "exec(", "drop database", "kill process"
        ]
        req_lower = requirement.lower()
        for phrase in ambiguous_phrases:
            if phrase in req_lower:
                return False, "Could you please clarify or provide more details about your Python code requirement?"
        for unsafe in unsafe_keywords:
            if unsafe in req_lower:
                return False, "I'm sorry, I cannot generate code that is unsafe or unethical."
        # Additional: block empty or too short
        if len(requirement.strip()) < 5:
            return False, "Could you please clarify or provide more details about your Python code requirement?"
        return True, None

# =========================
# PythonCodeFormatter
# =========================

class PythonCodeFormatter:
    """
    Formats Python code for readability and PEP8 compliance.
    """
    def __init__(self):
        pass

    def format_code(self, code: str) -> str:
        """
        Attempts to format code using black, then autopep8. Returns original code if formatting fails.
        """
        formatted = code
        try:
            import black
            mode = black.Mode()
            formatted = black.format_str(code, mode=mode)
            return formatted
        except Exception:
            pass
        try:
            import autopep8
            formatted = autopep8.fix_code(code)
            return formatted
        except Exception:
            pass
        return code

# =========================
# LLMService
# =========================

class LLMService:
    """
    Handles interaction with Azure OpenAI GPT-4.1.
    """
    def __init__(self):
        self._client = None

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_client(self):
        if self._client is not None:
            return self._client
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not configured")
        self._client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_code(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Calls Azure OpenAI GPT-4.1 with constructed prompt and context.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT},
        ]
        # Add few-shot examples as user/assistant pairs
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": self._example_completion(example)})
        # User requirement
        messages.append({"role": "user", "content": prompt})
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        try:
            client = self.get_llm_client()
            response = await client.chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4.1",
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return content
        except Exception as e:
            logging.warning(f"LLM call failed: {e}")
            return FALLBACK_RESPONSE

    def _example_completion(self, example: str) -> str:
        # Provide a minimal completion for few-shot examples.
        if "factorial" in example.lower():
            return (
                "def factorial(n):\n"
                "    if n == 0:\n"
                "        return 1\n"
                "    else:\n"
                "        return n * factorial(n-1)"
            )
        if "csv" in example.lower():
            return (
                "import csv\n"
                "with open('file.csv', newline='') as csvfile:\n"
                "    reader = csv.reader(csvfile)\n"
                "    for row in reader:\n"
                "        print(row)"
            )
        return "# Python code"

# =========================
# AgentOrchestrator
# =========================

class AgentOrchestrator:
    """
    Coordinates the flow: receives user input, validates requirements, manages clarification,
    invokes LLMService, applies formatting, and returns responses.
    """
    def __init__(self):
        self.llm_service = LLMService()
        self.validator = RequirementValidator()
        self.formatter = PythonCodeFormatter()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_user_request(self, user_requirement: str, output_format: str = "markdown") -> Dict[str, Any]:
        """
        Main entry point for handling user requests; validates, clarifies, generates, formats, and returns code.
        """
        async with trace_step(
            "validate_requirement",
            step_type="parse",
            decision_summary="Validate user requirement for clarity and safety",
            output_fn=lambda r: f"valid={r.get('success', '?')}",
        ) as step:
            is_valid, message = self.validator.validate(user_requirement)
            if not is_valid:
                step.capture({"success": False, "error": message})
                return {
                    "success": False,
                    "code": None,
                    "explanation": None,
                    "error": message,
                    "error_type": "INVALID_REQUIREMENT",
                    "fixing_tips": "Please provide a clear, specific, and safe Python code requirement."
                }
            step.capture({"success": True})

        async with trace_step(
            "generate_code",
            step_type="llm_call",
            decision_summary="Call LLM to generate Python code",
            output_fn=lambda r: f"code_len={len(r.get('code','')) if r.get('code') else 0}",
        ) as step:
            llm_response = await self.llm_service.generate_code(user_requirement, context={})
            llm_response_clean = sanitize_llm_output(llm_response, content_type="code")
            if llm_response_clean.strip() == FALLBACK_RESPONSE:
                step.capture({"success": False, "error": FALLBACK_RESPONSE})
                return {
                    "success": False,
                    "code": None,
                    "explanation": None,
                    "error": FALLBACK_RESPONSE,
                    "error_type": "CODE_GENERATION_ERROR",
                    "fixing_tips": "Try rephrasing your requirement or provide more details."
                }
            # Split explanation and code if present
            explanation, code = self._split_explanation_and_code(llm_response_clean)
            step.capture({"success": True, "code": code})

        async with trace_step(
            "format_code",
            step_type="process",
            decision_summary="Format Python code for readability and PEP8 compliance",
            output_fn=lambda r: f"formatted_len={len(r) if r else 0}",
        ) as step:
            formatted_code = self.formatter.format_code(code)
            step.capture(formatted_code)

        # Output formatting
        if output_format == "markdown":
            formatted_code_block = f"```python\n{formatted_code.strip()}\n```"
        else:
            formatted_code_block = formatted_code.strip()

        return {
            "success": True,
            "code": formatted_code_block,
            "explanation": explanation,
            "error": None,
            "error_type": None,
            "fixing_tips": None
        }

    def _split_explanation_and_code(self, llm_response: str) -> (Optional[str], str):
        """
        Splits explanation and code if both are present.
        """
        # Look for code block
        code_block = _FENCE_RE.search(llm_response)
        if code_block:
            code = code_block.group(1).strip()
            explanation = llm_response[:code_block.start()].strip()
            if explanation:
                return explanation, code
            return None, code
        # If no code block, treat all as code
        return None, llm_response.strip()

# =========================
# Main Agent Class
# =========================

class PythonCodeGenerationAgent:
    """
    Main agent class. Composes AgentOrchestrator.
    """
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.security_manager = SecurityComplianceManager()

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def run(self, user_requirement: str, output_format: str = "markdown") -> Dict[str, Any]:
        """
        Main agent entry point.
        """
        async with trace_step(
            "process_user_request",
            step_type="final",
            decision_summary="Process user request end-to-end",
            output_fn=lambda r: f"success={r.get('success', '?')}",
        ) as step:
            result = await self.orchestrator.process_user_request(user_requirement, output_format)
            step.capture(result)
            self.security_manager.log_event({
                "event": "user_request_processed",
                "requirement": user_requirement,
                "output_format": output_format,
                "success": result.get("success"),
                "error": result.get("error"),
                "timestamp": _time.time()
            })
            return result

# =========================
# FastAPI App & Endpoints
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Python Code Generation Assistant",
    description="Professional Python code generation assistant. Interprets user requirements and generates high-quality, well-structured Python code.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Malformed JSON or invalid input.",
            "error_type": "VALIDATION_ERROR",
            "fixing_tips": "Check your JSON formatting, required fields, and value types. Ensure quotes and commas are correct.",
            "details": exc.errors(),
        },
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Malformed JSON or invalid input.",
            "error_type": "VALIDATION_ERROR",
            "fixing_tips": "Check your JSON formatting, required fields, and value types. Ensure quotes and commas are correct.",
            "details": exc.errors(),
        },
    )

@app.post("/generate", response_model=GenerateCodeResponse)
async def generate_code_endpoint(req: GenerateCodeRequest):
    """
    Generate Python code from user requirement.
    """
    agent = PythonCodeGenerationAgent()
    try:
        result = await agent.run(
            user_requirement=req.user_requirement,
            output_format=req.output_format
        )
        # Sanitize LLM output before returning
        if result.get("code"):
            result["code"] = sanitize_llm_output(result["code"], content_type="code")
        if result.get("explanation"):
            result["explanation"] = sanitize_llm_output(result["explanation"], content_type="text")
        return result
    except Exception as e:
        logging.exception("Unhandled error in /generate endpoint")
        return GenerateCodeResponse(
            success=False,
            code=None,
            explanation=None,
            error="An unexpected error occurred.",
            error_type="UNHANDLED_ERROR",
            fixing_tips="Please try again later or contact support."
        )

# =========================
# Entrypoint
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())